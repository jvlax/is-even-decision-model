"""Fine-tune Laya's decision head so the is-even-decision questions actually work.

Head-only by default (encoder frozen); set ENCODER_LAYERS=N to also unfreeze the
top N encoder layers. Trains on the exact question phrasings index.js sends, so
the npm package needs no code change.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from rl_common import QTYPES, build_model, build_sequence, collate_items, load_cfg, render_options, seed_all

seed_all(42)
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
AMP = DEV.type == "cuda"  # fp16 autocast on Turing; MPS stays native
scaler = torch.amp.GradScaler("cuda", enabled=AMP)
CFG = load_cfg("rl_agent_config.json")
TOK = AutoTokenizer.from_pretrained("tokenizer")

# ----- the package's exact questions (index.js) -----
Q_IS_EVEN = {
    "t": "noul",
    "ins": "Is the number even? A number is even when dividing it by 2 leaves no remainder. "
           "Look at the last digit: 0, 2, 4, 6 and 8 are even; 1, 3, 5, 7 and 9 are odd. Negative numbers follow the same rule.",
    "crit": {"true": "the number is even, like 0, 2, 4, 6, 8, 10, 12, 100 or -4",
             "false": "the number is odd, like 1, 3, 5, 7, 9, 11, 13, 101 or -3"},
}
Q_IS_ODD = {
    "t": "noul",
    "ins": "Is the number odd? A number is odd when dividing it by 2 leaves a remainder of 1. "
           "Look at the last digit: 1, 3, 5, 7 and 9 are odd; 0, 2, 4, 6 and 8 are even. Negative numbers follow the same rule.",
    "crit": {"true": "the number is odd, like 1, 3, 5, 7, 9, 11, 13, 101 or -3",
             "false": "the number is even, like 0, 2, 4, 6, 8, 10, 12, 100 or -4"},
}
Q_EQ = {"t": "noul", "ins": "Are a and b the same number? Compare them digit by digit, including the sign.",
        "crit": {"true": "a and b are exactly the same number", "false": "a and b are different numbers"}}
Q_NEQ = {"t": "noul", "ins": "Are a and b different numbers? Compare them digit by digit, including the sign.",
         "crit": {"true": "a and b are different numbers", "false": "a and b are exactly the same number"}}
Q_GT = {"t": "noul", "ins": "Is a strictly greater than b, as numbers on the number line?",
        "crit": {"true": "a > b", "false": "a <= b (a is smaller than b, or they are equal)"}}
Q_LT = {"t": "noul", "ins": "Is a strictly less than b, as numbers on the number line?",
        "crit": {"true": "a < b", "false": "a >= b (a is greater than b, or they are equal)"}}

rng = random.Random(42)


def rand_int(max_digits=21):
    d = rng.choice([1] * 6 + [2] * 6 + [3] * 4 + [4] * 4 + list(range(5, max_digits)))
    lo, hi = (0, 9) if d == 1 else (10 ** (d - 1), 10 ** d - 1)
    n = rng.randint(lo, hi)
    if rng.random() < 0.15:
        n = -n
    return n


def make_items():
    items = []  # (state, q, y)  y in {0,1} -> target index (false=0,true=1) pre-shuffle
    nums = set()
    dense = int(os.environ.get("PARITY_DENSE", "0"))
    if dense:
        nums.update(range(-dense, dense + 1))
    else:
        nums.update(list(range(-210, 211)) + list(range(1900, 2101)))
    while len(nums) < max(len(nums), int(os.environ.get("PARITY_NUMS", "5000"))):
        nums.add(rand_int())
    for n in nums:
        even = 1 if abs(n) % 2 == 0 else 0
        st = {"number": str(n), "last_digit": str(n)[-1]}
        items.append((st, Q_IS_EVEN, even))
        items.append((st, Q_IS_ODD, 1 - even))
    for _ in range(int(os.environ.get("PAIR_ITEMS", "1100"))):
        a = rand_int(12)
        if rng.random() < 0.5:
            b = a
        elif rng.random() < 0.4 and abs(a) > 9:  # near-miss: swap two digits or off-by-one
            s = str(abs(a))
            i = rng.randrange(len(s) - 1)
            b = int(s[:i] + s[i + 1] + s[i] + s[i + 2:]) * (1 if a >= 0 else -1)
            if b == a:
                b = a + rng.choice([-1, 1])
        else:
            b = rand_int(12)
        eq = 1 if a == b else 0
        st = {"a": str(a), "b": str(b)}
        items.append((st, Q_EQ, eq))
        items.append((st, Q_NEQ, 1 - eq))
    for _ in range(int(os.environ.get("PAIR_ITEMS", "1100"))):
        a, b = rand_int(12), rand_int(12)
        if rng.random() < 0.1:
            b = a
        st = {"a": str(a), "b": str(b)}
        items.append((st, Q_GT, 1 if a > b else 0))
        items.append((st, Q_LT, 1 if a < b else 0))
    rng.shuffle(items)
    return items


def encode(state, q, y, shuffle):
    k = len(render_options(q))
    order = list(range(k))
    if shuffle and rng.random() < float(os.environ.get("SHUFFLE_P", "0.3")):
        order = order[::-1]
    ids, markers = build_sequence(TOK, state, q, CFG["max_len"], CFG["head_max_len"], option_order=order)
    target = [0.0] * k
    target[order.index(y)] = 1.0
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": target, "label": order.index(y),
            "episode": 0, "ep_step": 0, "ep_len": 1, "src": "parity"}


# ----- model -----
model = build_model(CFG, encoder_dir="encoder")
model.load_state_dict(load_file("model.safetensors"), strict=True)
if AMP:
    model.float()  # fp32 master weights under fp16 autocast (Turing has no bf16)
model.to(DEV)

enc_layers = int(os.environ.get("ENCODER_LAYERS", "0"))
for p in model.encoder.parameters():
    p.requires_grad = False
if enc_layers > 0:
    for layer in model.encoder.layers[-enc_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
for p in model.act_head.parameters():
    p.requires_grad = False
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"device={DEV} trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M (encoder layers unfrozen: {enc_layers})")

items = make_items()
overfit = int(os.environ.get("OVERFIT", "0"))
if overfit:
    items = [it for it in items if it[1] in (Q_IS_EVEN, Q_IS_ODD)][:overfit]
    train_items = val_items = items
else:
    val_items = items[:600]
    train_items = items[600:]
print(f"dataset: {len(train_items)} train / {len(val_items)} val")

STEPS = int(os.environ.get("STEPS", "700"))
BATCH = int(os.environ.get("BATCH", "24"))
LR = float(os.environ.get("LR", "3e-4"))
LR_ENC = float(os.environ.get("LR_ENC", "2e-5"))
enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
head_params = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith("encoder.")]
opt = torch.optim.AdamW(
    [{"params": head_params, "lr": LR}, {"params": enc_params, "lr": LR_ENC}], weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

# the actual acceptance suite (test/real.test.js numbers, both questions)
SUITE = [(n, 1) for n in [0, 2, 4, 8, 10, 12, 100, 2024, -6, 42]] + \
        [(n, 0) for n in [1, 3, 5, 7, 9, 11, 13, 101, 2025, -7]]


@torch.no_grad()
def evaluate():
    model.eval()
    # val CE-accuracy
    correct = 0
    for s in range(0, len(val_items), 64):
        enc = [encode(*it, shuffle=False) for it in val_items[s:s + 64]]
        b = collate_items([enc], TOK.pad_token_id)
        with torch.autocast("cuda", torch.float16, enabled=AMP):
            logits, _ = model(b["input_ids"].to(DEV), b["attention_mask"].to(DEV), b["marker_pos"].to(DEV),
                              b["marker_mask"].to(DEV), b["qtype"].to(DEV))
        correct += (logits.float().argmax(-1).cpu() == b["label"]).sum().item()
    # acceptance suite: exact inference-order ([false, true]) on both parity questions
    misses = []
    enc = []
    for n, even in SUITE:
        st = {"number": str(n), "last_digit": str(n)[-1]}
        enc.append(encode(st, Q_IS_EVEN, even, shuffle=False))
        enc.append(encode(st, Q_IS_ODD, 1 - even, shuffle=False))
    b = collate_items([enc], TOK.pad_token_id)
    with torch.autocast("cuda", torch.float16, enabled=AMP):
        logits, _ = model(b["input_ids"].to(DEV), b["attention_mask"].to(DEV), b["marker_pos"].to(DEV),
                          b["marker_mask"].to(DEV), b["qtype"].to(DEV))
    pred = logits.float().argmax(-1).cpu()
    for i, (n, even) in enumerate(SUITE):
        if pred[2 * i].item() != even:
            misses.append(("isEven", n))
        if pred[2 * i + 1].item() != 1 - even:
            misses.append(("isOdd", n))
    model.train()
    return correct / len(val_items), misses


acc, misses = evaluate()
print(f"before: val_acc={acc:.3f} suite_misses={len(misses)} {misses}")

model.train()
t0 = time.time()
i = 0
best_acc, best_miss = 0.0, 99
for step in range(1, STEPS + 1):
    batch = []
    for _ in range(BATCH):
        batch.append(encode(*train_items[i % len(train_items)], shuffle=True))
        i += 1
    b = collate_items([batch], TOK.pad_token_id)
    with torch.autocast("cuda", torch.float16, enabled=AMP):
        logits, _ = model(b["input_ids"].to(DEV), b["attention_mask"].to(DEV), b["marker_pos"].to(DEV),
                          b["marker_mask"].to(DEV), b["qtype"].to(DEV), detach_encoder=(enc_layers == 0))
        logp = F.log_softmax(logits.float(), dim=-1)
        loss = -(b["target"].to(DEV) * logp).sum(-1).mean()
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    if step % 50 == 0 or step == STEPS:
        acc, misses = evaluate()
        el = time.time() - t0
        print(f"step {step}/{STEPS} loss={loss.item():.4f} val_acc={acc:.3f} suite_misses={len(misses)} "
              f"({el:.0f}s, {el/step:.2f}s/step)")
        if acc >= best_acc and len(misses) <= best_miss:
            best_acc, best_miss = acc, len(misses)
            save_file(model.state_dict(), "model_ft_best.safetensors")
            print(f"  checkpointed best (val={acc:.3f}, misses={len(misses)})")
        if step >= 150 and acc >= 0.995 and not misses:
            print("early stop: suite clean and val >= 99.5%")
            break

save_file(model.state_dict(), "model_ft.safetensors")
print("saved model_ft.safetensors")
acc, misses = evaluate()
print(f"final: val_acc={acc:.3f} suite_misses={misses}")
