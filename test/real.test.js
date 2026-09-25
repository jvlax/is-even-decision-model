// The model itself, end to end: downloads about 1.7 GB of ONNX weights on first run
// (cached under ~/.cache/receptron-laya, or set IS_EVEN_MODEL_DIR to a
// local bundle) and asks the actual model. Opt in with IS_EVEN_REAL=1.
import assert from "node:assert/strict";
import { after, test } from "node:test";

import { close, howEven, isEven, isOdd, warmUp } from "../index.js";

const real = process.env.IS_EVEN_REAL === "1";

test("Laya knows its parity", { skip: !real && "set IS_EVEN_REAL=1 to download the model and run this" }, async () => {
  await warmUp();
  const evens = [0, 2, 4, 8, 10, 12, 100, 2024, -6];
  const odds = [1, 3, 5, 7, 9, 11, 13, 101, 2025, -7];
  const misses = [];
  for (const n of evens) {
    const p = await howEven(n);
    console.log(`P(${n} is even) = ${p}`);
    if (!(await isEven(n))) misses.push(n);
  }
  for (const n of odds) {
    const p = await howEven(n);
    console.log(`P(${n} is even) = ${p}`);
    if (await isEven(n)) misses.push(n);
    if (!(await isOdd(n))) misses.push(n);
  }
  // A 421M-parameter decision model is a calibrated guesser, not an ALU. It
  // should still get the overwhelming majority of single- and double-digit
  // cases right; this is the bar the README's claims are held to.
  const total = evens.length + odds.length * 2;
  console.log(`misses: ${misses.length}/${total} ${JSON.stringify(misses)}`);
  assert.ok(misses.length <= Math.floor(total * 0.15), `too many misses: ${JSON.stringify(misses)}`);
});

after(() => close());
