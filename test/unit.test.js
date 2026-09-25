// The plumbing, proven without the weights: a fake engine that answers each
// `noul` question the way a well-calibrated Laya would, and records what it
// was asked. `npm run test:real` asks the actual model.
import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import isEvenDefault, {
  areEqual,
  areNotEqual,
  close,
  explain,
  howEven,
  isEven,
  isGreaterThan,
  isLessThan,
  isOdd,
  savings,
  setEngine,
} from "../index.js";

/** @type {{ state: Record<string, string>, question: import("@receptron/laya").NoulQuestion }[]} */
let asked = [];
let closed = 0;

/**
 * Answers yes/no questions about integers by actually doing the arithmetic. Calibrated, allegedly.
 * @type {import("../index.js").Engine}
 */
const oracle = {
  async systemOne(state, questions) {
    const q = /** @type {import("@receptron/laya").NoulQuestion} */ (questions.verdict);
    const s = /** @type {Record<string, string>} */ (state);
    asked.push({ state: s, question: q });
    const text = String(q.instructions);
    let truth;
    if (text.startsWith("Is the number even")) truth = BigInt(s.number) % 2n === 0n;
    else if (text.startsWith("Is the number odd")) truth = BigInt(s.number) % 2n !== 0n;
    else if (text.startsWith("Are a and b the same")) truth = BigInt(s.a) === BigInt(s.b);
    else if (text.startsWith("Are a and b different")) truth = BigInt(s.a) !== BigInt(s.b);
    else if (text.startsWith("Is a strictly greater")) truth = BigInt(s.a) > BigInt(s.b);
    else if (text.startsWith("Is a strictly less")) truth = BigInt(s.a) < BigInt(s.b);
    else throw new Error(`the oracle has no opinion on: ${text}`);
    return {
      model: "laya",
      answers: { verdict: { type: "noul", noul: truth ? 0.97 : 0.03, rl_agent: { act_probability: 1 } } },
      usage: { input_tokens: 42, output_tokens: 0 },
    };
  },
  async close() {
    closed++;
  },
};

beforeEach(() => {
  asked = [];
  closed = 0;
  setEngine(oracle);
});
afterEach(() => setEngine(null));

test("isEven", async () => {
  assert.equal(await isEven(2), true);
  assert.equal(await isEven(3), false);
  assert.equal(await isEven(0), true);
  assert.equal(await isEven(-4), true);
  assert.equal(await isEven(-7), false);
  assert.equal(await isEvenDefault(10), true);
});

test("isEven accepts bigint and keeps every digit", async () => {
  assert.equal(await isEven(123456789012345678901234567890n), true);
  assert.equal(asked.at(-1)?.state.number, "123456789012345678901234567890");
});

test("howEven is the probability, not the verdict", async () => {
  assert.equal(await howEven(8), 0.97);
  assert.equal(await howEven(9), 0.03);
});

test("isOdd asks its own question instead of negating isEven", async () => {
  assert.equal(await isOdd(5), true);
  assert.equal(await isOdd(6), false);
  assert.match(String(asked.at(-1)?.question.instructions), /^Is the number odd/);
});

test("comparisons", async () => {
  assert.equal(await areEqual(6, 6), true);
  assert.equal(await areEqual(6, 7), false);
  assert.equal(await areNotEqual(6, 7), true);
  assert.equal(await areNotEqual(6, 6), false);
  assert.equal(await isGreaterThan(8, 7), true);
  assert.equal(await isGreaterThan(7, 8), false);
  assert.equal(await isLessThan(8, 9), true);
  assert.equal(await isLessThan(9, 8), false);
});

test("every question is a noul with both criteria spelled out", async () => {
  await isEven(1);
  await areEqual(1, 2);
  for (const { question } of asked) {
    assert.equal(question.type, "noul");
    assert.ok(question.criteria?.true);
    assert.ok(question.criteria?.false);
  }
});

test("the state is just digits: the model gets the number, not the answer", async () => {
  await isEven(42);
  assert.deepEqual(asked.at(-1)?.state, { number: "42" });
  await isGreaterThan(-1, 1);
  assert.deepEqual(asked.at(-1)?.state, { a: "-1", b: "1" });
});

test("rejects things that are not numbers", async () => {
  await assert.rejects(() => isEven(/** @type {any} */ ("2")), TypeError);
  await assert.rejects(() => isEven(NaN), TypeError);
  await assert.rejects(() => isEven(Infinity), TypeError);
  await assert.rejects(() => isEven(/** @type {any} */ (undefined)), TypeError);
  assert.equal(asked.length, 0);
});

test("close releases the engine", async () => {
  await isEven(2);
  await close();
  assert.equal(closed, 1);
});

test("explain is the whole record", async () => {
  const r = await explain(12);
  assert.equal(r.input, "12");
  assert.equal(r.decision, "even");
  assert.equal(r.probability, 0.97);
  assert.equal(r.confidence, 0.97);
  const o = await explain(13);
  assert.equal(o.decision, "odd");
  assert.equal(o.confidence, 0.97);
});

test("savings: 44 input + 1 output token at gpt-3.5-turbo prices, versus nothing", () => {
  const s = savings(1_000_000);
  assert.equal(s.tokens, 45_000_000);
  assert.ok(Math.abs(s.withAi - 23.5) < 1e-9);
  assert.equal(s.withDecisionModel, 0);
  assert.equal(s.saved, s.withAi);
  assert.equal(s.apiKeysRequired, 0);
  assert.equal(savings(0).saved, 0);
  assert.ok(Math.abs(savings(1e9, { inputPricePerMillion: 1, outputPricePerMillion: 1 }).withAi - 45_000) < 1e-6);
});
