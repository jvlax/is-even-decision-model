/**
 * is-even-decision — check if a number is even using the power of ✨a decision model✨.
 *
 * is-even asked is-odd. is-even-ai asked GPT-3.5 (and your credit card).
 * This asks Laya, an open-weights 421M-parameter System 1 decision model, on
 * your own CPU: no API key, no cloud, one forward pass, and a calibrated
 * probability that your number is even. It is only 1.7 GB.
 *
 * Drop-in for is-even-ai: the same six functions, minus setApiKey.
 *
 * Every function here is a single `noul` (yes/no) question to the model. The
 * model is downloaded on first use and loaded once per process.
 */

import { Laya } from "@receptron/laya";

/** @typedef {import("@receptron/laya").LayaOptions} LayaOptions */
/** @typedef {import("./index.d.ts").Engine} Engine */

/** @type {LayaOptions} */
let loadOptions = {};
/** @type {Promise<Engine> | null} */
let pending = null;
/** @type {Engine | null} */
let engine = null;

/** Where the ONNX bundle lives, when you already have one and would rather not download it again. */
const modelDirFromEnv = () => process.env.IS_EVEN_MODEL_DIR || process.env.LAYA_MODEL_DIR || undefined;

/**
 * Options forwarded to `Laya.load()` — model directory, Hugging Face repo, cache dir,
 * execution providers, a download progress callback. Takes effect on the next load.
 * @param {LayaOptions} options
 */
export function configure(options) {
  loadOptions = { ...options };
}

/**
 * Bring your own engine: anything with `systemOne(state, questions)`. Used by the tests so
 * they do not need 1.7 GB of weights, and by you if you would rather ask Jev.
 * @param {Engine | null} custom
 */
export function setEngine(custom) {
  engine = custom;
  pending = null;
}

/** Load the model now instead of on the first question. Returns the engine. */
export async function warmUp() {
  if (engine) return engine;
  if (!pending) {
    const modelDir = loadOptions.modelDir ?? modelDirFromEnv();
    pending = Laya.load({ ...loadOptions, ...(modelDir ? { modelDir } : {}) }).then((loaded) => {
      engine = loaded;
      return loaded;
    });
    pending.catch(() => {
      pending = null;
    });
  }
  return pending;
}

/** Release the model. The next question loads it again. */
export async function close() {
  const current = engine;
  engine = null;
  pending = null;
  if (current?.close) await current.close();
}

/**
 * @param {unknown} value
 * @param {string} name
 * @returns {string}
 */
function digits(value, name) {
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  throw new TypeError(`${name} must be a finite number or a bigint, got ${typeof value}`);
}

/**
 * Ask the model one yes/no question about a state and get P(true).
 * @param {Record<string, string>} state
 * @param {import("@receptron/laya").NoulQuestion} question
 * @returns {Promise<number>}
 */
export async function decide(state, question) {
  const model = await warmUp();
  const result = await model.systemOne(state, { verdict: question });
  const answer = result.answers.verdict;
  if (answer?.type !== "noul") throw new Error(`expected a noul answer, got ${JSON.stringify(answer)}`);
  return answer.noul;
}

/**
 * The calibrated probability that `n` is even. The whole package, really.
 * @param {number | bigint} n
 * @returns {Promise<number>}
 */
export async function howEven(n) {
  return decide(
    { number: digits(n, "n") },
    {
      type: "noul",
      instructions:
        "Is the number even? A number is even when dividing it by 2 leaves no remainder. " +
        "Look at the last digit: 0, 2, 4, 6 and 8 are even; 1, 3, 5, 7 and 9 are odd. Negative numbers follow the same rule.",
      criteria: {
        true: "the number is even, like 0, 2, 4, 6, 8, 10, 12, 100 or -4",
        false: "the number is odd, like 1, 3, 5, 7, 9, 11, 13, 101 or -3",
      },
    },
  );
}

/**
 * Check if a number is even.
 * @param {number | bigint} n
 * @returns {Promise<boolean>}
 */
export async function isEven(n) {
  return (await howEven(n)) >= 0.5;
}

/**
 * Check if a number is odd. Yes, it could call isEven. That would be cheating.
 * @param {number | bigint} n
 * @returns {Promise<boolean>}
 */
export async function isOdd(n) {
  const p = await decide(
    { number: digits(n, "n") },
    {
      type: "noul",
      instructions:
        "Is the number odd? A number is odd when dividing it by 2 leaves a remainder of 1. " +
        "Look at the last digit: 1, 3, 5, 7 and 9 are odd; 0, 2, 4, 6 and 8 are even. Negative numbers follow the same rule.",
      criteria: {
        true: "the number is odd, like 1, 3, 5, 7, 9, 11, 13, 101 or -3",
        false: "the number is even, like 0, 2, 4, 6, 8, 10, 12, 100 or -4",
      },
    },
  );
  return p >= 0.5;
}

/**
 * Check if two numbers are equal.
 * @param {number | bigint} a
 * @param {number | bigint} b
 * @returns {Promise<boolean>}
 */
export async function areEqual(a, b) {
  const p = await decide(
    { a: digits(a, "a"), b: digits(b, "b") },
    {
      type: "noul",
      instructions: "Are a and b the same number? Compare them digit by digit, including the sign.",
      criteria: { true: "a and b are exactly the same number", false: "a and b are different numbers" },
    },
  );
  return p >= 0.5;
}

/**
 * Check if two numbers are not equal.
 * @param {number | bigint} a
 * @param {number | bigint} b
 * @returns {Promise<boolean>}
 */
export async function areNotEqual(a, b) {
  const p = await decide(
    { a: digits(a, "a"), b: digits(b, "b") },
    {
      type: "noul",
      instructions: "Are a and b different numbers? Compare them digit by digit, including the sign.",
      criteria: { true: "a and b are different numbers", false: "a and b are exactly the same number" },
    },
  );
  return p >= 0.5;
}

/**
 * Check if a is greater than b.
 * @param {number | bigint} a
 * @param {number | bigint} b
 * @returns {Promise<boolean>}
 */
export async function isGreaterThan(a, b) {
  const p = await decide(
    { a: digits(a, "a"), b: digits(b, "b") },
    {
      type: "noul",
      instructions: "Is a strictly greater than b, as numbers on the number line?",
      criteria: { true: "a > b", false: "a <= b (a is smaller than b, or they are equal)" },
    },
  );
  return p >= 0.5;
}

/**
 * Check if a is less than b.
 * @param {number | bigint} a
 * @param {number | bigint} b
 * @returns {Promise<boolean>}
 */
export async function isLessThan(a, b) {
  const p = await decide(
    { a: digits(a, "a"), b: digits(b, "b") },
    {
      type: "noul",
      instructions: "Is a strictly less than b, as numbers on the number line?",
      criteria: { true: "a < b", false: "a >= b (a is greater than b, or they are equal)" },
    },
  );
  return p >= 0.5;
}

/**
 * The full decision record for `n`: what the model was asked, what it answered, and how sure it was.
 * @param {number | bigint} n
 */
export async function explain(n) {
  const probability = await howEven(n);
  return {
    input: digits(n, "n"),
    model: "laya",
    question: "noul: is the number even?",
    probability,
    decision: probability >= 0.5 ? "even" : "odd",
    confidence: Math.max(probability, 1 - probability),
    note: "Calibrated, not certain. The model has never been certain of anything.",
  };
}

/**
 * What the same calls cost through is-even-ai, and what they cost here.
 *
 * is-even-ai sends a 24-token system prompt and `Is ${n} an even number?` to gpt-3.5-turbo and
 * streams back one token. Measured with gpt-tokenizer: 44 input tokens for a small number, 1 out.
 * gpt-3.5-turbo is priced at $0.50 per million input tokens and $1.50 per million output tokens.
 * A decision model runs on the CPU you already pay for, so the marginal cost is zero.
 *
 * @param {number} calls  how many parity checks your system makes
 * @param {{ inputTokens?: number, outputTokens?: number, inputPricePerMillion?: number, outputPricePerMillion?: number }} [pricing]
 */
export function savings(calls, pricing = {}) {
  const inputTokens = pricing.inputTokens ?? 44;
  const outputTokens = pricing.outputTokens ?? 1;
  const inputPrice = pricing.inputPricePerMillion ?? 0.5;
  const outputPrice = pricing.outputPricePerMillion ?? 1.5;
  const perCall = (inputTokens * inputPrice + outputTokens * outputPrice) / 1e6;
  const withAi = calls * perCall;
  return {
    calls,
    tokens: calls * (inputTokens + outputTokens),
    withAi,
    withDecisionModel: 0,
    saved: withAi,
    apiKeysRequired: 0,
  };
}

export default isEven;
