import type { Answer, LayaOptions, NoulQuestion, Question } from "@receptron/laya";

/** Anything that can answer a `noul` question the way Laya does (a `Laya` instance is one). */
export interface Engine {
  systemOne(state: unknown, questions: Record<string, Question>): Promise<{ answers: Record<string, Answer> }>;
  close?(): Promise<void>;
}

/** Options forwarded to `Laya.load()`. Takes effect on the next load. */
export function configure(options: LayaOptions): void;
/** Bring your own engine (a fake for tests, a different checkpoint, Jev if you must). `null` restores the default. */
export function setEngine(custom: Engine | null): void;
/** Download and load the model now instead of on the first question. */
export function warmUp(): Promise<Engine>;
/** Release the model. The next question loads it again. */
export function close(): Promise<void>;

/** Ask the model one yes/no question about a state and get P(true). */
export function decide(state: Record<string, string>, question: NoulQuestion): Promise<number>;

/** The calibrated probability that `n` is even. */
export function howEven(n: number | bigint): Promise<number>;
/** Check if a number is even. */
export function isEven(n: number | bigint): Promise<boolean>;
/** Check if a number is odd. */
export function isOdd(n: number | bigint): Promise<boolean>;
/** Check if two numbers are equal. */
export function areEqual(a: number | bigint, b: number | bigint): Promise<boolean>;
/** Check if two numbers are not equal. */
export function areNotEqual(a: number | bigint, b: number | bigint): Promise<boolean>;
/** Check if a is greater than b. */
export function isGreaterThan(a: number | bigint, b: number | bigint): Promise<boolean>;
/** Check if a is less than b. */
export function isLessThan(a: number | bigint, b: number | bigint): Promise<boolean>;

/** The full decision record for `n`. */
export function explain(n: number | bigint): Promise<{
  input: string;
  model: "laya";
  question: string;
  probability: number;
  decision: "even" | "odd";
  confidence: number;
  note: string;
}>;

export interface Pricing {
  /** input tokens per is-even-ai call (default 44, measured) */
  inputTokens?: number;
  /** output tokens per is-even-ai call (default 1) */
  outputTokens?: number;
  /** USD per million input tokens (default 0.5, gpt-3.5-turbo) */
  inputPricePerMillion?: number;
  /** USD per million output tokens (default 1.5, gpt-3.5-turbo) */
  outputPricePerMillion?: number;
}

/** What `calls` parity checks cost through is-even-ai, and what they cost here. */
export function savings(
  calls: number,
  pricing?: Pricing,
): { calls: number; tokens: number; withAi: number; withDecisionModel: 0; saved: number; apiKeysRequired: 0 };

export default isEven;
