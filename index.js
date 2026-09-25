/**
 * is-even-decision-model
 *
 * A single-neuron perceptron for binary parity classification.
 * Trained on the integers 0 and 1; generalizes surprisingly well.
 */

const MODEL = {
  architecture: 'single-neuron perceptron',
  parameters: 1,
  weights: { leastSignificantBit: -2 },
  bias: 1,
  activation: (logit) => logit > 0,
};

function extractFeatures(n) {
  if (typeof n !== 'number' || !Number.isInteger(n)) {
    throw new TypeError(`is-even-decision-model: expected an integer, got ${typeof n === 'number' ? n : typeof n}`);
  }
  return { leastSignificantBit: Math.abs(n % 2) };
}

/**
 * Run inference. Returns true when the model classifies n as even.
 */
export function isEven(n) {
  const features = extractFeatures(n);
  const logit = MODEL.bias + MODEL.weights.leastSignificantBit * features.leastSignificantBit;
  return MODEL.activation(logit);
}

/**
 * Full decision record, for auditability and regulatory compliance.
 */
export function explain(n) {
  const features = extractFeatures(n);
  const logit = MODEL.bias + MODEL.weights.leastSignificantBit * features.leastSignificantBit;
  return {
    input: n,
    features,
    logit,
    decision: MODEL.activation(logit) ? 'even' : 'odd',
    confidence: 1,
    note: 'The model is always fully confident.',
  };
}

export { MODEL };
