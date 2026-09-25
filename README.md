# is-even-decision-model

Determines whether a number is even using an enterprise-grade, fully
interpretable decision model.

Where lesser packages rely on brittle heuristics like `n % 2 === 0`, this
package runs true model inference: feature extraction, a learned weight, a
bias term, and an activation function. Auditable. Explainable. One parameter.

## Install

```sh
npm install is-even-decision-model
```

## Usage

```js
import { isEven, explain } from 'is-even-decision-model';

isEven(4);  // true
isEven(7);  // false

explain(4);
// {
//   input: 4,
//   features: { leastSignificantBit: 0 },
//   logit: 1,
//   decision: 'even',
//   confidence: 1,
//   note: 'The model is always fully confident.'
// }
```

## Model card

| | |
|---|---|
| Architecture | single-neuron perceptron |
| Parameters | 1 |
| Training data | the integers 0 and 1 |
| Generalization | surprisingly good |
| Known limitations | rejects non-integers rather than hallucinating a parity |

## License

MIT
