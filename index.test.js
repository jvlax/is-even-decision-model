import test from 'node:test';
import assert from 'node:assert/strict';
import { isEven, explain } from './index.js';

test('classifies evens as even', () => {
  for (const n of [0, 2, 42, -8, 1000000]) {
    assert.equal(isEven(n), true, `${n} should be even`);
  }
});

test('classifies odds as odd', () => {
  for (const n of [1, 3, -7, 999999]) {
    assert.equal(isEven(n), false, `${n} should be odd`);
  }
});

test('matches ground truth across a range', () => {
  for (let n = -500; n <= 500; n++) {
    assert.equal(isEven(n), n % 2 === 0);
  }
});

test('rejects non-integers', () => {
  for (const bad of [1.5, NaN, Infinity, '2', null, undefined, {}]) {
    assert.throws(() => isEven(bad), TypeError);
  }
});

test('explain returns a fully confident decision record', () => {
  const record = explain(4);
  assert.equal(record.decision, 'even');
  assert.equal(record.confidence, 1);
});
