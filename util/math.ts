import assert from "node:assert";

export const MathUtil = {
  sigmoid: (x: number): number => {
    return 1 / (1 + Math.exp(-x));
  },

  clamp: (value: number, min: number, max: number): number => {
    return Math.max(min, Math.min(max, value));
  },

  weightedSum: (inputs: number[], weights: number[], bias: number): number => {
    assert.strictEqual(
      inputs.length,
      weights.length,
      "Inputs and weights must have the same length"
    );
    let sum = 0;
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i] * weights[i];
    }
    return sum + bias;
  },

  randomFloat: (min: number, max: number): number => {
    return Math.random() * (max - min) + min;
  },

  meanSquaredError: (predictions: number[], targets: number[]): number => {
    assert.strictEqual(
      predictions.length,
      targets.length,
      "Predictions and targets must have the same length"
    );
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
      const error = predictions[i] - targets[i];
      sum += error * error;
    }
    return sum / predictions.length;
  },
};
