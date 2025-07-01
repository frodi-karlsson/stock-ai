import assert from "node:assert";
import { MathUtil } from "./math";

export function testSigmoid() {
  const inputs = [-10, -5, 0, 5, 10];
  const expectedOutputs = [
    0.000045397868702434395, 0.0066928509242848554, 0.5, 0.9933071490757153,
    0.9999546021312976,
  ];

  inputs.forEach((input, index) => {
    const output = MathUtil.sigmoid(input);
    assert(
      Math.abs(output - expectedOutputs[index]) < 1e-10,
      `Sigmoid test failed for input ${input}: expected ${expectedOutputs[index]}, got ${output}`
    );
  });
}

export function testClamp() {
  const testCases = [
    { value: 5, min: 0, max: 10, expected: 5 },
    { value: -5, min: 0, max: 10, expected: 0 },
    { value: 15, min: 0, max: 10, expected: 10 },
    { value: 0, min: 0, max: 10, expected: 0 },
    { value: 10, min: 0, max: 10, expected: 10 },
  ];

  testCases.forEach(({ value, min, max, expected }) => {
    const result = MathUtil.clamp(value, min, max);
    assert.strictEqual(
      result,
      expected,
      `Clamp test failed for value ${value}: expected ${expected}, got ${result}`
    );
  });
}

export function testWeightedSum() {
  const inputs = [1, 2, 3];
  const weights = [0.5, -0.5, 0.2];
  const bias = 0.1;
  const expectedSum = 1 * 0.5 + 2 * -0.5 + 3 * 0.2 + bias;

  const result = MathUtil.weightedSum(inputs, weights, bias);
  assert.strictEqual(
    result,
    expectedSum,
    `Weighted sum test failed: expected ${expectedSum}, got ${result}`
  );
}

export function testRandomFloat() {
  const testCases = [
    { min: 1, max: 10 },
    { min: -5, max: 5 },
    { min: 0, max: 0 },
  ];

  testCases.forEach(({ min, max }) => {
    const result = MathUtil.randomFloat(min, max);
    assert(
      result >= min && result <= max,
      `Random float test failed: ${result} is not between
  ${min} and ${max}`
    );
    assert(
      Number.isFinite(result),
      `Random float test failed: ${result} is not a finite number`
    );
  });
}
