import assert from "node:assert";
import {
  MIN_WEIGHT,
  MAX_WEIGHT,
  MIN_BIAS,
  MAX_BIAS,
  MUTATION_RATE,
  MUTATION_AMOUNT,
} from "../constants";
import { MathUtil } from "../util/math";

export class Neuron {
  weights: number[];
  bias: number;
  activationFunction: (x: number) => number;
  numInputs: number;
  numOutputs: number;

  constructor(
    numInputs: number,
    activationFunction: (x: number) => number,
    private minWeight: number = MIN_WEIGHT,
    private maxWeight: number = MAX_WEIGHT,
    private minBias: number = MIN_BIAS,
    private maxBias: number = MAX_BIAS
  ) {
    this.numInputs = numInputs;
    this.numOutputs = 1;
    this.weights = Array.from(
      { length: numInputs },
      () => Math.random() * (maxWeight - minWeight) + minWeight
    );
    this.bias = Math.random() * (maxBias - minBias) + minBias;
    this.activationFunction = activationFunction;
  }

  activate(inputs: number[]): number {
    const weightedSum = MathUtil.weightedSum(inputs, this.weights, this.bias);
    return this.activationFunction(weightedSum);
  }

  clone(): this {
    const clone = new Neuron(this.numInputs, this.activationFunction);
    clone.weights = [...this.weights];
    clone.bias = this.bias;
    return clone as this;
  }

  mutate(
    mutationRate: number = MUTATION_RATE,
    mutationAmount: number = MUTATION_AMOUNT
  ): this {
    for (let i = 0; i < this.weights.length; i++) {
      const shouldMutate = Math.random() < mutationRate;
      if (!shouldMutate) continue;
      this.weights[i] = MathUtil.clamp(
        this.weights[i] +
          (Math.random() * (this.maxWeight - this.minWeight) + this.minWeight) *
            mutationAmount,
        this.minWeight,
        this.maxWeight
      );
    }
    this.bias = MathUtil.clamp(
      this.bias +
        (Math.random() * (this.maxBias - this.minBias) + this.minBias) *
          mutationAmount,
      this.minBias,
      this.maxBias
    );
    return this;
  }

  crossover(other: Neuron): Neuron {
    assert.strictEqual(
      this.numInputs,
      other.numInputs,
      "Input sizes must match for crossover"
    );

    const newNeuron = new Neuron(this.numInputs, this.activationFunction);
    for (let i = 0; i < this.weights.length; i++) {
      newNeuron.weights[i] =
        Math.random() < 0.5 ? this.weights[i] : other.weights[i];
    }
    newNeuron.bias = Math.random() < 0.5 ? this.bias : other.bias;

    return newNeuron;
  }

  toJSON(): any {
    return {
      weights: this.weights,
      bias: this.bias,
      activationFunction: this.activationFunction.name,
      minWeight: this.minWeight,
      maxWeight: this.maxWeight,
      minBias: this.minBias,
      maxBias: this.maxBias,
    };
  }

  static fromJSON(json: any): Neuron {
    assert(
      Array.isArray(json.weights),
      "Weights must be an array in JSON representation"
    );
    assert(
      typeof json.bias === "number",
      "Bias must be a number in JSON representation"
    );
    assert(
      typeof json.activationFunction === "string",
      "Activation function must be a string in JSON representation"
    );

    const neuron = new Neuron(
      json.weights.length,
      (MathUtil as any)[json.activationFunction],
      json.minWeight,
      json.maxWeight,
      json.minBias,
      json.maxBias
    );
    neuron.weights = json.weights;
    neuron.bias = json.bias;
    return neuron;
  }
}
