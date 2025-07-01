import assert from "node:assert";
import { MathUtil } from "../util/math";
import { WeightedInputPoint } from "./weighted-input-point";
import { ForwardState } from "./shared";

type ActivationFunction = (
  inputs: number[],
  weights: number[],
  bias: number
) => number;

export type LSTMBlockPoints = [
  // Input gate point
  WeightedInputPoint,
  // Forget gate point
  WeightedInputPoint,
  // Output gate point
  WeightedInputPoint,
  // Block activation point
  WeightedInputPoint
];

export class LSTMBlock {
  inputGate: ActivationFunction;
  forgetGate: ActivationFunction;
  outputActivation: ActivationFunction;
  blockActivation: ActivationFunction;

  constructor(public points: LSTMBlockPoints) {
    this.inputGate = this.createActivationFunction(MathUtil.sigmoid);
    this.forgetGate = this.createActivationFunction(MathUtil.sigmoid);
    this.outputActivation = this.createActivationFunction(MathUtil.sigmoid);
    this.blockActivation = this.createActivationFunction(Math.tanh);
  }
  forward(input: number[], previousState: ForwardState): ForwardState {
    const { hiddenState, cellState } = previousState;
    const combinedInputs = [...input, hiddenState];

    const [inputPoint, forgetPoint, outputGatePoint, blockPoint] = this.points;

    assert.strictEqual(
      combinedInputs.length,
      inputPoint.weights.length,
      "Combined inputs must match input point weights length"
    );

    const inputGateValue = this.inputGate(
      combinedInputs,
      inputPoint.weights,
      inputPoint.bias
    );

    const forgetGateValue = this.forgetGate(
      combinedInputs,
      forgetPoint.weights,
      forgetPoint.bias
    );

    const blockValue = this.blockActivation(
      combinedInputs,
      blockPoint.weights,
      blockPoint.bias
    );

    const newCellState =
      forgetGateValue * cellState + inputGateValue * blockValue;
    const outputGateValue = this.outputActivation(
      combinedInputs,
      outputGatePoint.weights,
      outputGatePoint.bias
    );

    const newHiddenState = outputGateValue * Math.tanh(newCellState);

    return {
      hiddenState: newHiddenState,
      cellState: newCellState,
    };
  }

  clone(): LSTMBlock {
    return new LSTMBlock(
      this.points.map((point) => point.clone()) as LSTMBlockPoints
    );
  }

  mutate(mutationRate: number = 0.1, mutationAmount: number = 0.1): LSTMBlock {
    return new LSTMBlock(
      this.points.map((point) =>
        point.mutate(mutationRate, mutationAmount)
      ) as LSTMBlockPoints
    );
  }

  static createRandom(inputFeatureSize: number): LSTMBlock {
    const totalInputSize = inputFeatureSize + 1; // +1 for the hidden state input
    return new LSTMBlock([
      // Input gate point
      WeightedInputPoint.createRandom(totalInputSize),
      // Forget gate point
      WeightedInputPoint.createRandom(totalInputSize),
      // Output gate point
      WeightedInputPoint.createRandom(totalInputSize),
      // Block activation point
      WeightedInputPoint.createRandom(totalInputSize),
    ]);
  }

  crossover(other: LSTMBlock): LSTMBlock {
    assert.strictEqual(
      this.points.length,
      other.points.length,
      "Both LSTM blocks must have the same number of points"
    );
    const newPoints: LSTMBlockPoints = this.points.map((point, index) => {
      const otherPoint = other.points[index];
      return point.crossover(otherPoint);
    }) as LSTMBlockPoints;
    return new LSTMBlock(newPoints);
  }

  toJSON(): any {
    return {
      points: this.points.map((point) => point.toJSON()),
    };
  }

  static fromJSON(json: any): LSTMBlock {
    assert(
      Array.isArray(json.points),
      "LSTMBlock JSON must have a points array"
    );
    const points = json.points.map((point: any) =>
      WeightedInputPoint.fromJSON(point)
    ) as LSTMBlockPoints;
    return new LSTMBlock(points);
  }

  private createActivationFunction(
    fn: (x: number) => number
  ): ActivationFunction {
    return (inputs: number[], weights: number[], bias: number): number => {
      assert.strictEqual(
        inputs.length,
        weights.length,
        "Inputs and weights must have the same length"
      );
      let sum = 0;
      for (let i = 0; i < inputs.length; i++) {
        sum += inputs[i] * weights[i];
      }
      return fn(sum + bias);
    };
  }
}
