import { MathUtil } from "../util/math";

export class WeightedInputPoint {
  constructor(
    public weights: number[],
    public bias: number = MathUtil.randomFloat(-1, 1)
  ) {}

  static createRandom(
    size: number,
    bias: number = MathUtil.randomFloat(-1, 1)
  ): WeightedInputPoint {
    const weights = Array.from({ length: size }, () =>
      MathUtil.randomFloat(-1, 1)
    );
    return new WeightedInputPoint(weights, bias);
  }

  mutate(
    mutationRate: number = 0.1,
    mutationAmount: number = 0.1
  ): WeightedInputPoint {
    const mutatedWeights = this.weights.map((weight) => {
      if (Math.random() < mutationRate) {
        return weight + MathUtil.randomFloat(-mutationAmount, mutationAmount);
      }
      return weight;
    });

    const mutatedBias =
      this.bias +
      (Math.random() < mutationRate
        ? MathUtil.randomFloat(-mutationAmount, mutationAmount)
        : 0);

    return new WeightedInputPoint(mutatedWeights, mutatedBias);
  }

  crossover(other: WeightedInputPoint): WeightedInputPoint {
    const newWeights = this.weights.map((weight, index) => {
      return Math.random() < 0.5 ? weight : other.weights[index];
    });

    const newBias = Math.random() < 0.5 ? this.bias : other.bias;

    return new WeightedInputPoint(newWeights, newBias);
  }

  clone(): WeightedInputPoint {
    return new WeightedInputPoint([...this.weights], this.bias);
  }

  toJSON(): any {
    return {
      weights: this.weights,
      bias: this.bias,
    };
  }

  static fromJSON(data: any): WeightedInputPoint {
    if (
      !data ||
      !Array.isArray(data.weights) ||
      typeof data.bias !== "number"
    ) {
      throw new Error("Invalid JSON data for WeightedInputPoint");
    }
    return new WeightedInputPoint(data.weights, data.bias);
  }
}
