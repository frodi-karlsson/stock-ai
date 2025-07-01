import assert from "node:assert";
import { MinMax } from "./shared";

export class Scaler {
  public minMax: MinMax[];

  constructor(featureData: number[][]) {
    assert(
      featureData.length,
      "Feature data must contain at least one element"
    );

    const numFeatures = featureData[0].length;
    this.minMax = Array.from({ length: numFeatures }, () => ({
      min: Number.POSITIVE_INFINITY,
      max: Number.NEGATIVE_INFINITY,
    }));

    for (const sequence of featureData) {
      for (let i = 0; i < numFeatures; i++) {
        const value = sequence[i];
        const minMax = this.minMax[i];
        if (value < minMax.min) {
          minMax.min = value;
        }
        if (value > minMax.max) {
          minMax.max = value;
        }
      }
    }
    assert(
      this.minMax.every(
        (mm) =>
          mm.min !== Number.POSITIVE_INFINITY &&
          mm.max !== Number.NEGATIVE_INFINITY
      ),
      "MinMax values must be initialized"
    );

    assert(
      this.minMax.every((mm) => mm.min < mm.max),
      "MinMax min must be less than max"
    );
  }

  scaleSingle(value: number, featureIndex: number): number {
    const { min, max } = this.minMax[featureIndex];
    const divisor = max - min;

    if (divisor === 0) {
      return 0;
    }

    return (value - min) / divisor;
  }

  scale(timestep: number[]): number[] {
    assert(
      timestep.length === this.minMax.length,
      `Timestep length must match number of features: ${this.minMax.length}`
    );
    return timestep.map((value, index) => this.scaleSingle(value, index));
  }

  inverseScaleSingle(value: number, featureIndex: number): number {
    const { min, max } = this.minMax[featureIndex];
    return value * (max - min) + min;
  }

  inverseScale(timestep: number[]): number[] {
    assert(
      timestep.length === this.minMax.length,
      `Timestep length must match number of features: ${this.minMax.length}`
    );
    return timestep.map((value, index) =>
      this.inverseScaleSingle(value, index)
    );
  }
}
