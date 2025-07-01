import assert from "node:assert";
import { CSVData } from "./data";
import { Scaler } from "./scaler";
import { SEQUENCE_LENGTH, TRAINING_SPLIT } from "../constants";

export interface TimeSeries {
  inputSequences: number[][][];
  targetOutputs: number[][];
  scaler: Scaler;
}

export interface PreparedData {
  training: TimeSeries;
  validation: TimeSeries;
}

export class DataPreparer<
  CSVDataType extends Record<string, any>,
  TargetKey extends keyof CSVDataType
> {
  constructor(private data: CSVData<CSVDataType, TargetKey>) {}

  prepareData(): PreparedData {
    const { data, targetKey, featureKeys } = this.data;

    assert(data.length, "Data must not be empty");
    assert(featureKeys.length, "Feature keys must not be empty");

    const allFeatureVectors: number[][] = [];
    const allTargetValues: number[] = [];

    for (const item of data) {
      const features = this.data.getFeatures(item);
      const target = this.data.getTarget(item);

      allFeatureVectors.push(features);
      allTargetValues.push(target);
    }

    const scaler = new Scaler(allFeatureVectors);

    const inputSequences: number[][][] = [];
    const targetOutputs: number[][] = [];

    for (let i = 0; i < allFeatureVectors.length - SEQUENCE_LENGTH; i++) {
      const sequence: number[][] = [];
      for (let j = 0; j < SEQUENCE_LENGTH; j++) {
        const index = i + j;
        const rawFeatures = allFeatureVectors[index];
        assert(
          rawFeatures,
          `Feature vector at index ${index} must not be undefined`
        );
        const scaledFeatures = rawFeatures.map((value, featureIndex) =>
          scaler.scaleSingle(value, featureIndex)
        );
        sequence.push(scaledFeatures);
      }

      inputSequences.push(sequence);

      const targetIndex = i + SEQUENCE_LENGTH;
      assert(
        targetIndex < allTargetValues.length,
        `Target index ${targetIndex} must be within bounds of target values`
      );
      const targetValue = allTargetValues[targetIndex];
      assert(
        typeof targetValue === "number",
        `Target value at index ${targetIndex} must be a number, got: ${typeof targetValue}`
      );

      assert(
        featureKeys.includes(targetKey),
        `Target key "${String(
          targetKey
        )}" must be one of feature keys: ${featureKeys.join(", ")}`
      );
      const featureIndexForTarget = featureKeys.indexOf(targetKey);

      const scaledTarget = scaler.scaleSingle(
        targetValue,
        featureIndexForTarget
      );

      targetOutputs.push([scaledTarget]);
    }

    const splitIndex = Math.floor(inputSequences.length * TRAINING_SPLIT);

    const trainingData: TimeSeries = {
      inputSequences: inputSequences.slice(0, splitIndex),
      targetOutputs: targetOutputs.slice(0, splitIndex),
      scaler,
    };

    const validationData: TimeSeries = {
      inputSequences: inputSequences.slice(splitIndex),
      targetOutputs: targetOutputs.slice(splitIndex),
      scaler,
    };

    return {
      training: trainingData,
      validation: validationData,
    };
  }
}
