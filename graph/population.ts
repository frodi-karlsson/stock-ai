import assert from "node:assert";
import {
  HIDDEN_UNITS_PER_LAYER,
  INPUT_FEATURE_SIZE,
  LSTM_LAYER_COUNT,
  MUTATION_AMOUNT,
  MUTATION_RATE,
  OUTPUT_SIZE,
  POPULATION_SIZE,
} from "../constants";
import { CSVData } from "./data";
import { DataPreparer, TimeSeries } from "./data-preparer";
import { Network } from "./network";
import { green } from "../util/console";

export class Population<
  CSVDataType extends Record<string, any>,
  TargetKey extends keyof CSVDataType
> {
  individuals: Network[];
  generation: number;
  bestIndividual: Network | null;
  bestFitness: number;
  trainingData: TimeSeries | null;
  validationData: TimeSeries | null;

  constructor(public data: CSVData<CSVDataType, TargetKey>) {
    this.individuals = [];
    this.generation = 0;
    this.bestIndividual = null;
    this.bestFitness = -Infinity;
    this.trainingData = null;
    this.validationData = null;
  }

  async loadData(): Promise<void> {
    const dataPreparer = new DataPreparer(this.data);
    const preparedData = dataPreparer.prepareData();
    this.trainingData = preparedData.training;
    this.validationData = preparedData.validation;
  }

  private initializePopulation(): void {
    for (let i = 0; i < POPULATION_SIZE; i++) {
      const individual = Network.createRandom(
        // Features per time step
        INPUT_FEATURE_SIZE,
        // Hidden units per layer
        HIDDEN_UNITS_PER_LAYER,
        // Layer count
        LSTM_LAYER_COUNT,
        // Final output size
        OUTPUT_SIZE
      );
      this.individuals.push(individual);
    }
  }

  evaluateFitness(): void {
    assert(
      this.trainingData,
      "Training data must be loaded before evaluating fitness"
    );
    assert(
      this.individuals.length,
      "Population must not be empty before evaluating fitness"
    );

    let currentBestFitness = -Infinity;
    let currentBestIndividual: Network | null = null;

    for (const individual of this.individuals) {
      let totalLoss = 0;

      for (let i = 0; i < this.trainingData.inputSequences.length; i++) {
        const inputSequence = this.trainingData.inputSequences[i];
        const targetOutput = this.trainingData.targetOutputs[i];

        const predictions = individual.forward(inputSequence);
        const lastPrediction = predictions[predictions.length - 1];

        assert(
          lastPrediction.length === targetOutput.length,
          `Prediction length ${lastPrediction.length} does not match target output length ${targetOutput.length}`
        );

        let loss = 0;
        for (let j = 0; j < lastPrediction.length; j++) {
          loss += Math.pow(lastPrediction[j] - targetOutput[j], 2);
        }
        totalLoss += loss;
      }

      const fitness = -totalLoss / this.trainingData.inputSequences.length;
      individual.fitness = fitness;

      if (fitness > currentBestFitness) {
        currentBestFitness = fitness;
        currentBestIndividual = individual;
      }
    }

    if (currentBestFitness > this.bestFitness) {
      this.bestFitness = currentBestFitness;
      if (currentBestIndividual) {
        this.bestIndividual = currentBestIndividual.clone();
      }
    }
  }

  selectParents(): Network[] {
    assert(
      this.individuals.length >= 2,
      "Population must have at least two individuals to select parents"
    );

    this.individuals.sort((a, b) => b.fitness - a.fitness);

    const parentCount = Math.floor(this.individuals.length * 0.25);
    return this.individuals.slice(0, parentCount);
  }

  crossoverAndMutate(): void {
    const newPopulation: Network[] = [];
    const parents = this.selectParents();

    while (newPopulation.length < POPULATION_SIZE) {
      const parent1 = parents[Math.floor(Math.random() * parents.length)];
      let parent2: Network | null = null;

      if (parents.length > 1) {
        do {
          parent2 = parents[Math.floor(Math.random() * parents.length)];
        } while (parent2 === parent1);
      }

      let child: Network;
      if (parent2) {
        child = parent1.crossover(parent2);
      } else {
        child = parent1.clone();
      }

      child.mutate(MUTATION_RATE, MUTATION_AMOUNT);
      newPopulation.push(child);
    }

    this.individuals = newPopulation;
    this.generation++;
  }

  async runGenerations(
    generations: number = 100,
    evaluateEvery: number = 10
  ): Promise<void> {
    await this.loadData();
    this.initializePopulation();

    for (let i = 0; i < generations; i++) {
      this.evaluateFitness();

      if (i % evaluateEvery === 0) {
        console.log(
          `Generation ${i}, Best Fitness: ${this.bestFitness.toFixed(4)}`
        );
      }

      this.crossoverAndMutate();
    }

    if (this.bestIndividual && this.validationData) {
      console.log(
        green(
          "Training complete. Evaluating best individual on validation data."
        )
      );
      let totalValidationLoss = 0;
      for (let i = 0; i < this.validationData.inputSequences.length; i++) {
        const inputSequence = this.validationData.inputSequences[i];
        const trueTarget = this.validationData.targetOutputs[i];
        const networkOutputs = this.bestIndividual.forward(inputSequence);
        const predictedOutput = networkOutputs[networkOutputs.length - 1];

        let sequenceLoss = 0;
        for (let j = 0; j < predictedOutput.length; j++) {
          sequenceLoss += Math.pow(trueTarget[j] - predictedOutput[j], 2);
        }
        totalValidationLoss += sequenceLoss;
      }
      const validationMse =
        totalValidationLoss / this.validationData.inputSequences.length;
      console.log(
        green(`Best Individual Validation MSE: ${validationMse.toFixed(6)}`)
      );

      if (this.trainingData && this.validationData.inputSequences.length > 0) {
        console.log("--- Sample Denormalized Predictions ---");
        const numSamples = Math.min(
          5,
          this.validationData.inputSequences.length
        );
        for (let i = 0; i < numSamples; i++) {
          const inputSeq = this.validationData.inputSequences[i];
          const trueTarget = this.validationData.targetOutputs[i];
          const networkOutputs = this.bestIndividual.forward(inputSeq);
          const predictedOutput = networkOutputs[networkOutputs.length - 1];

          const denormalizedPredicted =
            this.trainingData.scaler.inverseScaleSingle(
              predictedOutput[0],
              this.data.featureKeys.indexOf(this.data.targetKey)
            );
          const denormalizedTrue = this.trainingData.scaler.inverseScaleSingle(
            trueTarget[0],
            this.data.featureKeys.indexOf(this.data.targetKey)
          );
          console.log(
            `True: ${denormalizedTrue.toFixed(
              2
            )}, Predicted: ${denormalizedPredicted.toFixed(2)}`
          );
        }
      }
    }
  }
}
