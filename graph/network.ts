import assert from "node:assert";
import { ForwardState } from "./shared";
import { Neuron } from "./neuron";
import { DenseLayer, ILayer, LSTMLayer } from "./layer";

export class Network {
  layers: ILayer[] = [];
  fitness: number = Number.NEGATIVE_INFINITY;

  constructor(public inputSize: number, public outputSize: number) {}

  addLayer(layer: ILayer): void {
    if (this.layers.length) {
      assert.strictEqual(
        layer.inputSize,
        this.layers[this.layers.length - 1].outputSize,
        "Layer input size must match previous layer output size"
      );
    } else {
      assert.strictEqual(
        layer.inputSize,
        this.inputSize,
        "Layer input size must match network input size"
      );
    }

    this.layers.push(layer);
  }

  forward(sequenceInputs: number[][]): number[][] {
    assert(sequenceInputs.length, "Input sequence must not be empty");
    assert.strictEqual(
      sequenceInputs[0].length,
      this.inputSize,
      `Input size must match network input size: ${this.inputSize}`
    );

    const lstmLayerStates: ForwardState[][] = this.layers
      .filter(LSTMLayer.is)
      .map((layer) => {
        return Array.from({ length: layer.outputSize }, () => ({
          hiddenState: 0,
          cellState: 0,
        }));
      });

    const networkOutputs: number[][] = [];

    for (const currentInput of sequenceInputs) {
      let inputToCurrentLayer = currentInput;

      let currentLSTMLayerIndex = 0;

      for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex++) {
        const layer = this.layers[layerIndex];

        if (LSTMLayer.is(layer)) {
          const previousStates = lstmLayerStates[currentLSTMLayerIndex];

          const newStates = layer.activate(inputToCurrentLayer, previousStates);

          lstmLayerStates[currentLSTMLayerIndex] = newStates;
          inputToCurrentLayer = newStates.map((state) => state.hiddenState);
          currentLSTMLayerIndex++;
        } else if (DenseLayer.is(layer)) {
          inputToCurrentLayer = layer.activate(inputToCurrentLayer);
        } else {
          assert.fail(
            `Unsupported layer type: ${layer.constructor.name}. Only DenseLayer and LSTMLayer are supported.`
          );
        }
      }

      networkOutputs.push(inputToCurrentLayer);
    }

    assert.strictEqual(
      networkOutputs.length,
      sequenceInputs.length,
      `Output length must match input sequence length. Expected: ${sequenceInputs.length}, Got: ${networkOutputs.length}`
    );

    if (networkOutputs.length) {
      assert.strictEqual(
        networkOutputs[0].length,
        this.layers[this.layers.length - 1].outputSize,
        `Output size must match last layer output size: ${
          this.layers[this.layers.length - 1].outputSize
        }. Got: ${networkOutputs[0].length}`
      );
    }

    return networkOutputs;
  }

  clone(): Network {
    const clonedNetwork = new Network(this.inputSize, this.outputSize);
    clonedNetwork.layers = this.layers.map((layer) => layer.clone());
    return clonedNetwork;
  }

  mutate(mutationRate: number = 0.1, mutationAmount: number = 0.1): void {
    this.layers.forEach((layer) => layer.mutate(mutationRate, mutationAmount));
  }

  crossover(other: Network): Network {
    assert.strictEqual(
      this.inputSize,
      other.inputSize,
      "Input size must match for crossover"
    );
    assert.strictEqual(
      this.outputSize,
      other.outputSize,
      "Output size must match for crossover"
    );
    assert.strictEqual(
      this.layers.length,
      other.layers.length,
      "Cannot crossover Networks with different numbers of layers"
    );

    const newNetwork = new Network(this.inputSize, this.outputSize);
    const layerCount = Math.max(this.layers.length, other.layers.length);

    for (let i = 0; i < layerCount; i++) {
      const layerA = this.layers[i];
      const layerB = other.layers[i];

      if (LSTMLayer.is(layerA) && LSTMLayer.is(layerB)) {
        newNetwork.addLayer(layerA.crossover(layerB));
      } else if (DenseLayer.is(layerA) && DenseLayer.is(layerB)) {
        newNetwork.addLayer(layerA.crossover(layerB));
      } else {
        if (layerA && layerB) {
          assert.fail(
            `Cannot crossover layers of different types at index ${i}: ` +
              `${layerA.constructor.name} vs ${layerB.constructor.name}`
          );
        } else if (layerA) {
          newNetwork.addLayer(layerA.clone());
        } else if (layerB) {
          newNetwork.addLayer(layerB.clone());
        } else {
          assert.fail(
            `Unexpected state: Both layers missing at index ${i} during crossover.`
          );
        }
      }
    }

    return newNetwork;
  }

  static createRandom(
    featuresPerTimeStep: number,
    hiddenUnitsPerLayer: number,
    numLSTMLayers: number,
    finalOutputSize: number
  ): Network {
    const network = new Network(featuresPerTimeStep, finalOutputSize);

    let currentFeatureSize = featuresPerTimeStep;
    for (let i = 0; i < numLSTMLayers; i++) {
      const layer = LSTMLayer.createRandom(
        currentFeatureSize,
        hiddenUnitsPerLayer
      );
      network.addLayer(layer);
      currentFeatureSize = hiddenUnitsPerLayer; // Next layer input size is the previous layer's output size
    }

    const finalLayer = DenseLayer.createRandom(
      currentFeatureSize,
      finalOutputSize
    );
    network.addLayer(finalLayer);

    return network;
  }

  toJSON(): Record<string, any> {
    return {
      inputSize: this.inputSize,
      outputSize: this.outputSize,
      layers: this.layers.map((layer) => layer.toJSON()),
      fitness: this.fitness,
    };
  }

  static fromJSON(json: Record<string, any>): Network {
    const network = new Network(json.inputSize, json.outputSize);
    network.fitness = json.fitness;

    for (const layerData of json.layers) {
      let layer: ILayer;
      if (LSTMLayer.is(layerData)) {
        layer = LSTMLayer.fromJSON(layerData);
      } else if (DenseLayer.is(layerData)) {
        layer = DenseLayer.fromJSON(layerData);
      } else {
        assert.fail(
          `Unsupported layer type in JSON: ${layerData.constructor.name}`
        );
      }
      network.addLayer(layer);
    }

    return network;
  }
}
