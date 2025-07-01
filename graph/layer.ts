import assert from "node:assert";
import { LSTMBlock } from "./block";
import { Neuron } from "./neuron";
import { ForwardState } from "./shared";

export interface ILayer {
  inputSize: number;
  outputSize: number;

  activate(
    inputs: number[],
    previousStates?: ForwardState[]
  ): number[] | ForwardState[];

  clone(): ILayer;
  mutate(mutationRate?: number, mutationAmount?: number): void;
  crossover(other: ILayer): ILayer;
  toJSON(): any;
}

export class LSTMLayer implements ILayer {
  blocks: LSTMBlock[] = [];

  constructor(
    public inputSize: number,
    public outputSize: number,
    blocks?: LSTMBlock[]
  ) {
    this.blocks =
      blocks ||
      Array.from({ length: outputSize }, () =>
        LSTMBlock.createRandom(inputSize)
      );
  }

  activate(
    input: number[],
    previousStates: { hiddenState: number; cellState: number }[]
  ): { hiddenState: number; cellState: number }[] {
    return this.blocks.map((block, index) => {
      const previousState = previousStates[index];
      assert(
        previousState,
        `Previous state for block ${index} must be defined`
      );
      return block.forward(input, previousState);
    });
  }

  clone(): ILayer {
    return new LSTMLayer(
      this.inputSize,
      this.outputSize,
      this.blocks.map((block) => block.clone())
    );
  }

  mutate(mutationRate: number = 0.1, mutationAmount: number = 0.1): void {
    this.blocks = this.blocks.map((block) =>
      block.mutate(mutationRate, mutationAmount)
    );
  }

  crossover(other: ILayer): ILayer {
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

    assert(
      LSTMLayer.is(other),
      "Other layer must be an instance of LSTMLayer for crossover"
    );

    const newBlocks = this.blocks.map((block, index) =>
      block.crossover(other.blocks[index])
    );

    return new LSTMLayer(this.inputSize, this.outputSize, newBlocks);
  }

  toJSON(): any {
    return {
      type: "LSTMLayer",
      inputSize: this.inputSize,
      outputSize: this.outputSize,
      blocks: this.blocks.map((block) => block.toJSON()),
    };
  }

  static fromJSON(json: any): LSTMLayer {
    assert(json.type === "LSTMLayer", "JSON must represent an LSTMLayer");
    const blocks = json.blocks.map((blockJson: any) =>
      LSTMBlock.fromJSON(blockJson)
    );
    return new LSTMLayer(json.inputSize, json.outputSize, blocks);
  }

  static createRandom(inputSize: number, outputSize: number): LSTMLayer {
    const blocks = Array.from({ length: outputSize }, () =>
      LSTMBlock.createRandom(inputSize)
    );
    return new LSTMLayer(inputSize, outputSize, blocks);
  }

  static is(layer: ILayer): layer is LSTMLayer {
    return layer instanceof LSTMLayer;
  }
}

export class DenseLayer implements ILayer {
  neurons: Neuron[];

  constructor(
    public inputSize: number,
    public outputSize: number,
    neurons?: Neuron[]
  ) {
    this.neurons =
      neurons ||
      Array.from({ length: outputSize }, () => {
        return new Neuron(inputSize, (x) => x); // We only use it for output, so all of them can be linear
      });
  }

  activate(inputs: number[]): number[] {
    // Takes inputs, returns outputs
    assert(
      inputs.length === this.inputSize,
      `DenseLayer input length mismatch: expected ${this.inputSize}, got ${inputs.length}`
    );
    return this.neurons.map((neuron) => neuron.activate(inputs));
  }

  clone(): ILayer {
    const cloned = new DenseLayer(this.inputSize, this.outputSize);
    cloned.neurons = this.neurons.map((neuron) => neuron.clone());
    return cloned;
  }

  mutate(mutationRate: number = 0.1, mutationAmount: number = 0.1): void {
    this.neurons.forEach((neuron) =>
      neuron.mutate(mutationRate, mutationAmount)
    );
  }

  crossover(other: ILayer): ILayer {
    assert(
      this.inputSize === other.inputSize &&
        this.outputSize === other.outputSize,
      "Dense layers must have compatible sizes for crossover."
    );

    assert(
      DenseLayer.is(other),
      "Other layer must be an instance of DenseLayer for crossover"
    );

    const childNeurons = this.neurons.map((neuron, index) =>
      neuron.crossover(other.neurons[index])
    ); // Assuming Neuron.crossover
    return new DenseLayer(this.inputSize, this.outputSize, childNeurons);
  }

  toJSON(): any {
    return {
      type: "DenseLayer",
      inputSize: this.inputSize,
      outputSize: this.outputSize,
      neurons: this.neurons.map((neuron) => neuron.toJSON()),
    };
  }

  static fromJSON(json: any): DenseLayer {
    assert(json.type === "DenseLayer", "JSON must represent a DenseLayer");
    const neurons = json.neurons.map((neuronJson: any) =>
      Neuron.fromJSON(neuronJson)
    );
    return new DenseLayer(json.inputSize, json.outputSize, neurons);
  }

  static createRandom(inputSize: number, outputSize: number): DenseLayer {
    return new DenseLayer(inputSize, outputSize);
  }

  static is(layer: ILayer): layer is DenseLayer {
    return layer instanceof DenseLayer;
  }
}
