import assert from "node:assert";
import fs from "node:fs";

export type Data<
  T = Record<string, any>,
  TARGET_KEY extends keyof T = keyof T
> = {
  data: T[];
  targetKey: TARGET_KEY;
  featureKeys: (keyof T)[];

  getTarget: (item: T) => T[TARGET_KEY];
  getFeatures: (item: T) => number[];
};

export class CSVData<
  T = Record<string, any>,
  TARGET_KEY extends keyof T = keyof T
> implements Data<T, TARGET_KEY>
{
  data: T[];
  constructor(
    public path: string,
    public targetKey: TARGET_KEY,
    public featureKeys: (keyof T)[]
  ) {
    assert(fs.existsSync(path), `File not found: ${path}`);
    this.data = this.loadData();
  }

  private transformValue<V>(value: string): V {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      return numValue as V;
    }

    // Other types are not interesting at this time

    return value as V;
  }

  private loadData(): T[] {
    const content = fs.readFileSync(this.path, "utf-8");
    const lines = content.trim().split("\n");
    const headers = lines[0]
      .split(",")
      .map((header) => header.trim() as keyof T);
    return lines.slice(1).map((line) => {
      const values = line.split(",").map((value) => value.trim());
      return headers.reduce((obj, header, index) => {
        obj[header] = this.transformValue<T[keyof T]>(values[index]);
        return obj;
      }, {} as T);
    });
  }

  getTarget(item: T): T[TARGET_KEY] {
    return item[this.targetKey];
  }

  getFeatures(item: T): number[] {
    const features: number[] = [];
    for (const key of this.featureKeys) {
      const value = item[key];
      assert(
        typeof value === "number",
        `Feature value for key "${String(
          key
        )}" must be a number, got: ${typeof value}`
      );
      features.push(value);
    }

    return features;
  }
}
