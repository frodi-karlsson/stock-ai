import assert from "node:assert";
import { formatArgs } from "./console";

export function testFormatArgs() {
  class CustomError extends Error {
    constructor(message: string) {
      super(message);
    }
  }

  const testCases = [
    { input: ["Hello", "World"], expected: "Hello World" },
    { input: [123, 456], expected: "123 456" },
    { input: [true, false], expected: "true false" },
    { input: [null, undefined], expected: "null undefined" },
    {
      input: [{ key: "value" }],
      expected: JSON.stringify({ key: "value" }, null, 2),
    },
    {
      input: [new Date("2023-01-01")],
      expected: new Date("2023-01-01").toISOString(),
    },
    { input: [Symbol("test")], expected: "Symbol(test)" },
    { input: [function test() {}], expected: "function test(){}" },
    { input: [new Error("Test error")], expected: "Error: Test error" },
    {
      input: [new CustomError("Custom error")],
      expected: "CustomError: Custom error",
    },
    {
      input: [{ a: 1, b: 2 }, "extra string"],
      expected: JSON.stringify({ a: 1, b: 2 }, null, 2) + " extra string",
    },
    { input: [], expected: "" },
    { input: [undefined], expected: "undefined" },
    { input: [null], expected: "null" },
  ];

  testCases.forEach(({ input, expected }) => {
    const result = formatArgs(input);
    assert.strictEqual(
      result,
      expected,
      `Failed for input: ${JSON.stringify(input)}`
    );
  });
}
