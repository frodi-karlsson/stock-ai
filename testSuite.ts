import assert from "node:assert";
import { green, red } from "./util/console";
import fs from "node:fs";

// I muleishly didn't want any dependencies outside of node so no testing libraries to be found.

const files = fs
  .readdirSync(".", { recursive: true })
  .filter((file) => {
    const fileStr = file.toString();
    return fileStr.endsWith(".test.ts");
  })
  .map((file) => `./${file}`);

const modules = await Promise.all(
  files.map(async (file) => ({
    [file]: await import(file),
  }))
);

const tests: Record<string, Function> = {};

for (const module of modules) {
  for (const [key, value] of Object.entries(module[Object.keys(module)[0]])) {
    if (typeof value === "function" && key.startsWith("test")) {
      tests[key] = value;
    }
  }
}

function runTests() {
  const failedTests: string[] = [];
  for (const [testName, testFn] of Object.entries(tests)) {
    try {
      testFn();
      console.log(green(`Test ${testName} passed.`));
    } catch (error) {
      console.error(red(`Test ${testName} failed:`, error));
      failedTests.push(testName);
    }
  }

  assert(!failedTests.length, `Some tests failed: ${failedTests.join(", ")}`);
  console.log(green("All tests passed successfully!"));
}

runTests();
