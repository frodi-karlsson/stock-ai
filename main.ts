import { GENERATIONS } from "./constants";
import { CSVData } from "./graph/data";
import { Population } from "./graph/population";
import { green, red } from "./util/console";
import fs from "node:fs";

interface StockDataRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

async function main() {
  const featureKeys: (keyof StockDataRow)[] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
  ];
  const csvData = new CSVData<StockDataRow, "close">(
    "data/GOOG.csv",
    "close",
    featureKeys
  );

  const population = new Population<StockDataRow, "close">(csvData);

  await population.runGenerations(GENERATIONS, 10);

  console.log(green("\nGenetic algorithm finished successfully!"));
  if (population.bestIndividual) {
    console.log(`Overall Best Fitness: ${population.bestFitness.toFixed(6)}`);
    const timeStr = new Date().toISOString().replace(/[:.]/g, "-");
    const dirname = "models";

    if (!fs.existsSync(dirname)) {
      fs.mkdirSync(dirname, { recursive: true });
    }

    const filename = `${dirname}/${population.bestFitness.toFixed(
      4
    )}-${timeStr}.json`;
    fs.writeFileSync(
      filename,
      JSON.stringify(population.bestIndividual.toJSON(), null, 2)
    );
    console.log(green(`Best individual saved to ${filename}`));
  } else {
    console.error(red("No best individual found after all generations."));
  }
}

main()
  .then(() => console.log(green("Main function completed successfully.")))
  .catch((error) => console.error(red("Error in main function:"), error));
