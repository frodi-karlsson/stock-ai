export function formatArgs(args: any[]): string {
  return args
    .map((arg) => {
      if (
        ["string", "number", "boolean"].includes(typeof arg) ||
        [null, undefined].includes(arg)
      ) {
        return String(arg);
      }
      if (arg instanceof Error) {
        const errorClass = arg.constructor.name;
        return `${errorClass}: ${arg.message}`;
      }
      if (
        Object.prototype.toString.call(arg) === "[object Symbol]" ||
        typeof arg === "function"
      ) {
        return arg.toString();
      }
      if (arg instanceof Date) {
        return arg.toISOString();
      }
      try {
        return JSON.stringify(arg, null, 2);
      } catch {
        return String(arg);
      }
    })
    .join(" ");
}

export function green(...args: any[]): string {
  return `\x1b[32m${formatArgs(args)}\x1b[0m`;
}

export function red(...args: any[]): string {
  return `\x1b[31m${formatArgs(args)}\x1b[0m`;
}
