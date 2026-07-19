declare module 'node:url' {
  export class URL {
    constructor(input: string, base?: string | URL);
  }

  export function fileURLToPath(url: string | URL): string;
}

declare const process: {
  env: Record<string, string | undefined>;
};
