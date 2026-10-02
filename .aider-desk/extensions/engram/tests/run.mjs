/**
 * Engram offline test harness launcher.
 *
 * Boots jiti (the same TS loader AiderDesk 0.81.0 uses for extensions) and
 * runs tests/run.ts through it, so the harness executes the exact extension
 * sources — TypeScript, ESM imports and zod schemas included.
 *
 * Usage:
 *   node tests/run.mjs
 *   ./node_modules/.bin/jiti tests/run.ts      (equivalent, direct jiti)
 */
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const jitiPath = require.resolve('jiti');
const { createJiti } = await import(jitiPath);

const here = dirname(fileURLToPath(import.meta.url));

const jiti = createJiti(import.meta.url, {
  alias: {
    '@aiderdesk/extensions': join(
      here,
      '../node_modules/@aiderdesk/extensions/dist/index.js',
    ),
    zod: join(here, '../node_modules/zod'),
  },
});

try {
  await jiti.import('./run.ts', { default: true });
} catch (err) {
  console.error('[harness] crashed:', err);
  process.exit(1);
}
