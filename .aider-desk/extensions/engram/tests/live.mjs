/**
 * Live test launcher (real OpenAI-compatible endpoint).
 *
 * Same jiti bootstrap as run.mjs, but for tests/live.ts. The jiti instance is
 * also handed to the test through globalThis.__engramJiti so it can load a
 * fresh copy of the extension from a temp directory with the same aliases.
 *
 * Usage:
 *   ENGRAM_LIVE_URL=http://host:4000/v1 \
 *   ENGRAM_LIVE_KEY=sk-... \
 *   ENGRAM_LIVE_MODEL=synthetic/syn:small:text \
 *   node tests/live.mjs
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

// live.ts uses this to dynamically import a cold extension copy (temp dir).
globalThis.__engramJiti = jiti;

try {
  await jiti.import('./live.ts', { default: true });
} catch (err) {
  console.error('[live] crashed:', err);
  process.exit(1);
}
