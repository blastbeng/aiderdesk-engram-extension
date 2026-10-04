/**
 * Positional aliases for LLM round-trips.
 *
 * Production failure (all three consolidation runs, ~25/27 classification
 * rounds): prompts handed real memory UUIDs ("#0f3c...") and demanded they be
 * echoed back, but the secondary model paraphrased, truncated, or renumbered
 * them, so every action was dropped by the `byId.has(id)` guard and nothing
 * was ever merged, updated, or deleted. Short, sequential aliases (`m1`,
 * `m2`, ...) survive the round-trip; this module builds the alias table and
 * resolves whatever the model actually wrote back to the real id.
 *
 * Resolution is deliberately tolerant of how models format ids:
 *   'm3', 'M3', '#m3', '(m3)', 'm 3', 'm-3'  -> the entry behind alias m3
 *   '3'                                      -> prefix + digits, i.e. m3
 *   '0f3c9d62-...' (full real id, any case)  -> that entry
 * Anything else resolves to null and the caller counts it as skipped.
 */

export interface PromptEntry {
  id: string;
}

export interface AliasTable {
  /** Normalized alias or normalized real id -> real id. */
  readonly ids: ReadonlyMap<string, string>;
  /** Normalized real id -> the alias shown in the prompt. */
  readonly aliasOf: ReadonlyMap<string, string>;
  readonly prefix: string;
  /** Number of entries seated in the table. */
  readonly size: number;
}

function normalizeToken(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]/g, '');
}

export function buildAliasTable(entries: readonly PromptEntry[], prefix = 'm'): AliasTable {
  const ids = new Map<string, string>();
  const aliasOf = new Map<string, string>();
  entries.forEach((entry, index) => {
    const alias = `${prefix}${index + 1}`;
    ids.set(alias, entry.id);
    const normId = normalizeToken(entry.id);
    if (normId && !ids.has(normId)) {
      ids.set(normId, entry.id);
      aliasOf.set(normId, alias);
    }
  });
  return { ids, aliasOf, prefix, size: entries.length };
}

/** Canonical prompt line for one entry, using its positional alias. */
export function aliasFor(table: AliasTable, id: string): string {
  return table.aliasOf.get(normalizeToken(id)) ?? id;
}

/**
 * Resolve a value the model wrote back to the real memory id.
 * Accepts strings and numbers (a JSON `'targetIds': [3, 7]` must not be a
 * schema-level rejection), strips markdown emphasis, `#` prefixes and any
 * non-alphanumeric padding.
 */
export function resolveAlias(raw: unknown, table: AliasTable): string | null {
  if (raw === null || raw === undefined) return null;
  if (typeof raw !== 'string' && typeof raw !== 'number') return null;
  const collapsed = normalizeToken(String(raw).trim());
  if (!collapsed) return null;

  const direct = table.ids.get(collapsed);
  if (direct) return direct;

  // Bare digits: "3" means the alias `m3` (or any configured prefix).
  if (/^\d+$/.test(collapsed)) {
    const byDigits = table.ids.get(`${table.prefix}${collapsed}`);
    if (byDigits) return byDigits;
  }
  return null;
}
