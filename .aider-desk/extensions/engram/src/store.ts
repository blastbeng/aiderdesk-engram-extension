/**
 * Thin, defensive wrapper over AiderDesk's native MemoryContext.
 *
 * Native API at v0.81.0 (packages/common/src/extensions.ts, line 1603):
 *   storeMemory(projectId, taskId, type, content): Promise<string>   // '' on failure
 *   retrieveMemories(projectId, query, limit?): Promise<MemoryEntry[]>
 *   getMemory(id): Promise<MemoryEntry | null>
 *   deleteMemory(id): Promise<boolean>
 *   updateMemory(id, content): Promise<boolean>
 *   getAllMemories(): Promise<MemoryEntry[]>
 *   isMemoryEnabled(): boolean
 *   setMemoryEnabled(enabled): void
 *
 * Two native constraints this module works around:
 *  1. retrieveMemories filters EXACTLY on `projectid = projectId`, so global
 *     memories (stored with projectId '') need a second query.
 *  2. The relevance cutoff is the GLOBAL `memory.maxDistance` setting; the
 *     returned MemoryEntry carries no distance, so callers cannot re-filter.
 */
import type { ExtensionContext, MemoryContext, MemoryEntry } from '@aiderdesk/extensions';
import {
  decodeMemory,
  encodeMemory,
  memoryForPrompt,
  normalizeStatement,
  toEntryType,
  type MemoryMeta,
} from './memory-format';

export interface ScopedMemories {
  project: MemoryEntry[];
  global: MemoryEntry[];
}

export function getMemoryContextSafely(context: ExtensionContext): MemoryContext | null {
  try {
    const memory = context.getMemoryContext();
    if (!memory || !memory.isMemoryEnabled()) return null;
    return memory;
  } catch {
    // getMemoryContext() throws when the MemoryManager is unavailable.
    return null;
  }
}

/**
 * Query project memories and (optionally) global memories, then merge,
 * de-duplicate by id, and keep only entries this extension manages. Ranking is
 * RELEVANCE-PRIMARY: the native vector-search order (nearest first) is kept, so
 * what reaches the context is what the user is actually asking about. Importance
 * is enforced as a floor (retrieval.min_importance), never as a sort - an
 * importance-first sort evicted exactly the memories the query was about.
 */
export async function retrieveScoped(
  memory: MemoryContext,
  projectDir: string,
  query: string,
  limit: number,
  includeGlobal: boolean,
  managedOnly = true,
): Promise<MemoryEntry[]> {
  const merged = new Map<string, MemoryEntry>();

  const push = (entries: MemoryEntry[]) => {
    for (const entry of entries) {
      if (!entry?.id) continue;
      if (managedOnly && !isManaged(entry)) continue;
      if (!merged.has(entry.id)) merged.set(entry.id, entry);
    }
  };

  try {
    push(await memory.retrieveMemories(projectDir, query, Math.max(1, limit)));
  } catch {
    /* native store unavailable - degrade silently */
  }

  if (includeGlobal && projectDir) {
    try {
      push(await memory.retrieveMemories('', query, Math.max(1, limit)));
    } catch {
      /* ignore */
    }
  }

  const all = Array.from(merged.values());
  // Keep the native relevance order (nearest first). No re-ranking: importance
  // filtering happens afterwards in retrieval.ts via retrieval.min_importance,
  // so truncation to `limit` keeps the most relevant hits, not the loudest ones.

  // Deterministic statement-level dedup after ranking: keep the highest-ranked
  // copy of a fact and drop exact (normalized) duplicates. A verbatim
  // restatement that slipped past classification must never occupy two slots
  // of the limit or appear twice in a dedup corpus.
  const seenStatements = new Set<string>();
  const unique: MemoryEntry[] = [];
  for (const entry of all) {
    const key = normalizeStatement(statementOf(entry));
    if (key) {
      if (seenStatements.has(key)) continue;
      seenStatements.add(key);
    }
    unique.push(entry);
  }

  return unique.slice(0, Math.max(1, limit));
}

export function isManaged(entry: MemoryEntry): boolean {
  return decodeMemory(entry.content) !== null;
}

export function importanceOf(entry: MemoryEntry): number {
  const decoded = decodeMemory(entry.content);
  return decoded ? decoded.meta.importance : 3;
}

export function scopeOf(entry: MemoryEntry): 'global' | 'project' {
  const decoded = decodeMemory(entry.content);
  if (decoded) return decoded.meta.scope;
  return entry.projectId ? 'project' : 'global';
}

export function statementOf(entry: MemoryEntry): string {
  const decoded = decodeMemory(entry.content);
  return decoded ? decoded.statement : entry.content.trim();
}

export function metaForNew(meta: Omit<MemoryMeta, 'createdAt' | 'updatedAt'>): MemoryMeta {
  return { ...meta, createdAt: Date.now() };
}

/**
 * Meta for a write that REWRITES an existing memory (UPDATE / CONFLICT /
 * OBSOLETE / MERGE). `metaForNew` stamps createdAt with now, which silently
 * erased the age of every memory the system updated - a memory consolidated
 * ten times looked ten minutes old, and "oldest first" consolidation ordering
 * degraded to the order updates happened to arrive in. The original creation
 * time is provenance: carry it over, and fall back to now only when the
 * previous footer is missing or unreadable.
 */
export function metaForUpdate(
  previous: MemoryMeta | null | undefined,
  meta: Omit<MemoryMeta, 'createdAt' | 'updatedAt'>,
): MemoryMeta {
  const createdAt =
    typeof previous?.createdAt === 'number' && previous.createdAt > 0 ? previous.createdAt : Date.now();
  return { ...meta, createdAt };
}

export async function storeNew(
  memory: MemoryContext,
  projectDir: string,
  taskId: string,
  statement: string,
  meta: MemoryMeta,
): Promise<string> {
  const type = toEntryType(meta.category);
  const projectId = meta.scope === 'global' ? '' : projectDir;
  const id = await memory.storeMemory(projectId, taskId, type, encodeMemory(statement, meta));
  return typeof id === 'string' ? id : '';
}

export async function updateExisting(
  memory: MemoryContext,
  id: string,
  statement: string,
  meta: MemoryMeta,
): Promise<boolean> {
  const withUpdate: MemoryMeta = { ...meta, updatedAt: Date.now() };
  const ok = await memory.updateMemory(id, encodeMemory(statement, withUpdate));
  return ok === true;
}

export async function remove(memory: MemoryContext, id: string): Promise<boolean> {
  const ok = await memory.deleteMemory(id);
  return ok === true;
}

export async function listAll(memory: MemoryContext): Promise<MemoryEntry[]> {
  try {
    const entries = await memory.getAllMemories();
    return Array.isArray(entries) ? entries : [];
  } catch {
    return [];
  }
}

export function linesForPrompt(entries: MemoryEntry[]): string[] {
  return entries.map((e) => memoryForPrompt(e.id, e.content));
}

/**
 * Id-free statement list for the extraction prompt's do-not-restate corpus.
 * Those memories are shown only so the model avoids restating them - they are
 * never referenced by id - so short aliases would be noise there; plain
 * statements keep the prompt small.
 */
export function statementsForPrompt(entries: MemoryEntry[]): string[] {
  return entries
    .map((e) => statementOf(e).replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .slice(0, 200);
}

export interface DedupReport {
  /** Engram-managed entries examined. */
  scanned: number;
  /** Exact (normalized) duplicates removed. */
  removed: number;
  /**
   * Removed entries counted per project id ('' = global scope). Dedup scans
   * every project at once, but callers keep per-project stats: without this
   * breakdown a removal in another project would be booked to the caller's.
   */
  removedByProject: Record<string, number>;
}

/**
 * Deterministic exact-match dedup across every Engram-managed memory. Entries
 * are grouped by (project scope, normalized statement); within a group the best
 * copy (highest importance, then newest) is kept and the rest are deleted. No
 * LLM is involved: idempotent, cheap, safe to run at any time, and it only ever
 * touches rows this extension created (footer-decodable content).
 */
export async function deterministicDedup(memory: MemoryContext): Promise<DedupReport> {
  const all = (await listAll(memory)).filter((entry) => entry?.id && isManaged(entry));
  const groups = new Map<string, MemoryEntry[]>();
  for (const entry of all) {
    const statement = normalizeStatement(statementOf(entry));
    if (!statement) continue;
    const key = `${entry.projectId ?? ''}|${statement}`;
    const group = groups.get(key);
    if (group) group.push(entry);
    else groups.set(key, [entry]);
  }
  let removed = 0;
  const removedByProject: Record<string, number> = {};
  for (const group of groups.values()) {
    if (group.length < 2) continue;
    const best = [...group].sort((a, b) => {
      const ia = importanceOf(a);
      const ib = importanceOf(b);
      if (ia !== ib) return ib - ia;
      return (b.timestamp ?? 0) - (a.timestamp ?? 0);
    })[0];
    for (const entry of group) {
      if (entry.id === best.id) continue;
      if (await remove(memory, entry.id)) {
        removed += 1;
        const scope = entry.projectId ?? '';
        removedByProject[scope] = (removedByProject[scope] ?? 0) + 1;
      }
    }
  }
  return { scanned: all.length, removed, removedByProject };
}
