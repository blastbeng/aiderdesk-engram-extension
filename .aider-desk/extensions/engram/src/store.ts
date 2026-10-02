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
import { decodeMemory, encodeMemory, memoryForPrompt, toEntryType, type MemoryMeta } from './memory-format';

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
 * de-duplicate by id, keep only entries this extension manages, and rank by
 * importance so importance 4-5 dominate the surviving slots.
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
  // Importance-first, then recency. Importance 4-5 therefore survive truncation.
  all.sort((a, b) => {
    const ia = importanceOf(a);
    const ib = importanceOf(b);
    if (ia !== ib) return ib - ia;
    return (b.timestamp ?? 0) - (a.timestamp ?? 0);
  });

  return all.slice(0, Math.max(1, limit));
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
