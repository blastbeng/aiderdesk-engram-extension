/**
 * In-memory stand-in for AiderDesk's native MemoryContext, used only by the
 * offline test harness (tests/run.ts).
 *
 * It reproduces the two behaviours of the real v0.81.0 MemoryManager that the
 * extension depends on, and nothing else:
 *
 *  1. retrieveMemories(projectId, query, limit) filters EXACTLY on
 *     `projectid = projectId` (so global memories, stored with projectId '',
 *     are only reachable through a query with ''), and returns the top `limit`
 *     by similarity.
 *  2. MemoryEntry has exactly { id, content, type, taskId?, projectId?,
 *     timestamp } - no metadata column.
 *
 * Similarity is a lexical token-overlap score standing in for the native
 * embedding search. The real store applies a global `memory.maxDistance`
 * cutoff; here a low floor keeps the harness deterministic.
 */
import type { MemoryContext, MemoryEntry } from '@aiderdesk/extensions';

export type MockMemoryType = 'task' | 'user-preference' | 'code-pattern';

interface StoredEntry {
  id: string;
  content: string;
  type: string;
  taskId: string;
  projectId: string;
  timestamp: number;
}

const STOP = new Set([
  'the','a','an','and','or','is','are','for','to','of','in','on','with','this','that','it','we','you',
  'must','be','as','at','by','from','not','no','yes','all','any','will','can','has','have','was',
]);

function tokens(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9_]+/g, ' ')
      .split(' ')
      .filter((t) => t.length > 2 && !STOP.has(t)),
  );
}

function similarity(query: string, content: string): number {
  const a = tokens(query);
  const b = tokens(content);
  if (!a.size || !b.size) return 0;
  let shared = 0;
  for (const t of a) if (b.has(t)) shared += 1;
  return shared / Math.sqrt(a.size * b.size);
}

export class MockMemoryContext implements MemoryContext {
  private entries = new Map<string, StoredEntry>();
  private nextId = 1;
  private enabled = true;

  /** Every call the extension made, for assertions about API usage. */
  readonly calls: string[] = [];

  isMemoryEnabled(): boolean {
    return this.enabled;
  }

  setMemoryEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  async storeMemory(projectId: string, taskId: string, type: string, content: string): Promise<string> {
    this.calls.push(`storeMemory(${JSON.stringify(projectId)},${type})`);
    const id = `mem-${this.nextId++}`;
    this.entries.set(id, {
      id,
      content,
      type,
      taskId: taskId ?? '',
      projectId: projectId ?? '',
      timestamp: Date.now() + this.entries.size,
    });
    return id;
  }

  async retrieveMemories(projectId: string, query: string, limit?: number): Promise<MemoryEntry[]> {
    this.calls.push(`retrieveMemories(${JSON.stringify(projectId)})`);
    const cap = Math.max(1, limit ?? 10);
    return Array.from(this.entries.values())
      .filter((e) => (e.projectId ?? '') === (projectId ?? ''))
      .map((e) => ({ entry: e, score: similarity(query, e.content) }))
      .filter((x) => x.score > 0.02)
      .sort((a, b) => b.score - a.score)
      .slice(0, cap)
      .map((x) => this.toEntry(x.entry));
  }

  async getMemory(id: string): Promise<MemoryEntry | null> {
    const found = this.entries.get(id);
    return found ? this.toEntry(found) : null;
  }

  async deleteMemory(id: string): Promise<boolean> {
    this.calls.push(`deleteMemory(${id})`);
    return this.entries.delete(id);
  }

  async updateMemory(id: string, content: string): Promise<boolean> {
    this.calls.push(`updateMemory(${id})`);
    const found = this.entries.get(id);
    if (!found) return false;
    found.content = content;
    return true;
  }

  async getAllMemories(): Promise<MemoryEntry[]> {
    return Array.from(this.entries.values()).map((e) => this.toEntry(e));
  }

  /** Test-only helpers. */
  count(): number {
    return this.entries.size;
  }

  statements(): string[] {
    return Array.from(this.entries.values()).map((e) => e.content);
  }

  reset(): void {
    this.entries.clear();
    this.calls.length = 0;
  }

  private toEntry(e: StoredEntry): MemoryEntry {
    return {
      id: e.id,
      content: e.content,
      type: e.type as MemoryEntry['type'],
      taskId: e.taskId,
      projectId: e.projectId,
      timestamp: e.timestamp,
    };
  }
}
