/**
 * Activity feed: a small, bounded, in-memory record of what the extension is
 * doing, so the user can SEE it instead of guessing from a silent install.
 *
 * Two things end up here:
 *  - every line the extension logs (src/logger.ts forwards them through
 *    Logger.setSink), and
 *  - a handful of lifecycle entries index.ts records explicitly.
 *
 * Safety rules:
 *  - log lines are counts, ids, categories and short labels. Never conversation
 *    content, never secrets, never a model's raw output.
 *  - endpoint URLs are passed through `safeEndpoint()`, so credentials embedded
 *    in a base URL (http://user:pass@host/v1) never reach the viewer.
 *  - the memory statements a viewer user wants to see (what was injected, what
 *    was written) are attached ONLY when `ui.show_content` is on, and ONLY
 *    through `sanitizeStatements()`, which redacts secrets unconditionally.
 *
 * Nothing is written to disk: the feed is a debugging surface for the current
 * session, not a log file. Restarting AiderDesk starts it empty.
 */

import { redactSecrets } from './privacy';

export type ActivityLevel = 'debug' | 'info' | 'warn' | 'error';

/**
 * What the entry belongs to. The feed uses this for badges and filtering, so a
 * single flat stream still reads as "retrieval did this, extraction did that".
 */
export type ActivityKind =
  | 'lifecycle'
  | 'config'
  | 'retrieval'
  | 'extraction'
  | 'consolidation'
  | 'store'
  | 'command'
  | 'log';

/** Which task/project/agent an entry belongs to, and which operation produced it. */
export interface ActivityScope {
  kind?: ActivityKind;
  projectDir?: string;
  taskId?: string;
  agentId?: string | null;
}

/** Compact, machine-readable numbers attached to a digest entry. */
export type ActivityDetails = Record<string, string | number | boolean>;

export interface ActivityEntry {
  /** Monotonic id, so a viewer can poll incrementally. */
  id: number;
  /** Epoch ms. */
  ts: number;
  level: ActivityLevel;
  kind: ActivityKind;
  message: string;
  projectDir?: string;
  taskId?: string;
  agentId?: string;
  details?: ActivityDetails;
  /**
   * The memory statements this entry is about - what was injected into the
   * prompt, what was written to the store, what a merge produced. Present only
   * when the user turned on `ui.show_content` (see index.ts `contentFor`), and
   * always run through `sanitizeStatements` first: secrets redacted, whitespace
   * collapsed, each line capped, the list capped.
   */
  content?: string[];
}

export type ActivityScopeFilter = 'task' | 'project' | 'all';

export interface ActivityQuery {
  /** 'task' (default), 'project' or 'all'. */
  scope?: ActivityScopeFilter;
  taskId?: string;
  projectDir?: string;
  /** Drop entries below this level. */
  minLevel?: ActivityLevel;
  /** Keep only the newest N entries. */
  limit?: number;
}

const LEVEL_ORDER: Record<ActivityLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };

export const DEFAULT_ACTIVITY_LIMIT = 200;
const MIN_LIMIT = 20;
const MAX_LIMIT = 5000;

function normalizeLimit(limit: number, fallback = DEFAULT_ACTIVITY_LIMIT): number {
  const n = Math.round(limit);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.min(MAX_LIMIT, Math.max(MIN_LIMIT, n));
}

export interface ActivityRecordInput {
  level: ActivityLevel;
  message: string;
  kind?: ActivityKind;
  projectDir?: string;
  taskId?: string;
  agentId?: string | null;
  details?: ActivityDetails;
  /** Statements to show under this entry. Already sanitized by the caller. */
  content?: string[];
}

/** Cap on statements attached to one entry, and on how long one may be. */
const MAX_STATEMENTS = 12;
const MAX_STATEMENT_CHARS = 220;

/**
 * Make memory statements safe to show in a UI surface: secrets redacted
 * (always, independent of `privacy.redact_secrets` — the viewer is a display
 * surface, not the user's own store), whitespace collapsed to one line, each
 * statement capped, and the list capped.
 *
 * The statements themselves are the point of this feature: "what did the
 * secondary LLM just put into Memory, and what did it just inject into my
 * prompt". Redaction is the price of showing them, so it is not optional.
 */
export function sanitizeStatements(texts: readonly string[]): string[] {
  const out: string[] = [];
  for (const raw of texts) {
    if (typeof raw !== 'string') continue;
    const collapsed = redactSecrets(raw.replace(/\s+/g, ' ').trim(), true).text;
    if (!collapsed) continue;
    out.push(collapsed.length > MAX_STATEMENT_CHARS ? `${collapsed.slice(0, MAX_STATEMENT_CHARS - 1)}…` : collapsed);
    if (out.length >= MAX_STATEMENTS) break;
  }
  return out;
}

/**
 * A fixed-capacity ring of activity entries. Old entries are dropped, so a
 * long-lived desktop session cannot grow this without bound.
 */
export class ActivityLog {
  private limit: number;
  private buffer: ActivityEntry[] = [];
  private nextId = 1;
  private scope: ActivityScope = {};

  constructor(limit: number = DEFAULT_ACTIVITY_LIMIT) {
    this.limit = normalizeLimit(limit);
  }

  /** Re-cap the ring (Settings > Engram > activity_limit). */
  setLimit(limit: number): void {
    this.limit = normalizeLimit(limit, this.limit);
    this.trim();
  }

  getLimit(): number {
    return this.limit;
  }

  /**
   * Tag everything recorded from now on with the operation being run.
   *
   * index.ts sets this around each unit of work (retrieval, an extraction
   * round, a consolidation run), so the log mirror - which knows nothing about
   * tasks - still lands in the feed attributed to the right task, project,
   * agent and kind. Always paired with clearScope() in a finally block.
   */
  setScope(scope: ActivityScope): void {
    this.scope = { ...scope };
  }

  clearScope(): void {
    this.scope = {};
  }

  /**
   * Run `fn` with `scope` current, then RESTORE whatever scope was current
   * before. `setScope`/`clearScope` around an unawaited task (the startup
   * probe) used to clobber the scope a concurrent extraction had set, so its
   * digest entry lost its taskId and vanished from a task-scoped view. This
   * save/restore form cannot: the concurrent operation's scope survives.
   */
  withScope<T>(scope: ActivityScope, fn: () => T): T {
    const previous = this.scope;
    this.scope = { ...scope };
    try {
      return fn();
    } finally {
      this.scope = previous;
    }
  }

  getScope(): ActivityScope {
    return { ...this.scope };
  }

  record(input: ActivityRecordInput): ActivityEntry {
    const scope = this.scope;
    const entry: ActivityEntry = {
      id: this.nextId++,
      ts: Date.now(),
      level: input.level,
      kind: input.kind ?? scope.kind ?? 'log',
      message: input.message,
      projectDir: input.projectDir ?? scope.projectDir,
      taskId: input.taskId ?? scope.taskId,
      agentId: input.agentId ?? scope.agentId ?? undefined,
    };
    if (input.details && Object.keys(input.details).length) entry.details = input.details;
    if (input.content && input.content.length) entry.content = input.content;
    this.buffer.push(entry);
    this.trim();
    return entry;
  }

  /**
   * Entries matching the query, oldest first, newest last - what a log view
   * wants. Filtering is exact on taskId/projectDir, so a task-scoped view
   * never shows another task's work.
   */
  list(query: ActivityQuery = {}): ActivityEntry[] {
    const scope = query.scope ?? (query.taskId ? 'task' : 'all');
    const minLevel = LEVEL_ORDER[query.minLevel ?? 'debug'];
    const filtered = this.buffer.filter((entry) => {
      if (LEVEL_ORDER[entry.level] < minLevel) return false;
      if (scope === 'task' && query.taskId) return entry.taskId === query.taskId;
      if (scope === 'project' && query.projectDir) return (entry.projectDir ?? '') === query.projectDir;
      return true;
    });
    const limit = query.limit && query.limit > 0 ? Math.round(query.limit) : 0;
    return limit && filtered.length > limit ? filtered.slice(filtered.length - limit) : filtered;
  }

  /** The newest entry, regardless of level. */
  last(): ActivityEntry | null {
    return this.buffer.length ? this.buffer[this.buffer.length - 1] : null;
  }

  size(): number {
    return this.buffer.length;
  }

  clear(): void {
    this.buffer = [];
  }

  private trim(): void {
    if (this.buffer.length > this.limit) {
      this.buffer.splice(0, this.buffer.length - this.limit);
    }
  }
}

/**
 * Strip credentials from a URL before it is shown to anyone. A base_url may
 * legitimately carry them (http://user:pass@host/v1), and the activity viewer
 * is a UI surface, not a log file the user greps themselves.
 */
export function safeEndpoint(url: string): string {
  if (!url) return '';
  try {
    const parsed = new URL(url);
    if (parsed.username || parsed.password) {
      parsed.username = '***';
      parsed.password = '***';
    }
    return parsed.toString();
  } catch {
    return url.replace(/\/\/[^/@\s]+@/, '//***@');
  }
}

/** Process-wide feed, shared by index.ts and the logger. */
export const activity = new ActivityLog();
