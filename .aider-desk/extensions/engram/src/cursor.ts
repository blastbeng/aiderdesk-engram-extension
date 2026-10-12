/**
 * Extraction cursor: the gate that keeps the secondary LLM off content it has
 * already read.
 *
 * The problem this solves. `onAgentFinished` fires after EVERY agent run of a
 * task, and each round used to hand `runExtraction` the tail of the WHOLE
 * conversation. A task with 8 prompts therefore ran 8 rounds over 8 heavily
 * overlapping copies of the same transcript: 24 rounds cost 170 secondary-LLM
 * calls in production, 42 of the results were duplicates. That is the "it
 * loops and burns tokens" behaviour, and it is pure waste - the model is
 * re-reading facts it already extracted.
 *
 * The cursor records, per task, the last message a successful round consumed.
 * The next round gets only the messages AFTER it (plus that one boundary
 * message, so a fact split across the boundary is not lost), and a round with
 * nothing new to read is skipped before any HTTP request is made.
 *
 * Why message ids and not indexes. ContextMessage.id is stable for the life of
 * a task; indexes are not. AiderDesk rewrites the array under us in three
 * ordinary situations (verified against the host source at 0.87.0):
 *   - compact-conversation replaces the history with a summary,
 *   - "redo user prompt" truncates messages from the end,
 *   - restoring a task after an AiderDesk restart re-loads it from disk.
 * An index cursor silently extracts the wrong window in all three; an id cursor
 * either resolves exactly or reports `resynced`, and a resynced round falls back
 * to the old full-tail behaviour - which is the safe direction (a duplicate is
 * recoverable, a skipped fact is not).
 *
 * The content hash is the second lock. Two rounds whose new messages are
 * byte-identical (a prompt that produced no assistant text, a retry, a queued
 * prompt replayed) describe the same input, so the second one is skipped
 * outright. Zero LLM calls, zero tokens.
 *
 * In-memory on purpose: a cursor is a position in a live conversation, not
 * durable knowledge. After a restart the first round resyncs once and the
 * cursor rebuilds itself; persisting it would only make a stale position
 * survive into a context we can no longer map onto.
 */
import { createHash } from 'node:crypto';

import type { ContextMessage } from '@aiderdesk/extensions';

export interface CursorPosition {
  /** Id of the newest message a successful round consumed. */
  lastId: string;
  /** Hash of the exact message set that round read, for the no-new-content lock. */
  hash: string;
}

export interface FreshSlice {
  /** Messages this round should read (empty = nothing new). */
  fresh: ContextMessage[];
  /** True when the cursor could not be resolved, so `fresh` is the whole conversation. */
  resynced: boolean;
}

function textOf(message: ContextMessage): string {
  const content = message.content;
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let out = '';
  for (const part of content) {
    if (typeof part !== 'object' || part === null) continue;
    const p = part as { type?: string; text?: string };
    if (p.type === 'text' && typeof p.text === 'string') out += p.text;
  }
  return out;
}

/**
 * Messages that can carry a durable fact: user and assistant text. Tool
 * messages are reduced to tool names by the transcript builder, so they never
 * count as new material worth a round.
 */
export function countInformative(messages: readonly ContextMessage[]): number {
  let n = 0;
  for (const m of messages) {
    if (m.role !== 'user' && m.role !== 'assistant') continue;
    if (textOf(m).trim()) n += 1;
  }
  return n;
}

/**
 * Stable fingerprint of a message set. Role and text only: ids and timestamps
 * change when the host rewrites the array, the content we care about does not.
 */
export function hashMessages(messages: readonly ContextMessage[]): string {
  const digest = createHash('sha256');
  for (const m of messages) digest.update(`${m.role}\u0000${textOf(m)}\n`);
  return digest.digest('hex');
}

/**
 * Per-task cursor table. Keyed by task id; entries are dropped when the task
 * closes, so a long-lived desktop session does not accumulate positions for
 * tasks that no longer exist.
 */
export class ExtractionCursors {
  private readonly positions = new Map<string, CursorPosition>();

  /**
   * The messages this round has not seen yet.
   *
   * The cursor message itself is included: it is the boundary, and a fact
   * stated across two rounds must not fall between them. The duplicate that
   * can produce is caught by the extraction pipeline's own dedup.
   */
  sinceLast(taskKey: string, messages: readonly ContextMessage[]): FreshSlice {
    const position = this.positions.get(taskKey);
    if (!position) return { fresh: [...messages], resynced: false };

    const index = messages.findIndex((m) => m.id === position.lastId);
    if (index < 0) return { fresh: [...messages], resynced: true };
    return { fresh: messages.slice(index), resynced: false };
  }

  lastHash(taskKey: string): string | undefined {
    return this.positions.get(taskKey)?.hash;
  }

  /** Record what a SUCCESSFUL round consumed. A failed round must not call this. */
  remember(taskKey: string, messages: readonly ContextMessage[]): void {
    const last = messages[messages.length - 1];
    if (!last?.id) return;
    this.positions.set(taskKey, { lastId: last.id, hash: hashMessages(messages) });
  }

  forget(taskKey: string): void {
    this.positions.delete(taskKey);
  }

  size(): number {
    return this.positions.size;
  }

  clear(): void {
    this.positions.clear();
  }
}
