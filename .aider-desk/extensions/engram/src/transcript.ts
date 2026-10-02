/**
 * Transcript compaction for the secondary LLM.
 *
 * The secondary LLM must never receive the whole conversation. This module
 * builds a bounded, information-dense transcript:
 *   - user messages: text, truncated
 *   - assistant messages: text parts only (reasoning + tool-call parts dropped)
 *   - tool messages: reduced to the tool NAMES only (their outputs are the
 *     bulk of a session's tokens and almost never contain durable facts)
 *
 * Token counting: AiderDesk exposes no token-counting API to extensions
 * (verified at v0.81.0 - ExtensionContext has no tokenizer; only per-message
 * `usageReport` after the fact). So we estimate with chars/4, which is the
 * standard conservative heuristic for BPE tokenizers on English/code prose.
 */
import type { ContextMessage, TextPart } from '@aiderdesk/extensions';

/** Rough token estimate. AiderDesk exposes no tokenizer to extensions. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function textFromParts(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const out: string[] = [];
  for (const part of parts) {
    if (typeof part !== 'object' || part === null) continue;
    const p = part as { type?: string; text?: string };
    if (p.type === 'text' && typeof p.text === 'string') out.push(p.text);
  }
  return out.join('\n');
}

function toolNames(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const names = new Set<string>();
  for (const part of parts) {
    if (typeof part !== 'object' || part === null) continue;
    const p = part as { toolName?: string };
    if (typeof p.toolName === 'string' && p.toolName) names.add(p.toolName);
  }
  return Array.from(names).join(', ');
}

function truncate(text: string, max: number): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (clean.length <= max) return clean;
  return `${clean.slice(0, max - 1)}…`;
}

/**
 * Render one message to a compact line. Returns '' for messages that carry
 * no extractable information.
 */
function renderMessage(message: ContextMessage, limits: { user: number; assistant: number }): string {
  const role = message.role;

  if (role === 'user') {
    const content = message.content;
    const text = typeof content === 'string' ? content : textFromParts(content);
    if (!text.trim()) return '';
    return `USER: ${truncate(text, limits.user)}`;
  }

  if (role === 'assistant') {
    const content = message.content;
    const text = typeof content === 'string' ? content : textFromParts(content);
    if (!text.trim()) return '';
    const files = Array.isArray(message.editedFiles) && message.editedFiles.length
      ? ` [edited: ${message.editedFiles.slice(0, 8).join(', ')}]`
      : '';
    return `ASSISTANT: ${truncate(text, limits.assistant)}${files}`;
  }

  if (role === 'tool') {
    const names = toolNames(message.content);
    if (!names) return '';
    return `TOOLS: ${truncate(names, 160)}`;
  }

  return '';
}

export interface TranscriptOptions {
  maxMessages: number;
  maxTokens: number;
  /** Characters kept per user message. */
  userChars?: number;
  /** Characters kept per assistant message. */
  assistantChars?: number;
}

/**
 * Build the transcript from the TAIL of the conversation (most recent
 * messages first), stopping at the token budget.
 */
export function buildTranscript(messages: ContextMessage[], options: TranscriptOptions): {
  transcript: string;
  usedMessages: number;
  estimatedTokens: number;
} {
  const limits = {
    user: options.userChars ?? 1400,
    assistant: options.assistantChars ?? 1400,
  };

  const tail = messages.slice(-Math.max(1, options.maxMessages));
  const lines: string[] = [];
  let tokens = 0;
  let used = 0;

  // Walk backwards so the most recent turns are always kept.
  for (let i = tail.length - 1; i >= 0; i--) {
    const line = renderMessage(tail[i], limits);
    if (!line) continue;
    const cost = estimateTokens(line) + 2;
    if (tokens + cost > options.maxTokens) break;
    tokens += cost;
    lines.unshift(line);
    used++;
  }

  return {
    transcript: lines.join('\n\n'),
    usedMessages: used,
    estimatedTokens: tokens,
  };
}

/**
 * Drop messages this extension injected, so they never feed back into
 * extraction (self-referential memory loops).
 */
export function isExtensionInjected(message: ContextMessage): boolean {
  if (message.role !== 'user') return false;
  const content = typeof message.content === 'string' ? message.content : textFromParts(message.content);
  return content.includes('<engram-memory-context>');
}
