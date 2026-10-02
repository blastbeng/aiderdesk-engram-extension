/**
 * AiderDesk Memory has exactly one writable text field per entry:
 *
 *   MemoryEntry { id, content, type, taskId?, projectId?, timestamp }
 *   MemoryEntryType = string enum { 'task' | 'user-preference' | 'code-pattern' }
 *   (declared in the .d.ts, not exported - see NativeMemoryType below)
 *
 * There is no metadata column, no importance, no category, no scope.
 * (Verified at v0.81.0: packages/common/src/types/common.ts and
 *  src/main/memory/memory-manager.ts.)
 *
 * So the extension's richer record is encoded as a compact footer inside
 * `content`, with the human-readable statement FIRST so it dominates the
 * embedding vector:
 *
 *   Project uses llama.cpp, not Ollama, as the inference backend.
 *   [mem cat=decision imp=4 scope=project conf=0.92 ts=1730000000000]
 *
 * The footer is stripped before a memory is shown to the main model.
 */
import type { MemoryContext } from '@aiderdesk/extensions';
import type { MemoryCategory, MemoryScope } from './config';

/**
 * `MemoryEntryType` is declared in @aiderdesk/extensions' index.d.ts as a
 * string enum, but it is NOT exported - neither as a type nor as a runtime
 * value (the package's dist/index.js exports only AutonomyMode,
 * ContextMemoryMode, InvocationMode, OS, ToolApprovalState). Verified against
 * @aiderdesk/extensions 0.32.0 shipped with AiderDesk 0.81.0.
 *
 * The only way to name that type is to derive it from the method signature,
 * and the only values you can pass are the enum's string values, which is
 * exactly what the built-in memory MCP tool passes (its zod schema is
 * z.enum(['task','user-preference','code-pattern'])). The cast below is
 * therefore required, not decorative.
 */
export type NativeMemoryType = Parameters<MemoryContext['storeMemory']>[2];

const NATIVE_TYPE = {
  task: 'task',
  userPreference: 'user-preference',
  codePattern: 'code-pattern',
} as const;

function nativeType(value: (typeof NATIVE_TYPE)[keyof typeof NATIVE_TYPE]): NativeMemoryType {
  return value as unknown as NativeMemoryType;
}

export interface MemoryMeta {
  category: MemoryCategory;
  importance: number;
  scope: MemoryScope;
  confidence: number;
  /** Epoch ms when the fact was first stored. */
  createdAt: number;
  /** Epoch ms of the last update, when it has been updated. */
  updatedAt?: number;
}

export interface EncodedMemory {
  statement: string;
  meta: MemoryMeta;
}

const FOOTER_RE = /\n?\[\s*mem\s+([^\]]*?)\s*\]\s*$/;

/**
 * Map the extension's categories onto the three native AiderDesk memory
 * types, so native retrieval/native tooling classifies them sensibly.
 */
export function toEntryType(category: MemoryCategory): NativeMemoryType {
  switch (category) {
    case 'preference':
      return nativeType(NATIVE_TYPE.userPreference);
    case 'problem':
    case 'solution':
    case 'todo':
      return nativeType(NATIVE_TYPE.task);
    default:
      return nativeType(NATIVE_TYPE.codePattern);
  }
}

export function encodeMemory(statement: string, meta: MemoryMeta): string {
  const clean = statement.replace(/\s+/g, ' ').trim();
  const parts = [
    `cat=${meta.category}`,
    `imp=${meta.importance}`,
    `scope=${meta.scope}`,
    `conf=${meta.confidence.toFixed(2)}`,
    `ts=${meta.createdAt}`,
  ];
  if (meta.updatedAt) parts.push(`ut=${meta.updatedAt}`);
  return `${clean}\n[mem ${parts.join(' ')}]`;
}

/**
 * Decode a memory written by this extension.
 * Returns null for memories this extension does not manage (native AiderDesk
 * memories created by the built-in memory tools), so consolidation and
 * conflict resolution never touch them.
 */
export function decodeMemory(content: string): EncodedMemory | null {
  if (typeof content !== 'string') return null;
  const match = content.match(FOOTER_RE);
  if (!match) return null;

  const meta = parseFooter(match[1]);
  if (!meta) return null;

  const statement = content.slice(0, match.index).trim();
  if (!statement) return null;

  return { statement, meta };
}

export function isManagedMemory(content: string): boolean {
  return decodeMemory(content) !== null;
}

function parseFooter(raw: string): MemoryMeta | null {
  const kv = new Map<string, string>();
  for (const token of raw.split(/\s+/)) {
    const eq = token.indexOf('=');
    if (eq <= 0) continue;
    kv.set(token.slice(0, eq), token.slice(eq + 1));
  }

  const category = (kv.get('cat') ?? '') as MemoryCategory;
  const importance = Number(kv.get('imp'));
  const scope = (kv.get('scope') ?? '') as MemoryScope;
  const confidence = Number(kv.get('conf'));
  const createdAt = Number(kv.get('ts'));

  const knownCategories: MemoryCategory[] = [
    'project',
    'preference',
    'configuration',
    'decision',
    'constraint',
    'problem',
    'solution',
    'todo',
    'environment',
    'convention',
    'api',
    'other',
  ];

  if (!knownCategories.includes(category)) return null;
  if (!Number.isFinite(importance) || importance < 1 || importance > 5) return null;
  if (scope !== 'global' && scope !== 'project') return null;
  if (!Number.isFinite(createdAt) || createdAt <= 0) return null;

  const updatedAtRaw = kv.get('ut');
  const updatedAt = updatedAtRaw ? Number(updatedAtRaw) : undefined;

  return {
    category,
    importance,
    scope,
    confidence: Number.isFinite(confidence) ? Math.min(1, Math.max(0, confidence)) : 0.7,
    createdAt,
    updatedAt: Number.isFinite(updatedAt as number) ? updatedAt : undefined,
  };
}

/**
 * Strip the footer for display / embedding comparison.
 */
export function stripFooter(content: string): string {
  const decoded = decodeMemory(content);
  return decoded ? decoded.statement : content.trim();
}

/**
 * Compact line used when feeding existing memories to the secondary LLM.
 */
export function memoryForPrompt(id: string, content: string): string {
  const decoded = decodeMemory(content);
  if (!decoded) return `#${id}: ${content.replace(/\s+/g, ' ').trim().slice(0, 300)}`;
  const { statement, meta } = decoded;
  return `#${id} [imp=${meta.importance} cat=${meta.category} scope=${meta.scope}]: ${statement}`;
}

/**
 * Clean block injected into the main model's context. No footer noise.
 */
export function formatRetrievedBlock(memories: { content: string; projectId?: string }[]): string {
  const lines = memories.map((m) => {
    const decoded = decodeMemory(m.content);
    const statement = decoded ? decoded.statement : stripFooter(m.content);
    const scope = decoded ? decoded.meta.scope : m.projectId ? 'project' : 'global';
    const imp = decoded ? decoded.meta.importance : 3;
    return `- (${scope}, i${imp}) ${statement}`;
  });
  return lines.join('\n');
}
