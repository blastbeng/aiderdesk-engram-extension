/**
 * Structured-output handling: tolerant JSON recovery + Zod validation.
 *
 * `zod` is always resolvable inside an AiderDesk extension: the extension
 * loader (src/main/extensions/extension-loader.ts @ v0.81.0) registers the
 * alias `zod -> require.resolve('zod')`, so `import { z } from 'zod'` works
 * with zero dependencies installed by us.
 */
import { z } from 'zod';
import type { MemoryCategory, MemoryScope } from './config';

export const CATEGORIES: MemoryCategory[] = [
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

export const SCOPES: MemoryScope[] = ['global', 'project'];

export const MemoryCandidateSchema = z.object({
  content: z.string().trim().min(10).max(1400),
  category: z.enum(CATEGORIES as [MemoryCategory, ...MemoryCategory[]]),
  importance: z.coerce.number().int().min(1).max(5).catch(3),
  scope: z.enum(SCOPES as [MemoryScope, ...MemoryScope[]]).catch('project'),
  confidence: z.coerce.number().min(0).max(1).catch(0.7),
});

export const ExtractionSchema = z.object({
  memories: z.array(MemoryCandidateSchema).max(40).default([]),
});

export type MemoryCandidate = z.infer<typeof MemoryCandidateSchema>;

export const Verdict = ['NEW', 'DUPLICATE', 'UPDATE', 'CONFLICT', 'OBSOLETE'] as const;
export type VerdictKind = (typeof Verdict)[number];

export const ClassificationSchema = z.object({
  results: z
    .array(
      z.object({
        index: z.coerce.number().int().min(0),
        verdict: z.enum(Verdict),
        targetId: z.string().trim().nullable().optional(),
        mergedContent: z.string().trim().max(1400).nullable().optional(),
        note: z.string().trim().max(240).optional(),
      }),
    )
    .max(60)
    .default([]),
});

export type ClassificationResult = z.infer<typeof ClassificationSchema>['results'][number];

export const ConsolidationSchema = z.object({
  actions: z
    .array(
      z.object({
        action: z.enum(['KEEP', 'MERGE', 'UPDATE', 'DELETE']),
        targetIds: z.array(z.string().trim()).min(1).max(40),
        content: z.string().trim().max(1400).nullable().optional(),
        importance: z.coerce.number().int().min(1).max(5).nullable().optional(),
        category: z.enum(CATEGORIES as [MemoryCategory, ...MemoryCategory[]]).nullable().optional(),
        scope: z.enum(SCOPES as [MemoryScope, ...MemoryScope[]]).nullable().optional(),
        reason: z.string().trim().max(240).optional(),
      }),
    )
    .max(80)
    .default([]),
});

export type ConsolidationAction = z.infer<typeof ConsolidationSchema>['actions'][number];

export interface ParseOutcome<T> {
  ok: boolean;
  data?: T;
  /** Human-readable, log-safe validation summary. */
  errors?: string;
}

/**
 * Recover a JSON object from LLM output.
 *
 * Handles the real-world mess: markdown fences, leading prose, trailing
 * commentary, ```json fences, trailing commas, single-quoted keys.
 * Returns null when nothing parseable exists.
 */
export function recoverJson(raw: string): unknown | null {
  if (!raw || !raw.trim()) return null;

  const candidates: string[] = [];
  const text = raw.trim();

  candidates.push(text);

  // Strip markdown fences.
  const fenced = text.match(/```(?:json|JSON)?\s*([\s\S]*?)```/);
  if (fenced?.[1]) candidates.push(fenced[1].trim());

  // First balanced {...} block.
  const balanced = extractBalanced(text, '{', '}');
  if (balanced) candidates.push(balanced);

  for (const candidate of candidates) {
    const parsed = tryParseJson(candidate);
    if (parsed !== null) return parsed;
  }
  return null;
}

function tryParseJson(s: string): unknown | null {
  try {
    return JSON.parse(s);
  } catch {
    /* continue with repairs */
  }

  // Repair 1: strip trailing commas before } or ]
  const noTrailingCommas = s.replace(/,(\s*[}\]])/g, '$1');
  try {
    return JSON.parse(noTrailingCommas);
  } catch {
    /* continue */
  }

  // Repair 2: quote bare object keys, single-quote strings.
  const repaired = noTrailingCommas
    .replace(/([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)/g, '$1"$2"$3')
    .replace(/'/g, '"');
  try {
    return JSON.parse(repaired);
  } catch {
    return null;
  }
}

function extractBalanced(text: string, open: string, close: string): string | null {
  const start = text.indexOf(open);
  if (start < 0) return null;

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === open) depth++;
    else if (ch === close) {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

/**
 * Validate LLM output against a schema. Returns a log-safe error summary.
 */
export function validate<T>(schema: z.ZodType<T>, payload: unknown): ParseOutcome<T> {
  const parsed = schema.safeParse(payload);
  if (parsed.success) return { ok: true, data: parsed.data };

  const errors = parsed.error.issues
    .slice(0, 6)
    .map((issue) => `${issue.path.join('.') || '(root)'}: ${issue.message}`)
    .join('; ');
  return { ok: false, errors: errors || 'schema validation failed' };
}

/**
 * Full pipeline: recover JSON, then validate.
 */
export function parseStructured<T>(schema: z.ZodType<T>, raw: string): ParseOutcome<T> {
  const json = recoverJson(raw);
  if (json === null) return { ok: false, errors: 'no parseable JSON object in model output' };
  return validate(schema, json);
}
