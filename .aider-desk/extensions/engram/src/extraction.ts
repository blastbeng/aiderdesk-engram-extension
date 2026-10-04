/**
 * Extraction pipeline.
 *
 * transcript (redacted, bounded)
 *   -> secondary LLM: extract candidate facts (JSON)
 *   -> validate (repair round-trip on failure)
 *   -> secondary LLM: classify each candidate against its nearest existing
 *      memories as NEW / DUPLICATE / UPDATE / CONFLICT / OBSOLETE
 *   -> apply through the native AiderDesk Memory API
 *
 * Never throws. Every failure is reported in the returned object.
 */
import type { ExtensionContext, ContextMessage, MemoryContext, MemoryEntry, TaskContext } from '@aiderdesk/extensions';
import type { EngramConfig, MemoryCategory, MemoryScope } from './config';
import { budget, chatWithTransport } from './llm';
import { aliasFor, buildAliasTable, resolveAlias, type AliasTable } from './aliases';
import {
  ClassificationSchema,
  ExtractionSchema,
  parseStructured,
  validate,
  type ClassificationResult,
  type MemoryCandidate,
} from './json';
import {
  CLASSIFICATION_SYSTEM,
  EXTRACTION_SYSTEM,
  REPAIR_SYSTEM,
  buildClassificationUser,
  buildExtractionUser,
  buildRepairUser,
} from './prompts';
import { buildTranscript, estimateTokens, isExtensionInjected } from './transcript';
import { memoryForPromptAliased, normalizeStatement } from './memory-format';
import { looksSecret, redactSecrets } from './privacy';
import {
  getMemoryContextSafely,
  importanceOf,
  metaForNew,
  remove,
  retrieveScoped,
  scopeOf,
  statementOf,
  statementsForPrompt,
  storeNew,
  updateExisting,
} from './store';
import { logger } from './logger';
import { projectStats, saveState, type EngramState } from './state';

/**
 * Time-bound phrasing that can never be durable: a memory that is only true
 * "today"/"tonight"/"right now" is noise next week. The extraction prompt
 * already forbids it; this is the deterministic backstop for when the model
 * writes one anyway. Genuinely time-dependent facts survive by phrasing the
 * time context explicitly ("As of 2026-10, ..."), which this regex ignores.
 */
const TRANSIENT_PHRASE_RE =
  /\b(today|yesterday|tonight|this morning|this afternoon|this evening|right now|just now|at the moment|this session|earlier today)\b/i;

export interface ExtractionReport {
  candidates: number;
  stored: number;
  updated: number;
  duplicates: number;
  obsolete: number;
  skipped: number;
  failure?: string;
}

export interface ExtractionOptions {
  context: ExtensionContext;
  messages: ContextMessage[];
  projectDir: string;
  taskId: string;
  config: EngramConfig;
  state: EngramState;
  statePath: string;
  signal?: AbortSignal;
  /**
   * Native AiderDesk task context, required only by the 'aiderdesk' transport
   * (TaskContext.generateText). Optional for the default HTTP transport.
   */
  taskContext?: TaskContext | null;
}

const SYSTEM_OVERHEAD_TOKENS = 900;

function lastUserQuery(messages: ContextMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== 'user') continue;
    if (isExtensionInjected(m)) continue;
    const text = typeof m.content === 'string' ? m.content : '';
    if (text.trim()) return text.replace(/\s+/g, ' ').trim().slice(0, 600);
  }
  return '';
}

export async function runExtraction(options: ExtractionOptions): Promise<ExtractionReport> {
  const report: ExtractionReport = {
    candidates: 0,
    stored: 0,
    updated: 0,
    duplicates: 0,
    obsolete: 0,
    skipped: 0,
  };

  const { context, config, state, statePath, projectDir, taskId } = options;

  const memory = getMemoryContextSafely(context);
  if (!memory) {
    logger.debug('memory store unavailable or disabled - extraction skipped');
    return report;
  }

  const messages = options.messages.filter((m) => !isExtensionInjected(m));
  if (!messages.length) return report;

  // --- 1. Nearest existing memories, used both as a "do not restate" list
  //        and as the dedup corpus. Bounded so they cannot crowd out the
  //        transcript from the token budget.
  const query = lastUserQuery(messages) || projectDir || 'project';
  const existing = await retrieveScoped(
    memory,
    projectDir,
    query,
    config.extraction.max_existing_for_dedup,
    true,
  );
  // Id-free statements: this corpus exists only so the model avoids
  // restating known facts; nothing ever references it by id, and short
  // handles here would only be noise.
  const existingLines = statementsForPrompt(existing);
  const existingTokens = estimateTokens(existingLines.join('\n'));

  // --- 2. Bounded, redacted transcript.
  const transcriptBudget = Math.max(
    600,
    config.extraction.max_input_tokens - existingTokens - SYSTEM_OVERHEAD_TOKENS,
  );
  const built = buildTranscript(messages, {
    maxMessages: config.extraction.max_messages,
    maxTokens: transcriptBudget,
  });
  if (!built.transcript.trim()) return report;

  const redaction = redactSecrets(built.transcript, config.privacy.redact_secrets);
  if (redaction.hits > 0) {
    logger.info(`redacted ${redaction.hits} secret(s): ${redaction.categories.join(', ')}`);
  }

  // --- 3. Extraction call.
  logger.info(`extracting memories... (${built.usedMessages} messages, ~${built.estimatedTokens} tok)`);

  const first = await chatWithTransport(config.secondary_llm, {
    system: EXTRACTION_SYSTEM,
    user: buildExtractionUser({
      transcript: redaction.text,
      existingMemories: existingLines,
      projectDir: projectDir || '(no project)',
      minImportance: config.extraction.min_importance,
    }),
    // Bounded per-call budget: inheriting the configured max_tokens (32768 in
    // practice) let one extraction call run for minutes on a reasoning model.
    maxTokens: budget(config.secondary_llm, 'extraction'),
    signal: options.signal,
  }, options.taskContext);

  state.totals.llmCalls += 1;
  if (!first.ok) {
    // An external abort (extension unload) is not an LLM failure.
    if (first.kind !== 'aborted') state.totals.llmFailures += 1;
    report.failure = first.message;
    logger.warn(`extraction skipped: secondary LLM ${first.kind} - ${first.message}`);
    saveState(statePath, state);
    return report;
  }

  // --- 4. Validate, with one repair round-trip.
  let parsed = parseStructured(ExtractionSchema, first.text);
  if (!parsed.ok) {
    logger.warn(`model returned invalid JSON (${parsed.errors}) - requesting repair`);
    const repaired = await chatWithTransport(config.secondary_llm, {
      system: REPAIR_SYSTEM,
      user: buildRepairUser(first.text, parsed.errors ?? 'invalid'),
      temperature: 0,
      maxTokens: budget(config.secondary_llm, 'repair'),
      signal: options.signal,
    }, options.taskContext);
    state.totals.llmCalls += 1;
    if (!repaired.ok) {
      if (repaired.kind !== 'aborted') state.totals.llmFailures += 1;
      report.failure = `repair call failed: ${repaired.message}`;
      saveState(statePath, state);
      return report;
    }
    parsed = parseStructured(ExtractionSchema, repaired.text);
    if (!parsed.ok) {
      report.failure = `JSON still invalid after repair (${parsed.errors})`;
      logger.error('no memory written - refusing to store corrupt data');
      saveState(statePath, state);
      return report;
    }
  }

  // --- 5. Filter: importance floor, secret sweep, hard candidate cap, and a
  //        deterministic within-batch dedup (the model sometimes restates the
  //        same fact twice in one round; only the first copy is classified).
  const candidates: MemoryCandidate[] = [];
  const seenCandidates = new Set<string>();
  for (const candidate of parsed.data!.memories) {
    if (candidate.importance < config.extraction.min_importance) {
      report.skipped += 1;
      continue;
    }
    if (TRANSIENT_PHRASE_RE.test(candidate.content)) {
      report.skipped += 1;
      logger.debug(`dropped a time-bound candidate: ${candidate.content.slice(0, 80)}`);
      continue;
    }
    if (looksSecret(candidate.content, config.privacy.redact_secrets)) {
      report.skipped += 1;
      logger.warn('dropped a candidate that contained a secret');
      continue;
    }
    const key = normalizeStatement(candidate.content);
    if (key && seenCandidates.has(key)) {
      report.skipped += 1;
      continue;
    }
    if (key) seenCandidates.add(key);
    candidates.push(candidate);
  }
  report.candidates = candidates.length;

  if (!candidates.length) {
    logger.info('0 candidates found');
    bumpStats(state, statePath, projectDir, { extractions: 1 });
    return report;
  }
  logger.info(`${report.candidates} candidate(s) found`);

  // --- 6. Dedup / update / conflict classification against nearest existing memories.
  const existingForCandidate: { index: number; lines: string[] }[] = [];
  const corpora: MemoryEntry[][] = [];
  // One alias table per candidate corpus: verdict targetIds are resolved
  // through the same table the prompt was built with.
  const corpusTables: AliasTable[] = [];

  for (let i = 0; i < candidates.length; i++) {
    const corpus = await retrieveScoped(
      memory,
      projectDir,
      candidates[i].content,
      config.extraction.max_existing_for_dedup,
      true,
    );
    corpora.push(corpus);
    // Short handles (m1, m2, ...) instead of raw UUIDs: the secondary model
    // cannot echo UUIDs back reliably, which silently zeroed UPDATE/CONFLICT
    // verdicts in production (~25 of 27 rounds produced no update).
    const table = buildAliasTable(corpus);
    corpusTables.push(table);
    existingForCandidate.push({
      index: i,
      lines: corpus.map((e) => memoryForPromptAliased(aliasFor(table, e.id), e.content)),
    });
  }

  const classified = await classify(config, candidates, existingForCandidate, options.signal, state, statePath, options.taskContext);
  if (!classified) {
    // Classification unavailable. The old fallback (write everything as NEW)
    // polluted the store with duplicates every time the classifier was down -
    // observed live: 36 stored, 0 updated, in one deployment. The fallback is
    // now conservative:
    //   - verbatim restatements of a known memory count as DUPLICATE;
    //   - only candidates at least one importance step above the floor are
    //     written as NEW (worst case a duplicate, never a data loss);
    //   - the rest are dropped - a later successful round re-extracts them.
    logger.warn('dedup classification unavailable - using deterministic fallback');
    const gate = Math.min(5, config.extraction.min_importance + 1);
    for (let i = 0; i < candidates.length; i++) {
      const candidate = candidates[i];
      if (exactDuplicateOf(corpora[i] ?? [], candidate.content)) {
        report.duplicates += 1;
        state.totals.duplicatesSkipped += 1;
        continue;
      }
      if (candidate.importance < gate) {
        report.skipped += 1;
        continue;
      }
      report.stored += await writeNew(memory, projectDir, taskId, candidate);
    }
    finish(report, state, statePath, projectDir);
    return report;
  }

  const byIndex = new Map<number, ClassificationResult>();
  for (const result of classified) byIndex.set(result.index, result);

  // --- 7. Apply. Every NEW write first passes a deterministic exact-match
  //        guard against the candidate's own dedup corpus, so a NEW verdict
  //        that contradicts a verbatim existing memory cannot create a row.
  const writeNewChecked = async (candidate: MemoryCandidate, corpus: MemoryEntry[]): Promise<1 | 0> => {
    if (exactDuplicateOf(corpus, candidate.content)) {
      report.duplicates += 1;
      state.totals.duplicatesSkipped += 1;
      return 0;
    }
    return writeNew(memory, projectDir, taskId, candidate);
  };

  for (let i = 0; i < candidates.length; i++) {
    const candidate = candidates[i];
    const corpus = corpora[i];
    const byId = new Map(corpus.map((e) => [e.id, e]));
    const verdict = byIndex.get(i);

    if (!verdict) {
      report.stored += await writeNewChecked(candidate, corpus);
      continue;
    }

    // Resolve whatever handle the model wrote back to the real memory id;
    // an unresolvable handle degrades to NEW via the writeNewChecked path.
    const resolvedId =
      verdict.targetId && corpusTables[i]
        ? resolveAlias(verdict.targetId, corpusTables[i])
        : null;
    const target = resolvedId ? byId.get(resolvedId) : undefined;

    switch (verdict.verdict) {
      case 'DUPLICATE': {
        report.duplicates += 1;
        state.totals.duplicatesSkipped += 1;
        break;
      }

      case 'NEW': {
        report.stored += await writeNewChecked(candidate, corpus);
        break;
      }

      case 'UPDATE':
      case 'CONFLICT':
      case 'OBSOLETE': {
        if (!target) {
          report.stored += await writeNewChecked(candidate, corpus);
          break;
        }
        const statement = (verdict.mergedContent ?? '').trim() || candidate.content;
        const meta = metaForNew({
          category: candidate.category,
          importance: Math.max(candidate.importance, importanceOf(target)),
          scope: scopeOf(target),
          confidence: candidate.confidence,
        });
        const ok = await updateExisting(memory, target.id, statement, meta);
        if (ok) {
          report.updated += 1;
          state.totals.updated += 1;
          if (verdict.verdict === 'OBSOLETE') report.obsolete += 1;
        } else {
          report.stored += await writeNewChecked(candidate, corpus);
        }
        break;
      }
    }
  }

  finish(report, state, statePath, projectDir);
  return report;
}

async function classify(
  config: EngramConfig,
  candidates: MemoryCandidate[],
  existingForCandidate: { index: number; lines: string[] }[],
  signal: AbortSignal | undefined,
  state: EngramState,
  statePath: string,
  taskContext?: TaskContext | null,
): Promise<ClassificationResult[] | null> {
  const result = await chatWithTransport(config.secondary_llm, {
    system: CLASSIFICATION_SYSTEM,
    user: buildClassificationUser({ candidates, existingForCandidate }),
    temperature: Math.min(config.secondary_llm.temperature, 0.2),
    maxTokens: budget(config.secondary_llm, 'classification'),
    signal,
  }, taskContext);

  state.totals.llmCalls += 1;
  if (!result.ok) {
    if (result.kind !== 'aborted') state.totals.llmFailures += 1;
    return null;
  }

  const parsed = parseStructured(ClassificationSchema, result.text);
  if (!parsed.ok || !parsed.data) return null;

  return parsed.data.results;
}

/**
 * Deterministic duplicate guard: the candidate statement, normalized, already
 * exists verbatim in the corpus. Cheaper and more reliable than the LLM
 * classifier for this narrow case; catches the exact restatements that
 * otherwise became permanent duplicates.
 */
function exactDuplicateOf(corpus: MemoryEntry[], content: string): MemoryEntry | null {
  const key = normalizeStatement(content);
  if (!key) return null;
  for (const entry of corpus) {
    if (normalizeStatement(statementOf(entry)) === key) return entry;
  }
  return null;
}

async function writeNew(
  memory: MemoryContext,
  projectDir: string,
  taskId: string,
  candidate: MemoryCandidate,
): Promise<1 | 0> {
  const id = await storeNew(
    memory,
    projectDir,
    taskId,
    candidate.content,
    metaForNew({
      category: candidate.category,
      importance: candidate.importance,
      scope: candidate.scope,
      confidence: candidate.confidence,
    }),
  );
  return id ? 1 : 0;
}

/**
 * Merge a stats patch additively: numeric fields INCREMENT the counter.
 * `Object.assign` SET the counter, so `state.json` showed `extractions: 1`
 * forever no matter how many extraction rounds ran.
 */
function bumpStats(
  state: EngramState,
  statePath: string,
  projectDir: string,
  patch: Partial<ReturnType<typeof projectStats>>,
): void {
  const stats = projectStats(state, projectDir);
  const record = stats as unknown as Record<string, unknown>;
  for (const [key, value] of Object.entries(patch)) {
    const current = record[key];
    record[key] = typeof value === 'number' && typeof current === 'number' ? current + value : value;
  }
  saveState(statePath, state);
}

function finish(report: ExtractionReport, state: EngramState, statePath: string, projectDir: string): void {
  const stats = projectStats(state, projectDir);
  stats.extractions += 1;
  stats.stored += report.stored;
  stats.updated += report.updated;
  stats.duplicates += report.duplicates;
  stats.obsolete += report.obsolete;
  stats.lastExtractionAt = Date.now();

  state.totals.stored += report.stored;
  state.totals.updated += report.updated;
  saveState(statePath, state);

  logger.info(
    `${report.candidates} candidates: ${report.stored} new, ${report.updated} update, ` +
      `${report.duplicates} duplicate, ${report.obsolete} obsolete, ${report.skipped} skipped ` +
      `-> stored ${report.stored} memory`,
  );
}
