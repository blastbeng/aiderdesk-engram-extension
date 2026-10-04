/**
 * Consolidation pipeline: 100 memories -> secondary LLM -> dedupe / merge /
 * remove obsolete / resolve conflicts -> a small set of high-quality memories.
 *
 * Runs on a task-count interval (configurable) or on demand via the
 * `memory:consolidate` command. Never runs per message.
 */
import type { ExtensionContext, MemoryEntry, TaskContext } from '@aiderdesk/extensions';
import type { EngramConfig } from './config';
import { budget, chatWithTransport } from './llm';
import { ConsolidationSchema, parseStructured, type ConsolidationAction } from './json';
import { CONSOLIDATION_SYSTEM, buildConsolidationUser } from './prompts';
import { estimateTokens } from './transcript';
import {
  getMemoryContextSafely,
  isManaged,
  listAll,
  metaForUpdate,
  remove,
  updateExisting,
} from './store';
import { decodeMemory, memoryForPromptAliased } from './memory-format';
import { looksSecret } from './privacy';
import { aliasFor, buildAliasTable, resolveAlias, type AliasTable } from './aliases';
import { logger } from './logger';
import { projectStats, saveState, type EngramState } from './state';

export interface ConsolidationReport {
  scanned: number;
  merged: number;
  updated: number;
  deleted: number;
  kept: number;
  skipped: number;
  failure?: string;
}

export interface ConsolidationOptions {
  context: ExtensionContext;
  projectDir: string;
  config: EngramConfig;
  state: EngramState;
  statePath: string;
  signal?: AbortSignal;
  /** Native AiderDesk task context; used only by the 'aiderdesk' transport. */
  taskContext?: TaskContext | null;
}

/** Per-call budget for the memory list. */
const CONSOLIDATION_TOKEN_BUDGET = 9000;
const MAX_BATCHES = 6;

export async function runConsolidation(options: ConsolidationOptions): Promise<ConsolidationReport> {
  const report: ConsolidationReport = {
    scanned: 0,
    merged: 0,
    updated: 0,
    deleted: 0,
    kept: 0,
    skipped: 0,
  };

  const { context, config, state, statePath, projectDir } = options;

  const memory = getMemoryContextSafely(context);
  if (!memory) {
    report.failure = 'memory store unavailable or disabled';
    return report;
  }

  const all = await listAll(memory);
  // Only touch memories this extension wrote. Native AiderDesk memories
  // (created by the built-in memory tools) are never consolidated away.
  const managed = all.filter((entry) => entry?.id && isManaged(entry));

  const projectMemories = managed.filter((entry) => (entry.projectId ?? '') === projectDir);
  const globalMemories = managed.filter((entry) => (entry.projectId ?? '') === '');
  const targets = [...projectMemories, ...globalMemories];

  report.scanned = targets.length;
  if (targets.length < 2) {
    logger.info(`consolidation: ${targets.length} managed memory - nothing to do`);
    // The counter must be reset here too. Without it a project with nothing
    // to consolidate stays above the interval forever, so every single
    // extraction round re-triggers a consolidation that can only answer
    // "nothing to do" - a permanent no-op loop (and a stats line that always
    // reads "N/N rounds").
    const stats = projectStats(state, projectDir);
    stats.tasksSinceConsolidation = 0;
    stats.lastConsolidationAt = Date.now();
    saveState(statePath, state);
    return report;
  }

  const batches = chunkByTokens(targets, CONSOLIDATION_TOKEN_BUDGET);
  const windowSize = Math.min(batches.length, MAX_BATCHES);
  // True when the store is too large for one run: batches beyond the cap are
  // never sent to the LLM this round.
  const truncated = batches.length > MAX_BATCHES;
  logger.info(`consolidating ${targets.length} memories in ${windowSize} batch(es)...`);

  for (let i = 0; i < windowSize; i++) {
    const batch = batches[i];
    const byId = new Map(batch.map((entry) => [entry.id, entry]));
    // Short positional aliases (m1, m2, ...) instead of real UUIDs: models
    // cannot reliably echo back long ids, which silently zeroed every
    // production consolidation run. Resolution back to real ids is tolerant
    // (see src/aliases.ts).
    const table = buildAliasTable(batch);
    const scopeLabel = `${projectDir || '(no project)'} + global (batch ${i + 1}/${windowSize})`;

    const result = await chatWithTransport(config.secondary_llm, {
      system: CONSOLIDATION_SYSTEM,
      user: buildConsolidationUser(
        batch.map((e) => memoryForPromptAliased(aliasFor(table, e.id), e.content)),
        scopeLabel,
      ),
      temperature: Math.min(config.secondary_llm.temperature, 0.2),
      // A background job over many memories legitimately runs for minutes on
      // a slow local GPU; the configured timeout is a floor, not a cap.
      maxTokens: budget(config.secondary_llm, 'consolidation'),
      timeoutMs: Math.max(config.secondary_llm.timeout_ms, 120_000),
      signal: options.signal,
    }, options.taskContext);

    state.totals.llmCalls += 1;
    if (!result.ok) {
      // An external abort (extension unload) is not an LLM failure.
      if (result.kind !== 'aborted') state.totals.llmFailures += 1;
      report.failure = `batch ${i + 1}: ${result.kind} - ${result.message}`;
      logger.warn(`consolidation stopped: ${report.failure}`);
      break;
    }

    const parsed = parseStructured(ConsolidationSchema, result.text);
    if (!parsed.ok || !parsed.data) {
      report.failure = `batch ${i + 1}: invalid consolidation JSON (${parsed.errors ?? 'no JSON recovered'})`;
      logger.warn('consolidation batch discarded - no memory written');
      continue;
    }

    await applyActions(parsed.data.actions, table, byId, memory, projectDir, report, config, state);
  }

  const stats = projectStats(state, projectDir);
  if (truncated) {
    // Do NOT reset the round counter on a truncated run: batches beyond the
    // cap were never seen, and resetting here would schedule the next run
    // past them forever - with oldest-first batching the newest memories
    // would never consolidate. Leaving the counter above the interval
    // re-triggers consolidation on the next trigger until the backlog clears
    // (earlier batches shrink as their duplicates merge away).
    logger.warn(
      `consolidation: ${batches.length - windowSize} batch(es) beyond the ${MAX_BATCHES}-batch cap ` +
        'were not processed this run; they are picked up on the next consolidation trigger',
    );
  } else {
    stats.tasksSinceConsolidation = 0;
    stats.lastConsolidationAt = Date.now();
  }
  // Project counters must reflect consolidation too: `memory:stats` reads
  // these per-project numbers, and they used to show only extraction work, so
  // a project that merged 20 memories and deleted 15 reported none of it.
  stats.updated += report.updated;
  stats.deleted += report.deleted;
  saveState(statePath, state);

  logger.info(
    `consolidation: scanned ${report.scanned} -> merged ${report.merged}, ` +
      `updated ${report.updated}, deleted ${report.deleted}, kept ${report.kept}, skipped ${report.skipped}`,
  );

  return report;
}

function chunkByTokens(entries: MemoryEntry[], budget: number): MemoryEntry[][] {
  const batches: MemoryEntry[][] = [];
  let current: MemoryEntry[] = [];
  let tokens = 0;

  // Oldest first: consolidation should collapse the accumulated cruft before
  // touching the freshest memories.
  const ordered = [...entries].sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0));

  for (const entry of ordered) {
    const cost = estimateTokens(entry.content) + 12;
    if (tokens + cost > budget && current.length) {
      batches.push(current);
      current = [];
      tokens = 0;
    }
    current.push(entry);
    tokens += cost;
  }
  if (current.length) batches.push(current);
  return batches;
}

async function applyActions(
  actions: ConsolidationAction[],
  table: AliasTable,
  byId: Map<string, MemoryEntry>,
  memory: ReturnType<typeof getMemoryContextSafely>,
  projectDir: string,
  report: ConsolidationReport,
  config: EngramConfig,
  state: EngramState,
): Promise<void> {
  if (!memory) return;
  const touched = new Set<string>();

  for (const action of actions) {
    // Resolve whatever the model wrote back to real ids; entries it referenced
    // twice in one batch are applied only once.
    const ids: string[] = [];
    for (const raw of action.targetIds) {
      const id = resolveAlias(raw, table);
      if (id && byId.has(id) && !touched.has(id) && !ids.includes(id)) ids.push(id);
    }
    if (!ids.length) {
      report.skipped += 1;
      continue;
    }
    for (const id of ids) touched.add(id);

    if (action.action === 'KEEP') {
      report.kept += ids.length;
      continue;
    }

    if (action.action === 'DELETE') {
      if (config.consolidation.safe_mode) {
        report.skipped += ids.length;
        continue;
      }
      for (const id of ids) {
        if (await remove(memory, id)) {
          report.deleted += 1;
          state.totals.deleted += 1;
        }
      }
      continue;
    }

    // MERGE / UPDATE: rewrite the first target, drop the rest.
    const primary = byId.get(ids[0])!;
    let statement = (action.content ?? '').trim() || statementOf(primary);
    // action.content is model output and was never secret-swept: a model that
    // echoes a credential into the merged statement must not write it. Fall
    // back to the primary's already-stored statement, which passed the sweep
    // when it was written.
    if (looksSecret(statement, config.privacy.redact_secrets)) statement = statementOf(primary);
    const decoded = decodeMemory(primary.content);
    // metaForUpdate: a merge rewrites an existing memory, so the primary's
    // original createdAt survives (metaForNew would stamp it as brand new).
    const meta = metaForUpdate(decoded?.meta, {
      category: action.category ?? decoded?.meta.category ?? 'other',
      importance: action.importance ?? decoded?.meta.importance ?? 3,
      scope: action.scope ?? decoded?.meta.scope ?? (projectDir ? 'project' : 'global'),
      confidence: decoded?.meta.confidence ?? 0.8,
    });

    const ok = await updateExisting(memory, primary.id, statement, meta);
    if (ok) {
      report.updated += 1;
      state.totals.updated += 1;
    }

    const extras = ids.slice(1);
    if (extras.length) {
      report.merged += 1;
      for (const id of extras) {
        if (config.consolidation.safe_mode) {
          // Safe mode: keep the redundant memory but demote it to importance 1
          // so retrieval ranking drops it out of the injected context.
          const entry = byId.get(id)!;
          const d = decodeMemory(entry.content);
          if (d) {
            await updateExisting(memory, id, d.statement, { ...d.meta, importance: 1, updatedAt: Date.now() });
          }
          continue;
        }
        if (await remove(memory, id)) {
          report.deleted += 1;
          state.totals.deleted += 1;
        }
      }
    }
  }

  // Memories the model never mentioned were reviewed and survive. Counting
  // only explicit KEEP actions reported `kept 0` for healthy production
  // batches; every batch entry is accounted for exactly once here.
  for (const id of byId.keys()) {
    if (!touched.has(id)) report.kept += 1;
  }
}

function statementOf(entry: MemoryEntry): string {
  const decoded = decodeMemory(entry.content);
  return decoded ? decoded.statement : entry.content.trim();
}
