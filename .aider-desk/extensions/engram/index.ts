/**
 * Engram Memory - automatic memory management for AiderDesk using a
 * secondary, local, OpenAI-compatible LLM.
 *
 * Target: AiderDesk 0.81.0.
 *
 * Everything here is built on API verified at git tag v0.81.0 of
 * hotovo/aider-desk:
 *   - Extension interface + ExtensionContext  (packages/common/src/extensions.ts)
 *   - MemoryContext (storeMemory / retrieveMemories / getMemory / deleteMemory /
 *     updateMemory / getAllMemories / isMemoryEnabled)
 *   - onAgentFinished / onPromptStarted / onPromptFinished / onTaskClosed /
 *     onImportantReminders events
 *   - getCommands() -> CommandDefinition[]
 *   - getConfigComponent / getConfigData / saveConfigData
 *
 * The secondary LLM is reached over plain HTTP by this extension. It is never
 * an agent, never gets tools, and never touches the main model's context.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import type {
  Extension,
  ExtensionContext,
  CommandDefinition,
  UIComponentDefinition,
  AgentFinishedEvent,
  PromptStartedEvent,
  PromptFinishedEvent,
  TaskClosedEvent,
  ImportantRemindersEvent,
  ContextMessage,
} from '@aiderdesk/extensions';

import { loadConfig, saveConfig, mergeConfig, resolveConfig, hasAgentOverride, type EngramConfig } from './src/config';
import { listAgentProfiles, type AgentMeta } from './src/agents';
import { activity, safeEndpoint, sanitizeStatements, type ActivityEntry, type ActivityLevel, type ActivityScope } from './src/activity';
import { ExtractionCursors, countInformative, hashMessages } from './src/cursor';
import { logger } from './src/logger';
import { loadState, saveState, projectStats, type EngramState, type ProjectStats } from './src/state';
import { runExtraction } from './src/extraction';
import { runConsolidation, type ConsolidationReport } from './src/consolidation';
import { retrieveForPrompt, stripBlock, wrapBlock } from './src/retrieval';
import {
  deterministicDedup,
  getMemoryContextSafely,
  importanceOf,
  isManaged,
  listAll,
  remove,
  retrieveScoped,
  scopeOf,
  statementOf,
} from './src/store';
import { decodeMemory } from './src/memory-format';
import { probe } from './src/llm';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CONFIG_PATH = join(__dirname, 'config.json');
const STATE_PATH = join(__dirname, 'state.json');
const CONFIG_JSX_PATH = join(__dirname, 'ConfigComponent.jsx');
const ACTIVITY_JSX_PATH = join(__dirname, 'EngramActivity.jsx');

/** AiderDesk UI component id for the activity badge + viewer. */
const ACTIVITY_COMPONENT_ID = 'engram-activity';

/** How many activity lines `/memory:activity` prints into the task log by default. */
const ACTIVITY_DUMP_LINES = 25;

/**
 * `onAgentFinished` fires for EVERY agent run, not just the ones the user
 * started. AiderDesk runs the same machinery for internal jobs, and their
 * `mode` is one of these:
 *
 *   - `subagent`             - a subagent spawned by the main agent
 *   - `compact-conversation` - automatic context compaction
 *   - `conflict-resolution`  - merge/conflict repair
 *
 * Extracting on those runs is pure token waste (they are not the user's
 * conversation, and a single task can spawn many) and it is what made the
 * extension look like it "loops": a long task with a dozen subagents queued a
 * dozen extraction rounds. Only a run whose mode is NOT in this set counts as
 * "the task is running". Kept as a set so an unknown future internal mode is a
 * one-line change rather than a heuristic that silently guesses wrong.
 */
const INTERNAL_AGENT_MODES = new Set(['subagent', 'compact-conversation', 'conflict-resolution']);

/**
 * Minimum wall-clock gap between two AUTOMATIC consolidation runs for the same
 * project.
 *
 * Consolidation keeps the round counter above `interval_tasks` when a run left
 * memories unreviewed, on purpose: the next trigger should retry them. But the
 * trigger is "an extraction round ended", and runConsolidation deliberately
 * does NOT reset the counter when batches failed. With a secondary endpoint
 * that fails most batches (observed live: finish_reason=length and HTTP errors
 * on 2 of every 3 batches), that combination made consolidation run on EVERY
 * extraction round - a retry storm that burned tokens against an endpoint that
 * was not going to answer. A cooldown keeps the retry intent ("soon") while
 * capping how often it can cost anything.
 */
const CONSOLIDATION_COOLDOWN_MS = 10 * 60 * 1000;

/**
 * Ceiling for the growing automatic-consolidation cooldown. The cooldown
 * doubles once per no-progress run (recordConsolidationProgress), so a store
 * stuck above MAX_BATCHES stops costing a batch of calls every 10 minutes and
 * settles at a few runs per day instead.
 */
const CONSOLIDATION_MAX_COOLDOWN_MS = 6 * 60 * 60 * 1000;

/**
 * Deepest stall streak that still widens the cooldown. 10 doublings of the
 * 10-minute base is already past CONSOLIDATION_MAX_COOLDOWN_MS; the cap only
 * keeps `2 ** stalls` finite against a hand-edited state.json.
 */
const CONSOLIDATION_STALL_CAP = 10;

/** The persisted no-progress streak for a project, clamped to a sane range. */
function stallRuns(stats: ProjectStats): number {
  return Math.min(CONSOLIDATION_STALL_CAP, Math.max(0, Math.floor(stats.consolidationStalls ?? 0)));
}

/** Automatic-consolidation cooldown for a streak of no-progress runs. */
function consolidationCooldown(stalls: number): number {
  const capped = Math.min(CONSOLIDATION_STALL_CAP, Math.max(0, Math.floor(stalls)));
  return Math.min(CONSOLIDATION_MAX_COOLDOWN_MS, CONSOLIDATION_COOLDOWN_MS * 2 ** capped);
}

/**
 * Consecutive automatic-round failures that trip the per-project circuit breaker.
 *
 * Production counters said 170 secondary-LLM calls, 98 of them failures: the
 * endpoint was down, malformed, or timing out more often than not, and every
 * failed round still cost its full price - the extraction call, the repair
 * call, the classification calls, each with up to 3 transport retries. A
 * secondary model that has answered nothing three rounds running is not going
 * to answer the fourth, and the agent does not need us to find that out again
 * with its own tokens.
 *
 * Automatic rounds only. `/memory:extract`, `/memory:consolidate` and the
 * viewer's buttons are the user asking directly, and always run.
 */
const BREAKER_TRIP_FAILURES = 3;

/** How long automatic rounds stay paused once the breaker trips. */
const BREAKER_COOLDOWN_MS = 15 * 60 * 1000;

function readJsx(path: string): string | undefined {
  try {
    if (!existsSync(path)) return undefined;
    return readFileSync(path, 'utf-8');
  } catch {
    return undefined;
  }
}

/**
 * Report to the user wherever possible: the task log when a task is open,
 * the extension log otherwise. The raw
 * `ctx.getTaskContext()?.addLogMessage(...)` chain is completely silent when
 * a command runs with no task open (e.g. invoked from a UI surface without a
 * task), which made command feedback disappear.
 *
 * This is the low-level sink; it does NOT touch the activity feed, so callers
 * that dump feed contents (e.g. `/memory:activity`) do not create a feedback
 * loop of entries about entries.
 */
function emit(context: ExtensionContext, level: 'info' | 'warn' | 'warning' | 'error', message: string): void {
  // The two sinks name the warning level differently ('warn' for
  // ExtensionContext.log, 'warning' for addLogMessage) - accept both here.
  const isWarning = level === 'warn' || level === 'warning';
  try {
    const taskContext = context.getTaskContext();
    if (taskContext && typeof taskContext.addLogMessage === 'function') {
      taskContext.addLogMessage(isWarning ? 'warning' : level, message);
      return;
    }
  } catch {
    /* fall through to the extension log */
  }
  try {
    context.log(message, isWarning ? 'warn' : level);
  } catch {
    if (isWarning) logger.warn(message);
    else if (level === 'error') logger.error(message);
    else logger.info(message);
  }
}

/**
 * Command feedback: emit it to the user AND record it in the activity feed.
 * Commands speak through the task log, which bypasses the logger, so their
 * feedback only reaches the feed if it is recorded here.
 */
function say(context: ExtensionContext, level: 'info' | 'warn' | 'warning' | 'error', message: string): void {
  const isWarning = level === 'warn' || level === 'warning';
  activity.record({ level: isWarning ? 'warn' : level, kind: 'command', message });
  emit(context, level, message);
}

const ACTIVITY_LEVELS: ActivityLevel[] = ['debug', 'info', 'warn', 'error'];

function isActivityLevel(value: unknown): value is ActivityLevel {
  return typeof value === 'string' && (ACTIVITY_LEVELS as string[]).includes(value);
}

function isPlainObjectArg(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** How many memories the viewer's memory list shows by default. */
const MEMORIES_VIEW_LIMIT = 60;

/**
 * Render the newest activity entries as log lines for `/memory:activity`.
 * Counts, ids and labels, plus the memory statements the entry carries — those
 * exist only when `ui.show_content` is on, and are secret-redacted at the feed.
 */
function formatActivityDump(entries: ActivityEntry[], total: number): string[] {
  const counts: Record<string, number> = {};
  for (const entry of entries) counts[entry.level] = (counts[entry.level] ?? 0) + 1;
  const lines = [
    `[Memory] activity: ${total} entries in the feed, showing ${entries.length}` +
      ` (warn ${counts.warn ?? 0}, error ${counts.error ?? 0})`,
  ];
  for (const entry of entries) {
    const time = new Date(entry.ts).toLocaleTimeString();
    const details = entry.details
      ? '  ' +
        Object.entries(entry.details)
          .map(([key, value]) => `${key}=${value}`)
          .join(' ')
      : '';
    lines.push(`[Memory] ${time} ${entry.level.padEnd(5)} ${entry.kind.padEnd(12)} ${entry.message}${details}`);
    for (const statement of entry.content ?? []) lines.push(`[Memory] ${' '.repeat(28)}· ${statement}`);
  }
  return lines;
}

/**
 * Header block of the activity viewer: what the extension is, what it is
 * pointed at, what it has done, and whether anything is running right now.
 * Every field is a count, an id or a label - nothing from a conversation.
 */
export interface EngramActivityStatus {
  version: string;
  enabled: boolean;
  /** False when AiderDesk Memory is switched off, so nothing can be stored. */
  memoryAvailable: boolean;
  /** An extraction or consolidation round is queued or running. */
  busy: boolean;
  agentId: string | null;
  /** True when this agent has its own override block in the config. */
  perAgent: boolean;
  llm: {
    transport: string;
    /** Model id served over HTTP. */
    model: string;
    /** AiderDesk model id, for the 'aiderdesk' transport. */
    modelId: string;
    /** Base URL with any embedded credentials removed. */
    endpoint: string;
  };
  logging: { enabled: boolean; level: ActivityLevel | string };
  activity: { show: boolean; limit: number; size: number; warnings: number; errors: number };
  /**
   * `showContent` is the switch that decides whether the viewer may show the
   * memory statements themselves (what was injected, what was written). When it
   * is false the feed carries counts only, and the viewer says so.
   */
  ui: { showContent: boolean };
  /**
   * Token-gating state, so the viewer can show WHY nothing is happening:
   * `cursorTasks` is how many tasks have a live extraction position, `paused`
   * is the circuit breaker holding automatic rounds off a failing endpoint,
   * `consolidationStalls` is how far automatic consolidation has backed off
   * after runs that rewrote nothing.
   */
  gating: { cursorTasks: number; paused: boolean; consolidationStalls: number };
  project: {
    dir: string;
    counts: {
      extractions: number;
      stored: number;
      updated: number;
      deleted: number;
      duplicates: number;
      obsolete: number;
    };
  };
  totals: { llmCalls: number; llmFailures: number; stored: number; updated: number; deleted: number };
  /** Newest entry of the whole feed, whatever its scope. */
  last: { ts: number; level: string; kind: string; message: string } | null;
}

export default class EngramMemoryExtension implements Extension {
  static metadata = {
    name: 'Engram Memory',
    version: '1.5.2',
    description:
      'Automatic long-term memory: extracts durable facts from conversations with a secondary local OpenAI-compatible LLM, dedupes/updates/resolves conflicts in AiderDesk Memory, consolidates periodically, injects only relevant memories, and shows what it is doing in a per-task activity viewer.',
    author: 'local',
    capabilities: ['commands', 'settings', 'events', 'ui'],
  };

  /**
   * Test seam (tests/run.ts scenario 8): redirect the config/state files so a
   * harness can drive this class end to end without touching the real
   * config.json/state.json. AiderDesk never sets it - production always uses
   * the extension directory.
   */
  static testPaths: { configPath?: string; statePath?: string } | null = null;

  private readonly configPath = EngramMemoryExtension.testPaths?.configPath ?? CONFIG_PATH;
  private readonly statePath = EngramMemoryExtension.testPaths?.statePath ?? STATE_PATH;
  private config: EngramConfig = loadConfig(this.configPath);
  private state: EngramState = loadState(this.statePath);
  private readonly abortController = new AbortController();
  /** Serialized work queue per project, so memory writes never interleave. */
  private readonly queues = new Map<string, Promise<void>>();
  private readonly inFlight = new Set<string>();
  /** Last user prompt per task, so retrieval has a query on the reminder hook. */
  private readonly lastPrompt = new Map<string, string>();
  /**
   * Per-task position of the last message a successful extraction round
   * consumed (src/cursor.ts). This is the gate that stops a task from paying
   * for the same transcript over and over.
   */
  private readonly cursors = new ExtractionCursors();
  /**
   * Per-project automatic-round health, see BREAKER_TRIP_FAILURES. In-memory:
   * it describes the endpoint as it is right now, not as it was before a
   * restart, and a restart re-probes it from scratch.
   */
  private readonly breaker = new Map<string, { failures: number; pausedUntil: number }>();
  /**
   * Agent profiles last seen through the authoritative API (project-scoped
   * contexts). The settings dialog runs without a project, so this cache plus
   * a disk scan (src/agents.ts) fill the gap there.
   */
  private agentMetaCache: AgentMeta[] = [];
  /**
   * Last automatic consolidation attempt per project (epoch ms). In-memory on
   * purpose: it is a rate limiter, not durable state, and the persistent
   * `stats.lastConsolidationAt` is used as a floor so a reload does not reset
   * the cooldown for a project that already consolidated recently.
   */
  private readonly lastAutoConsolidate = new Map<string, number>();

  // ---------------------------------------------------------------- lifecycle

  /**
   * Resolve the AiderDesk agent profile id for the current task.
   *
   * `TaskContext.getTaskAgentProfile()` is the documented resolver: task-level
   * agentProfileId first, project default second, task provider/model overrides
   * applied. Falls back to the raw task data, then to null, which means
   * "no per-agent override - use the global config".
   */
  private async agentIdFor(context: ExtensionContext): Promise<string | null> {
    const taskContext = context.getTaskContext();
    if (!taskContext) return null;
    try {
      const profile = await taskContext.getTaskAgentProfile();
      if (profile?.id) return profile.id;
    } catch {
      /* resolver unavailable - fall back to the raw task field */
    }
    return taskContext.data?.agentProfileId ?? null;
  }

  async onLoad(context: ExtensionContext): Promise<void> {
    logger.bind(context);
    logger.setConfig(this.config.logging);
    // Everything the logger emits from here on also lands in the activity
    // feed, tagged with whatever scope the current operation set.
    logger.setSink((level, message) => activity.record({ level, message }));
    activity.setLimit(this.config.ui.activity_limit);

    const overrides = Object.keys(this.config.agents);
    activity.withScope({ kind: 'lifecycle' }, () =>
      logger.info(
        `loaded (secondary LLM: ${this.config.secondary_llm.model} @ ${safeEndpoint(this.config.secondary_llm.base_url)}, ` +
          `per-agent overrides: ${overrides.length ? overrides.join(', ') : 'none'})`,
      ),
    );

    if (!this.config.enabled) {
      activity.withScope({ kind: 'lifecycle' }, () =>
        logger.info('extension is disabled by configuration - idle'),
      );
      return;
    }

    const memory = getMemoryContextSafely(context);
    if (!memory) {
      activity.withScope({ kind: 'lifecycle' }, () =>
        logger.warn(
          'AiderDesk Memory is disabled or unavailable - enable it in Settings > Memory, otherwise Engram cannot store anything',
        ),
      );
    }

    // Fire-and-forget connectivity probe so the log tells you immediately
    // whether the secondary endpoint is reachable. Never blocks startup, and
    // `withScope` keeps it from wiping the scope a concurrent extraction set.
    void probe(this.config.secondary_llm)
      .then((result) => {
        activity.withScope({ kind: 'lifecycle' }, () => {
          if (result.ok) {
            logger.info(`secondary LLM reachable (${result.durationMs} ms)`);
          } else {
            logger.warn(`secondary LLM not reachable: ${result.kind} - ${result.message}`);
          }
        });
      })
      .catch(() => undefined);

    context.addDisposable(() => () => {
      this.abortController.abort('extension unloaded');
    });
  }

  async onUnload(): Promise<void> {
    this.abortController.abort('extension unloaded');
    // Let queued work settle so we do not leave half-written memories.
    await Promise.allSettled(Array.from(this.queues.values())).catch(() => undefined);
    saveState(this.statePath, this.state);
  }

  // ---------------------------------------------------------------- retrieval

  /**
   * Remember the current prompt so retrieval can run on the reminder hook,
   * which is the point where content actually reaches the main model.
   */
  async onPromptStarted(event: PromptStartedEvent, context: ExtensionContext): Promise<void> {
    const agentId = await this.agentIdFor(context);
    const cfg = resolveConfig(this.config, agentId);
    if (!cfg.enabled || !cfg.retrieval.enabled) return;
    const taskId = context.getTaskContext()?.data?.id;
    if (taskId) this.lastPrompt.set(taskId, event.prompt ?? '');
  }

  /**
   * Injection point. `remindersContent` is appended to the user request inside
   * <ThisIsImportant> by src/main/agent/optimizer.ts.
   */
  async onImportantReminders(
    event: ImportantRemindersEvent,
    context: ExtensionContext,
  ): Promise<void | Partial<ImportantRemindersEvent>> {
    const taskContext = context.getTaskContext();
    const taskId = taskContext?.data?.id;
    const prompt = (taskId && this.lastPrompt.get(taskId)) || '';
    if (!prompt) return;

    const agentId = event.agentProfile?.id ?? (await this.agentIdFor(context));
    const cfg = resolveConfig(this.config, agentId);
    if (!cfg.enabled || !cfg.retrieval.enabled) return;

    activity.setScope({ kind: 'retrieval', projectDir: context.getProjectDir(), taskId, agentId });
    try {
      const outcome = await retrieveForPrompt(context, context.getProjectDir(), prompt, cfg);
      if (!outcome.block) return;

      // What the user asked for: the viewer must show WHAT was injected, not
      // only that something was. Statements are attached only when
      // `ui.show_content` is on, and always secret-redacted (see contentFor).
      activity.record({
        level: 'info',
        kind: 'retrieval',
        message: `injected ${outcome.count} of ${outcome.hits} retrieved memories into the prompt`,
        details: { injected: outcome.count, hits: outcome.hits },
        content: this.contentFor(outcome.statements),
      });
      this.refreshActivityUI(context, taskId);

      // Strip-then-append: the reminder hook can fire more than once per
      // request, and append-only injection would stack a second copy of the
      // same block (observed live). Replacement keeps exactly one.
      return {
        remindersContent: [stripBlock(event.remindersContent), wrapBlock(outcome.block)].filter(Boolean).join('\n\n'),
      };
    } catch (error) {
      logger.warn(`retrieval failed: ${error instanceof Error ? error.message : String(error)}`);
      return;
    } finally {
      activity.clearScope();
    }
  }

  // -------------------------------------------------------------- extraction

  async onAgentFinished(event: AgentFinishedEvent, context: ExtensionContext): Promise<void> {
    if (event.aborted) return;
    this.cacheAgentProfiles(context);
    const agentId = await this.agentIdFor(context);

    // Only the user's own task runs may spend tokens. Subagent, compaction and
    // conflict-resolution runs all dispatch this same event with their own
    // `mode`; treating them as extraction rounds is the loop/burn the user saw.
    if (INTERNAL_AGENT_MODES.has(event.mode)) {
      activity.withScope({ kind: 'lifecycle', projectDir: context.getProjectDir(), agentId }, () =>
        logger.debug(`skipping ${event.mode} run - not a task turn`),
      );
      return;
    }

    const cfg = resolveConfig(this.config, agentId);
    if (!cfg.enabled || !cfg.extraction.enabled) return;
    if (cfg.extraction.trigger !== 'agent_end') return;

    this.queueExtraction(context, event.contextMessages, agentId);
    // Consolidation must also be driven by agent_end rounds: wiring it only
    // into onTaskClosed left it unrunnable for the default trigger mode.
    this.maybeConsolidate(context, agentId);
  }

  async onPromptFinished(event: PromptFinishedEvent, context: ExtensionContext): Promise<void> {
    this.cacheAgentProfiles(context);
    // Resolve the per-agent config BEFORE checking the trigger: the global
    // trigger must not mask a per-agent prompt_end override (and vice versa).
    const agentId = await this.agentIdFor(context);
    const cfg = resolveConfig(this.config, agentId);
    if (!cfg.enabled || !cfg.extraction.enabled || cfg.extraction.trigger !== 'prompt_end') return;

    const taskContext = context.getTaskContext();
    if (!taskContext) return;

    void (async () => {
      try {
        const messages = await taskContext.getContextMessages();
        this.queueExtraction(context, messages, agentId);
      } catch (error) {
        logger.warn(`cannot read conversation: ${error instanceof Error ? error.message : String(error)}`);
      }
    })();
  }

  async onTaskClosed(event: TaskClosedEvent, context: ExtensionContext): Promise<void> {
    this.cacheAgentProfiles(context);
    // The prompt map is keyed by task id; without this the entry of every
    // task that ever ran stays for the life of the process (the reminder
    // hook is keyed by the *current* task, so a stale entry is never read -
    // it is pure unbounded growth on a long-lived desktop session).
    const closedTaskId = event.task?.id ?? context.getTaskContext()?.data?.id;
    if (closedTaskId) this.lastPrompt.delete(closedTaskId);
    const agentId = event.task?.agentProfileId ?? (await this.agentIdFor(context));
    const cfg = resolveConfig(this.config, agentId);
    if (!cfg.enabled) return;

    if (cfg.extraction.enabled && cfg.extraction.trigger === 'task_end') {
      const taskContext = context.getTaskContext();
      if (taskContext) {
        void (async () => {
          try {
            const messages = await taskContext.getContextMessages();
            // Queue first: queueExtraction reads the cursor synchronously, so
            // the final round still sees what the earlier rounds consumed and
            // extracts only the tail. Dropping the cursor before it would make
            // every task_end round re-read the whole conversation.
            this.queueExtraction(context, messages, agentId);
          } catch {
            /* task already torn down */
          } finally {
            // The task is gone; its position in a conversation that no longer
            // exists must not stay in the table.
            if (closedTaskId) this.cursors.forget(closedTaskId);
          }
        })();
      }
    } else if (closedTaskId) {
      this.cursors.forget(closedTaskId);
    }

    // Consolidation is driven by extraction rounds, not by wall-clock, so the
    // interval means the same thing for every trigger mode.
    this.maybeConsolidate(context, agentId);
  }

  // ------------------------------------------------------------- scheduling

  /**
   * Fire-and-forget. The handler returns immediately, so the main agent is
   * never blocked by the (potentially slow) secondary LLM. Work is serialized
   * per project so two runs cannot race on the same memories.
   */
  /**
   * Fire-and-forget. The handler returns immediately, so the main agent is
   * never blocked by the (potentially slow) secondary LLM. Work is serialized
   * per project so two runs cannot race on the same memories.
   *
   * Three gates run BEFORE anything is queued, so a round that has no reason to
   * spend tokens never reaches the queue, the endpoint, or the counters:
   *  - the circuit breaker (endpoint failing rounds in a row),
   *  - the cursor (nothing new since the last successful round),
   *  - the content hash (the new messages are byte-identical to last time).
   *
   * `manual` is the user asking directly - `/memory:extract`, the viewer's
   * "Extract now". It reads the whole conversation and ignores both gates.
   */
  private queueExtraction(
    context: ExtensionContext,
    messages: ContextMessage[],
    agentId?: string | null,
    options: { manual?: boolean } = {},
  ): void {
    const projectDir = context.getProjectDir();
    if (!messages?.length) return;

    const cfg = resolveConfig(this.config, agentId);
    if (!cfg.enabled || !cfg.extraction.enabled) return;

    const taskId = context.getTaskContext()?.data?.id ?? '';
    const taskKey = taskId || projectDir;

    if (!options.manual) {
      const paused = this.breakerPaused(projectDir);
      if (paused) {
        logger.debug('extraction skipped - secondary LLM failing repeatedly, automatic rounds paused');
        return;
      }

      const { fresh, resynced } = this.cursors.sinceLast(taskKey, messages);
      if (!fresh.length) {
        logger.debug('extraction skipped - nothing new since the last round');
        return;
      }

      const informative = countInformative(fresh);
      const floor = Math.max(0, Math.round(cfg.extraction.min_new_messages));
      if (!resynced && informative < floor) {
        logger.debug(
          `extraction skipped - ${informative} new message(s) since the last round, needs ${floor}`,
        );
        return;
      }

      const hash = hashMessages(fresh);
      if (!resynced && hash === this.cursors.lastHash(taskKey)) {
        logger.debug('extraction skipped - the new messages are identical to the last round');
        return;
      }

      this.inFlight.add(projectDir);
      this.enqueue(projectDir, async () => {
        await this.runExtractionRound(context, fresh, projectDir, taskId, taskKey, agentId, hash);
      });
      return;
    }

    this.inFlight.add(projectDir);
    this.enqueue(projectDir, async () => {
      await this.runExtractionRound(context, messages, projectDir, taskId, taskKey, agentId, undefined);
    });
  }

  /**
   * One extraction round, with the cursor and breaker bookkeeping that must
   * happen exactly once per round: the cursor advances only on a round that
   * actually read the content, so a failed round leaves the same messages
   * queued for the next trigger.
   */
  private async runExtractionRound(
    context: ExtensionContext,
    messages: ContextMessage[],
    projectDir: string,
    taskId: string,
    taskKey: string,
    agentId?: string | null,
    hash?: string,
  ): Promise<void> {
    if (this.inFlight.has(projectDir) && this.inFlight.size > 1) {
      logger.debug('extraction already running for this project - queued');
    }
    const cfg = resolveConfig(this.config, agentId);
    logger.setConfig(cfg.logging);
    activity.setScope({ kind: 'extraction', projectDir, taskId, agentId });
    try {
      if (hasAgentOverride(this.config, agentId)) {
        logger.debug(`extraction using agent profile "${agentId}" overrides`);
      }
      if (!cfg.enabled || !cfg.extraction.enabled) {
        logger.debug('extraction skipped - disabled for this agent');
        return;
      }
      const report = await runExtraction({
        context,
        messages,
        projectDir,
        taskId,
        config: cfg,
        state: this.state,
        statePath: this.statePath,
        signal: this.abortController.signal,
        taskContext: context.getTaskContext(),
      });
      if (report.failure) {
        logger.debug(`extraction report: ${report.failure}`);
        this.breakerRecord(projectDir, true, 'extraction');
      } else {
        if (hash !== undefined) this.cursors.remember(taskKey, messages);
        this.breakerRecord(projectDir, false, 'extraction');
        // Digest entry: the module's own log line is the long form, this is
        // the machine-readable one the viewer renders as chips.
        activity.record({
          level: 'info',
          kind: 'extraction',
          message:
            `round done: ${report.stored} stored, ${report.updated} updated, ` +
            `${report.duplicates} duplicate, ${report.obsolete} obsolete`,
          details: {
            candidates: report.candidates,
            stored: report.stored,
            updated: report.updated,
            duplicates: report.duplicates,
            obsolete: report.obsolete,
            skipped: report.skipped,
          },
          content: this.contentFor(report.writes.map((write) => `${write.action}: ${write.text}`)),
        });
      }
      this.countRound(projectDir);
    } catch (error) {
      logger.warn(`extraction error: ${error instanceof Error ? error.message : String(error)}`);
      this.breakerRecord(projectDir, true, 'extraction');
    } finally {
      this.inFlight.delete(projectDir);
      logger.setConfig(this.config.logging);
      activity.clearScope();
      this.refreshActivityUI(context, taskId);
    }
  }

  // ------------------------------------------------------- circuit breaker

  /** True while automatic rounds for this project are paused. */
  private breakerPaused(projectDir: string): boolean {
    const entry = this.breaker.get(projectDir);
    if (!entry) return false;
    if (Date.now() < entry.pausedUntil) return true;
    // Cooldown expired: give the endpoint another chance, starting clean.
    if (entry.failures >= BREAKER_TRIP_FAILURES) {
      this.breaker.set(projectDir, { failures: 0, pausedUntil: 0 });
      logger.info('automatic memory rounds resumed');
      activity.record({ level: 'info', kind: 'lifecycle', message: 'automatic rounds resumed' });
    }
    return false;
  }

  /**
   * One automatic round's outcome. `BREAKER_TRIP_FAILURES` failures in a row
   * pause automatic rounds for BREAKER_COOLDOWN_MS; a single success clears
   * the streak.
   */
  private breakerRecord(projectDir: string, failed: boolean, what: 'extraction' | 'consolidation'): void {
    if (!failed) {
      this.breaker.delete(projectDir);
      return;
    }
    const entry = this.breaker.get(projectDir) ?? { failures: 0, pausedUntil: 0 };
    entry.failures += 1;
    this.breaker.set(projectDir, entry);
    if (entry.failures === BREAKER_TRIP_FAILURES) {
      entry.pausedUntil = Date.now() + BREAKER_COOLDOWN_MS;
      logger.warn(
        `${what} failed ${entry.failures} rounds in a row - automatic memory rounds paused for ` +
          `${Math.round(BREAKER_COOLDOWN_MS / 60000)} min (check Settings > Engram, or run /memory:stats)`,
      );
    }
  }


  /**
   * Ask the renderer to re-fetch the activity badge's data. Best-effort: the
   * EventManager is absent in headless runs, and a stale badge is harmless.
   */
  private refreshActivityUI(context: ExtensionContext | null, taskId?: string): void {
    if (!this.config.ui.show_activity) return;
    try {
      context?.triggerUIDataRefresh(ACTIVITY_COMPONENT_ID, taskId);
    } catch {
      /* no renderer attached */
    }
  }

  private maybeConsolidate(context: ExtensionContext, agentId?: string | null): void {
    const cfg = resolveConfig(this.config, agentId);
    if (!cfg.enabled || !cfg.consolidation.enabled) return;
    const projectDir = context.getProjectDir();
    const taskId = context.getTaskContext()?.data?.id ?? '';
    const stats = projectStats(this.state, projectDir);
    if (stats.tasksSinceConsolidation < Math.max(2, cfg.consolidation.interval_tasks)) return;

    // Rate-limit automatic runs (see CONSOLIDATION_COOLDOWN_MS). A manual
    // /memory:consolidate is a different code path and is never throttled.
    // The window grows per no-progress run (recordConsolidationProgress): a
    // store stuck above the batch cap keeps the round counter above the
    // interval on purpose (src/consolidation.ts), so a flat 10-minute cooldown
    // alone means a fresh batch of LLM calls every 10 minutes, forever.
    const now = Date.now();
    const anchor = Math.max(stats.lastConsolidationAt ?? 0, this.lastAutoConsolidate.get(projectDir) ?? 0);
    const stalls = stallRuns(stats);
    const cooldown = consolidationCooldown(stalls);
    if (now - anchor < cooldown) {
      logger.debug(
        'consolidation skipped - cooldown after the previous run' +
          (stalls > 0 ? ` (${stalls} no-progress run(s), window ${Math.round(cooldown / 60000)} min)` : ''),
      );
      return;
    }
    // Consolidation is the most expensive thing this extension does - many
    // batches over every memory in the project. When the endpoint has been
    // answering nothing, it is also the one that can burn the most against it.
    if (this.breakerPaused(projectDir)) {
      logger.debug('consolidation skipped - secondary LLM failing repeatedly, automatic rounds paused');
      return;
    }
    this.lastAutoConsolidate.set(projectDir, now);

    this.enqueue(projectDir, async () => {
      const effective = resolveConfig(this.config, agentId);
      logger.setConfig(effective.logging);
      activity.setScope({ kind: 'consolidation', projectDir, taskId, agentId });
      try {
        if (!effective.enabled || !effective.consolidation.enabled) {
          logger.debug('consolidation skipped - disabled for this agent');
          return;
        }
        const report = await runConsolidation({
          context,
          projectDir,
          config: effective,
          state: this.state,
          statePath: this.statePath,
          signal: this.abortController.signal,
          taskContext: context.getTaskContext(),
        });
        if (report.failure) {
          logger.debug(`consolidation report: ${report.failure}`);
          this.breakerRecord(projectDir, true, 'consolidation');
        } else {
          this.breakerRecord(projectDir, false, 'consolidation');
          // Digest entry: the module's own log line is the long form, this is
          // the machine-readable one the viewer renders as chips.
          activity.record({
            level: 'info',
            kind: 'consolidation',
            message:
              `round done: ${report.scanned} scanned, merged ${report.merged}, ` +
              `updated ${report.updated}, deleted ${report.deleted}, kept ${report.kept}`,
            details: {
              scanned: report.scanned,
              merged: report.merged,
              updated: report.updated,
              deleted: report.deleted,
              kept: report.kept,
              skipped: report.skipped,
            },
          });
        }
        this.recordConsolidationProgress(projectDir, report);
      } catch (error) {
        logger.warn(`consolidation error: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        logger.setConfig(this.config.logging);
        activity.clearScope();
        this.refreshActivityUI(context, taskId);
      }
    });
  }

  /**
   * Track whether an automatic consolidation run actually rewrote anything, so
   * `maybeConsolidate` can widen its cooldown instead of re-running the same
   * window every 10 minutes.
   *
   * A run is counted as a stall only when it rewrote nothing AND left work
   * undone (a failed batch, or batches deferred past MAX_BATCHES). A clean run
   * that found nothing to merge is not a stall - it resets the round counter in
   * src/consolidation.ts, so backing off would only starve a healthy store.
   * The streak is persisted because the in-memory cooldown anchor is not: an
   * extension reload must not reset the backoff of a stuck store.
   */
  private recordConsolidationProgress(projectDir: string, report: ConsolidationReport): void {
    const stats = projectStats(this.state, projectDir);
    const rewrite = report.merged + report.updated + report.deleted;
    const unfinished = (report.failedBatches ?? 0) + (report.deferredBatches ?? 0);
    if (report.aborted || rewrite > 0 || unfinished === 0) {
      if (stallRuns(stats) > 0) {
        logger.debug(`consolidation made progress (${rewrite} rewrite(s)) - stall backoff reset`);
      }
      stats.consolidationStalls = 0;
    } else {
      stats.consolidationStalls = stallRuns(stats) + 1;
      const why = report.failedBatches
        ? `${report.failedBatches} batch(es) failed`
        : `${report.deferredBatches} batch(es) beyond the cap`;
      logger.warn(
        `consolidation made no progress (${why}); automatic rounds backed off to ` +
          `${Math.round(consolidationCooldown(stats.consolidationStalls) / 60000)} min`,
      );
    }
    saveState(this.statePath, this.state);
  }

  private countRound(projectDir: string): void {
    const stats = projectStats(this.state, projectDir);
    stats.tasksSinceConsolidation += 1;
    saveState(this.statePath, this.state);
  }

  private enqueue(key: string, fn: () => Promise<void>): void {
    const previous = this.queues.get(key) ?? Promise.resolve();
    const next = previous
      .then(fn)
      .catch((error) => {
        logger.warn(`queued job failed: ${error instanceof Error ? error.message : String(error)}`);
      });
    this.queues.set(key, next);
  }

  /**
   * Test seam (tests/run.ts scenario 8): await every queued extraction and
   * consolidation without unloading the extension, so a harness can assert on
   * the result of fire-and-forget work. AiderDesk never calls it.
   */
  async drainQueues(): Promise<void> {
    await Promise.allSettled(Array.from(this.queues.values()));
  }

  // ---------------------------------------------------------------- commands

  getCommands(context: ExtensionContext): CommandDefinition[] {
    return [
      {
        name: 'memory:extract',
        description: 'Engram: extract memories from this conversation now (secondary LLM, runs in background)',
        execute: async (_args, ctx) => {
          const taskContext = ctx.getTaskContext();
          if (!taskContext) {
            say(ctx, 'warn', '[Memory] no task context - open a task first');
            return;
          }
          const agentId = await this.agentIdFor(ctx);
          const cfg = resolveConfig(this.config, agentId);
          if (!cfg.enabled || !cfg.extraction.enabled) {
            say(ctx, 'info', '[Memory] extraction is disabled for this agent');
            return;
          }
          let messages: ContextMessage[];
          try {
            messages = await taskContext.getContextMessages();
          } catch (error) {
            say(ctx, 'warning', `[Memory] cannot read the conversation: ${error instanceof Error ? error.message : String(error)}`);
            return;
          }
          say(ctx, 'info', `[Memory] extraction started (background, agent ${agentId ?? 'global'})`);
          this.queueExtraction(ctx, messages, agentId, { manual: true });
        },
      },
      {
        name: 'memory:consolidate',
        description: 'Engram: consolidate project + global memories (dedupe, merge, resolve conflicts)',
        arguments: [{ description: 'force - disable safe mode (aggressive merge and delete)' }],
        execute: async (args, ctx) => {
          const projectDir = ctx.getProjectDir();
          const agentId = await this.agentIdFor(ctx);
          const cfg = resolveConfig(this.config, agentId);
          if (!cfg.enabled || !cfg.consolidation.enabled) {
            say(ctx, 'info', '[Memory] consolidation is disabled for this agent');
            return;
          }
          const force = args.includes('force');
          say(ctx, 'info', `[Memory] consolidation started (background, agent ${agentId ?? 'global'})`);
          this.enqueue(projectDir, async () => {
            const report = await runConsolidation({
              context: ctx,
              projectDir,
              config: {
                ...cfg,
                consolidation: {
                  ...cfg.consolidation,
                  safe_mode: force ? false : cfg.consolidation.safe_mode,
                },
              },
              state: this.state,
              statePath: this.statePath,
              // Forced consolidation runs up to 6 batches at 120 s each. Without
              // the unload signal it kept `onUnload` awaiting a job that could
              // not be cancelled, hanging the extension reload for minutes.
              signal: this.abortController.signal,
              taskContext: ctx.getTaskContext(),
            });
            // A manual run is never throttled, and a successful one is proof
            // the endpoint is healthy again - clear any automatic backoff.
            this.recordConsolidationProgress(projectDir, report);
            const line = report.failure
              ? `consolidation failed: ${report.failure}`
              : `consolidated ${report.scanned} memories -> merged ${report.merged}, updated ${report.updated}, deleted ${report.deleted}, kept ${report.kept}`;
            const failedNote =
              report.failedBatches && !report.failure
                ? ` (${report.failedBatches} batch(es) failed${report.batchFailures?.length ? `: ${report.batchFailures.slice(0, 2).join('; ')}` : ''} - retried on the next consolidation)`
                : '';
            say(ctx, 'info', `[Memory] ${line}${failedNote}`);
          });
        },
      },
      {
        name: 'memory:dedup',
        description: 'Engram: remove exact duplicate memories deterministically (no LLM involved)',
        execute: async (_args, ctx) => {
          const memory = getMemoryContextSafely(ctx);
          if (!memory) {
            say(ctx, 'warning', '[Memory] AiderDesk Memory is disabled/unavailable');
            return;
          }
          const projectDir = ctx.getProjectDir();
          this.enqueue(projectDir, async () => {
            const report = await deterministicDedup(memory);
            this.state.totals.deleted += report.removed;
            // Dedup scans every project; only the removals that happened in
            // THIS project belong in its per-project counter (the report
            // breaks them down by project id).
            projectStats(this.state, projectDir).deleted += report.removedByProject[projectDir] ?? 0;
            saveState(this.statePath, this.state);
            say(
              ctx,
              'info',
              `[Memory] dedup: scanned ${report.scanned} managed memories, removed ${report.removed} exact duplicate(s)`,
            );
          });
        },
      },
      {
        name: 'memory:activity',
        description: 'Engram: show what the extension has been doing (recent activity feed)',
        arguments: [{ description: 'how many recent entries to show (default 25)' }],
        execute: async (args, ctx) => {
          const requested = parseInt(args.find((arg) => /^\d+$/.test(arg)) ?? '', 10);
          const count = Number.isFinite(requested) && requested > 0 ? requested : ACTIVITY_DUMP_LINES;
          const entries = activity.list({ scope: 'all', minLevel: 'debug', limit: count });
          if (!entries.length) {
            emit(ctx, 'info', '[Memory] activity: nothing recorded yet this session');
            return;
          }
          for (const line of formatActivityDump(entries, activity.size())) emit(ctx, 'info', line);
        },
      },
      {
        name: 'memory:stats',
        description: 'Engram: show memory statistics and secondary LLM status',
        execute: async (_args, ctx) => {
          const projectDir = ctx.getProjectDir();
          const memory = getMemoryContextSafely(ctx);
          if (!memory) {
            say(ctx, 'warning', '[Memory] AiderDesk Memory is disabled/unavailable');
            return;
          }
          const all = await listAll(memory);
          const managed = all.filter((entry) => entry?.id && isManaged(entry));
          const mine = managed.filter((entry) => (entry.projectId ?? '') === projectDir);
          const global = managed.filter((entry) => (entry.projectId ?? '') === '');
          const native = all.filter((entry) => entry?.id && !isManaged(entry));
          const stats = projectStats(this.state, projectDir);
          const agentId = await this.agentIdFor(ctx);
          const cfg = resolveConfig(this.config, agentId);

          const lines = [
            `[Memory] AiderDesk entries: ${all.length} (Engram-managed ${managed.length}, native ${native.length})`,
            `[Memory] this project: ${mine.length} | global: ${global.length}`,
            `[Memory] agent: ${agentId ?? 'none'} (${hasAgentOverride(this.config, agentId) ? 'per-agent overrides applied' : 'global config'})`,
            `[Memory] project counters: extractions ${stats.extractions}, stored ${stats.stored}, updated ${stats.updated}, deleted ${stats.deleted}, duplicates ${stats.duplicates}, obsolete ${stats.obsolete}`,
            `[Memory] totals: LLM calls ${this.state.totals.llmCalls} (failures ${this.state.totals.llmFailures}), stored ${this.state.totals.stored}, updated ${this.state.totals.updated}, deleted ${this.state.totals.deleted}`,
            `[Memory] extraction: enabled=${cfg.extraction.enabled} trigger=${cfg.extraction.trigger} min_importance=${cfg.extraction.min_importance}`,
            `[Memory] retrieval: enabled=${cfg.retrieval.enabled} max_memories=${cfg.retrieval.max_memories} min_importance=${cfg.retrieval.min_importance} min_overlap=${cfg.retrieval.min_overlap}`,
            `[Memory] consolidation: ${stats.tasksSinceConsolidation}/${cfg.consolidation.interval_tasks} rounds, safe_mode=${cfg.consolidation.safe_mode}, cooldown=${Math.round(consolidationCooldown(stallRuns(stats)) / 60000)} min (${stallRuns(stats)} no-progress run(s))`,
            `[Memory] secondary LLM: ${cfg.secondary_llm.model} @ ${cfg.secondary_llm.base_url}`,
          ];
          for (const line of lines) say(ctx, 'info', line);

          const result = await probe(cfg.secondary_llm);
          say(
            ctx,
            result.ok ? 'info' : 'warning',
            result.ok
              ? `[Memory] secondary LLM reachable, replied in ${result.durationMs} ms`
              : `[Memory] secondary LLM ${result.kind}: ${result.message}`,
          );
        },
      },
      {
        name: 'memory:forget',
        description: 'Engram: delete the memory that best matches the given text',
        arguments: [{ description: 'text describing the memory to forget', required: true }],
        execute: async (args, ctx) => {
          const query = args.join(' ').trim();
          if (!query) {
            say(ctx, 'warning', '[Memory] usage: /memory:forget <text>');
            return;
          }
          const memory = getMemoryContextSafely(ctx);
          if (!memory) {
            say(ctx, 'warning', '[Memory] AiderDesk Memory is disabled/unavailable');
            return;
          }
          const cfg = resolveConfig(this.config, await this.agentIdFor(ctx));
          const candidates = await retrieveScoped(
            memory,
            ctx.getProjectDir(),
            query,
            3,
            cfg.retrieval.include_global,
          );
          if (!candidates.length) {
            say(ctx, 'info', '[Memory] no matching memory found');
            return;
          }
          const target = candidates[0];
          const decoded = decodeMemory(target.content);
          const ok = await remove(memory, target.id);
          say(
            ctx,
            ok ? 'info' : 'warning',
            ok
              ? `[Memory] forgot: ${(decoded ? decoded.statement : target.content).slice(0, 160)}`
              : '[Memory] delete failed',
          );
        },
      },
      {
        name: 'memory:clear-project',
        description: 'Engram: delete every Engram-managed memory of this project (requires the word "confirm")',
        arguments: [{ description: 'confirm', required: true, options: ['confirm'] }],
        execute: async (args, ctx) => {
          if (!args.includes('confirm')) {
            say(ctx, 'warning', '[Memory] refusing to clear without the explicit "confirm" argument');
            return;
          }
          const projectDir = ctx.getProjectDir();
          const memory = getMemoryContextSafely(ctx);
          if (!memory) {
            say(ctx, 'warning', '[Memory] AiderDesk Memory is disabled/unavailable');
            return;
          }
          const all = await listAll(memory);
          const targets = all.filter(
            (entry) => entry?.id && isManaged(entry) && (entry.projectId ?? '') === projectDir,
          );
          let deleted = 0;
          for (const entry of targets) {
            if (await remove(memory, entry.id)) deleted += 1;
          }
          this.state.totals.deleted += deleted;
          // Every removed entry belongs to this project (the filter above is
          // exact), so book them all into the per-project counter too.
          projectStats(this.state, projectDir).deleted += deleted;
          saveState(this.statePath, this.state);
          say(ctx, 'info', `[Memory] cleared ${deleted} project memories`);
        },
      },
    ];
  }

  // ---------------------------------------------------------------- settings

  getConfigComponent(_context: ExtensionContext): string | undefined {
    return readJsx(CONFIG_JSX_PATH);
  }

  // ------------------------------------------------------------------- UI

  /**
   * The activity surface: a compact badge in the task status bar that opens a
   * live viewer for the current task. Registered only when
   * `ui.show_activity` is on, so turning it off removes the badge entirely.
   */
  getUIComponents(_context: ExtensionContext): UIComponentDefinition[] {
    if (!this.config.ui.show_activity) return [];
    const jsx = readJsx(ACTIVITY_JSX_PATH);
    if (!jsx) return [];
    return [
      {
        id: ACTIVITY_COMPONENT_ID,
        placement: 'task-status-bar-right',
        name: 'Engram activity',
        jsx,
        loadData: true,
        noDataCache: true,
      },
    ];
  }

  async getUIExtensionData(componentId: string, context: ExtensionContext): Promise<unknown> {
    if (componentId !== ACTIVITY_COMPONENT_ID) return undefined;
    return this.activityStatus(context);
  }

  /**
   * Backing actions for the activity viewer. Every return value is plain JSON
   * (counts, ids, short labels) - never conversation content, never secrets.
   */
  async executeUIExtensionAction(
    componentId: string,
    action: string,
    args: unknown[],
    context: ExtensionContext,
  ): Promise<unknown> {
    if (componentId !== ACTIVITY_COMPONENT_ID) return undefined;
    const opts = (isPlainObjectArg(args[0]) ? args[0] : {}) as Record<string, unknown>;
    const taskId =
      typeof opts.taskId === 'string' && opts.taskId
        ? opts.taskId
        : context.getTaskContext()?.data?.id ?? undefined;
    const projectDir = typeof opts.projectDir === 'string' && opts.projectDir ? opts.projectDir : context.getProjectDir();

    switch (action) {
      case 'status':
        return this.activityStatus(context);

      case 'activity': {
        const entries = this.filteredActivity(opts, taskId, projectDir);
        return { entries, status: await this.activityStatus(context) };
      }

      case 'clear': {
        activity.clear();
        return { entries: [], status: await this.activityStatus(context) };
      }

      case 'refresh': {
        this.refreshActivityUI(context, taskId);
        return { ok: true };
      }

      case 'extract': {
        const taskContext = context.getTaskContext();
        if (!taskContext) return { error: 'no task context - open a task first' };
        const agentId = await this.agentIdFor(context);
        const cfg = resolveConfig(this.config, agentId);
        if (!cfg.enabled || !cfg.extraction.enabled) return { error: 'extraction is disabled for this agent' };
        try {
          const messages = await taskContext.getContextMessages();
          this.queueExtraction(context, messages, agentId, { manual: true });
          return { started: true };
        } catch (error) {
          return { error: `cannot read the conversation: ${error instanceof Error ? error.message : String(error)}` };
        }
      }

      case 'consolidate': {
        const agentId = await this.agentIdFor(context);
        const cfg = resolveConfig(this.config, agentId);
        if (!cfg.enabled || !cfg.consolidation.enabled) return { error: 'consolidation is disabled for this agent' };
        const force = opts.force === true;
        this.enqueue(projectDir, async () => {
          const effective = resolveConfig(this.config, agentId);
          logger.setConfig(effective.logging);
          activity.setScope({ kind: 'consolidation', projectDir, taskId, agentId });
          try {
            const report = await runConsolidation({
              context,
              projectDir,
              config: {
                ...effective,
                consolidation: { ...effective.consolidation, safe_mode: force ? false : effective.consolidation.safe_mode },
              },
              state: this.state,
              statePath: this.statePath,
              signal: this.abortController.signal,
              taskContext: context.getTaskContext(),
            });
            // Same as the /memory:consolidate command: a manual, unthrottled
            // run that made progress clears the automatic backoff.
            this.recordConsolidationProgress(projectDir, report);
          } catch (error) {
            logger.warn(`consolidation error: ${error instanceof Error ? error.message : String(error)}`);
          } finally {
            logger.setConfig(this.config.logging);
            activity.clearScope();
            this.refreshActivityUI(context, taskId);
          }
        });
        return { started: true };
      }

      case 'dump': {
        const entries = this.filteredActivity(opts, taskId, projectDir);
        const lines = formatActivityDump(entries, activity.size());
        for (const line of lines) emit(context, 'info', line);
        return { lines, count: entries.length };
      }

      case 'memories': {
        // "What is actually in my Memory right now, and what did the secondary
        // LLM write there?" — the managed rows for this project plus the global
        // ones, newest first. Statements appear only with `ui.show_content`.
        const memory = getMemoryContextSafely(context);
        if (!memory) return { error: 'AiderDesk Memory is disabled - nothing to list' };
        const showContent = this.config.ui.show_content;
        const managed = (await listAll(memory)).filter((entry) => entry?.id && isManaged(entry));
        const scoped = managed.filter((entry) => {
          const pid = entry.projectId ?? '';
          return pid === projectDir || pid === '';
        });
        const requested =
          typeof opts.limit === 'number' && opts.limit > 0 ? Math.round(opts.limit) : MEMORIES_VIEW_LIMIT;
        const limit = Math.min(200, Math.max(1, requested));
        const items = scoped
          .slice()
          .sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
          .slice(-limit)
          .reverse()
          .map((entry) => {
            const decoded = decodeMemory(entry.content);
            return {
              id: entry.id,
              scope: scopeOf(entry),
              category: decoded?.meta.category ?? 'unknown',
              importance: importanceOf(entry),
              createdAt: decoded?.meta.createdAt,
              updatedAt: decoded?.meta.updatedAt,
              statement: showContent ? sanitizeStatements([statementOf(entry)])[0] : undefined,
            };
          });
        return { items, total: scoped.length, showContent };
      }

      default:
        return undefined;
    }
  }

  /**
   * The statements a viewer entry may carry. Off unless `ui.show_content` is
   * on, and always sanitized: secrets redacted, one line per statement, the
   * list capped. `undefined` keeps the feed to counts only.
   */
  private contentFor(texts: readonly string[] | undefined): string[] | undefined {
    if (!this.config.ui.show_content || !texts || !texts.length) return undefined;
    const content = sanitizeStatements(texts);
    return content.length ? content : undefined;
  }

  /** The entries matching the viewer's current scope/level filters. */
  private filteredActivity(
    opts: Record<string, unknown>,
    taskId: string | undefined,
    projectDir: string,
  ): ActivityEntry[] {
    const scope =
      opts.scope === 'task' || opts.scope === 'project' || opts.scope === 'all' ? opts.scope : 'all';
    const minLevel = isActivityLevel(opts.minLevel) ? opts.minLevel : 'debug';
    const limit = typeof opts.limit === 'number' && opts.limit > 0 ? opts.limit : this.config.ui.activity_limit;
    return activity.list({ scope, minLevel, taskId, projectDir, limit });
  }

  /**
   * Header snapshot the badge and the settings panel render. Every field is a
   * count, an id or a label; the endpoint is credential-stripped.
   */
  private async activityStatus(context: ExtensionContext): Promise<EngramActivityStatus> {
    const agentId = await this.agentIdFor(context);
    const cfg = resolveConfig(this.config, agentId);
    const projectDir = context.getProjectDir();
    const stats = projectStats(this.state, projectDir);
    const all = activity.list({ scope: 'all' });
    const last = activity.last();
    return {
      version: EngramMemoryExtension.metadata.version,
      enabled: cfg.enabled,
      memoryAvailable: Boolean(getMemoryContextSafely(context)),
      busy: this.inFlight.size > 0,
      agentId: agentId ?? null,
      perAgent: hasAgentOverride(this.config, agentId),
      llm: {
        transport: cfg.secondary_llm.transport,
        model: cfg.secondary_llm.model,
        modelId: cfg.secondary_llm.model_id,
        endpoint: safeEndpoint(cfg.secondary_llm.base_url),
      },
      logging: { enabled: cfg.logging.enabled, level: cfg.logging.level },
      gating: {
        cursorTasks: this.cursors.size(),
        paused: this.breakerPaused(projectDir),
        consolidationStalls: stallRuns(projectStats(this.state, projectDir)),
      },
      activity: {
        show: cfg.ui.show_activity,
        limit: activity.getLimit(),
        size: all.length,
        warnings: all.filter((entry) => entry.level === 'warn').length,
        errors: all.filter((entry) => entry.level === 'error').length,
      },
      ui: { showContent: cfg.ui.show_content },
      project: {
        dir: projectDir,
        counts: {
          extractions: stats.extractions,
          stored: stats.stored,
          updated: stats.updated,
          deleted: stats.deleted,
          duplicates: stats.duplicates,
          obsolete: stats.obsolete,
        },
      },
      totals: {
        llmCalls: this.state.totals.llmCalls,
        llmFailures: this.state.totals.llmFailures,
        stored: this.state.totals.stored,
        updated: this.state.totals.updated,
        deleted: this.state.totals.deleted,
      },
      last: last ? { ts: last.ts, level: last.level, kind: last.kind, message: last.message } : null,
    };
  }

  /**
   * Refresh the agent-profile cache from the authoritative API. No-op outside
   * a project context (getProjectContext throws there) - the previous cache
   * stays and the settings dialog falls back to the disk scan.
   */
  private cacheAgentProfiles(context: ExtensionContext): void {
    try {
      const profiles = (context.getProjectContext().getAgentProfiles() ?? [])
        .filter((profile) => profile?.id)
        .map((profile) => ({
          id: profile.id,
          name: profile.name,
          provider: profile.provider,
          model: profile.model,
          isSubagent: Boolean(profile.isSubagent),
        }));
      if (profiles.length) this.agentMetaCache = profiles;
    } catch {
      /* outside a project scope - nothing to refresh */
    }
  }

  async getConfigData(context: ExtensionContext): Promise<unknown> {
    const config = loadConfig(this.configPath);
    // `_agents` is UI-only: the list of AiderDesk agent profiles, so the
    // settings panel can render one tab per agent. mergeConfig() ignores
    // unknown keys, so it never reaches config.json.
    //
    // The settings dialog runs in a GLOBAL context, where getProjectContext()
    // throws. Fall back to the runtime cache, then to reading the agent
    // profile files straight off disk (global + open projects).
    let profiles: AgentMeta[] = [];
    try {
      profiles = (context.getProjectContext().getAgentProfiles() ?? [])
        .filter((profile) => profile?.id)
        .map((profile) => ({
          id: profile.id,
          name: profile.name,
          provider: profile.provider,
          model: profile.model,
          isSubagent: Boolean(profile.isSubagent),
        }));
    } catch (error) {
      logger.debug(`agent profiles unavailable via API: ${error instanceof Error ? error.message : String(error)}`);
    }
    if (profiles.length) {
      this.agentMetaCache = profiles;
    } else {
      let openProjectDirs: string[] = [];
      try {
        openProjectDirs = context.getOpenProjectDirs();
      } catch {
        /* store unavailable - global agents dir only */
      }
      // Disk scan first, runtime cache wins on duplicate ids (it reflects the
      // live manager, including extension-provided profiles that have no file).
      const merged = new Map(listAgentProfiles({ openProjectDirs }).map((p) => [p.id, p]));
      for (const profile of this.agentMetaCache) merged.set(profile.id, profile);
      profiles = Array.from(merged.values());
    }
    // Read-only activity snapshot for the settings panel (newest entries +
    // status header). Like `_agents`, these are UI-only keys that mergeConfig()
    // ignores, so they never reach config.json.
    const snapshot = activity.list({ scope: 'all', limit: 100 });
    const status = await this.activityStatus(context);
    const extra = { _activity: snapshot, _activityStatus: status };
    return profiles.length ? { ...config, _agents: profiles, ...extra } : { ...config, ...extra };
  }

  async saveConfigData(configData: unknown, _context: ExtensionContext): Promise<unknown> {
    const merged = mergeConfig(configData);
    saveConfig(this.configPath, merged);
    this.config = merged;
    logger.setConfig(merged.logging);
    const overrides = Object.keys(merged.agents);
    logger.info(
      `configuration saved (secondary LLM: ${merged.secondary_llm.model} @ ${merged.secondary_llm.base_url})`,
    );
    logger.info(
      overrides.length
        ? `per-agent overrides: ${overrides.length} (${overrides.join(', ')})`
        : 'per-agent overrides: none (global config applies to every agent)',
    );
    // mergeConfig() strips `_agents` (the UI-only agent-profile list injected
    // by getConfigData()). If the settings framework re-renders from this
    // return value, the agent tabs would vanish after the first save - pass
    // the list back through when the caller sent one.
    const agents =
      typeof configData === 'object' && configData !== null && '_agents' in configData
        ? (configData as { _agents: unknown })._agents
        : undefined;
    // Same round-trip rule for the read-only activity snapshot: if the caller
    // sent one, hand it back so the panel does not lose its Activity section.
    const source = typeof configData === 'object' && configData !== null ? (configData as Record<string, unknown>) : {};
    const activityKeys: Record<string, unknown> = {};
    if ('_activity' in source) activityKeys._activity = source._activity;
    if ('_activityStatus' in source) activityKeys._activityStatus = source._activityStatus;
    return agents !== undefined ? { ...merged, _agents: agents, ...activityKeys } : { ...merged, ...activityKeys };
  }
}
