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
  AgentFinishedEvent,
  PromptStartedEvent,
  PromptFinishedEvent,
  TaskClosedEvent,
  ImportantRemindersEvent,
  ContextMessage,
} from '@aiderdesk/extensions';

import { loadConfig, saveConfig, mergeConfig, resolveConfig, hasAgentOverride, type EngramConfig } from './src/config';
import { listAgentProfiles, type AgentMeta } from './src/agents';
import { logger } from './src/logger';
import { loadState, saveState, projectStats, type EngramState } from './src/state';
import { runExtraction } from './src/extraction';
import { runConsolidation } from './src/consolidation';
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
} from './src/store';
import { decodeMemory } from './src/memory-format';
import { probe } from './src/llm';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CONFIG_PATH = join(__dirname, 'config.json');
const STATE_PATH = join(__dirname, 'state.json');
const CONFIG_JSX_PATH = join(__dirname, 'ConfigComponent.jsx');

function readConfigComponent(): string | undefined {
  try {
    if (!existsSync(CONFIG_JSX_PATH)) return undefined;
    return readFileSync(CONFIG_JSX_PATH, 'utf-8');
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
 */
function say(context: ExtensionContext, level: 'info' | 'warn' | 'warning' | 'error', message: string): void {
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

export default class EngramMemoryExtension implements Extension {
  static metadata = {
    name: 'Engram Memory',
    version: '1.2.0',
    description:
      'Automatic long-term memory: extracts durable facts from conversations with a secondary local OpenAI-compatible LLM, dedupes/updates/resolves conflicts in AiderDesk Memory, consolidates periodically, and injects only relevant memories.',
    author: 'local',
    capabilities: ['commands', 'settings', 'events'],
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
   * Agent profiles last seen through the authoritative API (project-scoped
   * contexts). The settings dialog runs without a project, so this cache plus
   * a disk scan (src/agents.ts) fill the gap there.
   */
  private agentMetaCache: AgentMeta[] = [];

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

    const overrides = Object.keys(this.config.agents);
    logger.info(
      `loaded (secondary LLM: ${this.config.secondary_llm.model} @ ${this.config.secondary_llm.base_url}, ` +
        `per-agent overrides: ${overrides.length ? overrides.join(', ') : 'none'})`,
    );

    if (!this.config.enabled) {
      logger.info('extension is disabled by configuration - idle');
      return;
    }

    const memory = getMemoryContextSafely(context);
    if (!memory) {
      logger.warn(
        'AiderDesk Memory is disabled or unavailable - enable it in Settings > Memory, otherwise Engram cannot store anything',
      );
    }

    // Fire-and-forget connectivity probe so the log tells you immediately
    // whether the secondary endpoint is reachable. Never blocks startup.
    void probe(this.config.secondary_llm).then((result) => {
      if (result.ok) {
        logger.info(`secondary LLM reachable (${result.durationMs} ms)`);
      } else {
        logger.warn(`secondary LLM not reachable: ${result.kind} - ${result.message}`);
      }
    });

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
    const cfg = resolveConfig(this.config, await this.agentIdFor(context));
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

    try {
      const { block } = await retrieveForPrompt(context, context.getProjectDir(), prompt, cfg);
      if (!block) return;

      // Strip-then-append: the reminder hook can fire more than once per
      // request, and append-only injection would stack a second copy of the
      // same block (observed live). Replacement keeps exactly one.
      return {
        remindersContent: [stripBlock(event.remindersContent), wrapBlock(block)].filter(Boolean).join('\n\n'),
      };
    } catch (error) {
      logger.warn(`retrieval failed: ${error instanceof Error ? error.message : String(error)}`);
      return;
    }
  }

  // -------------------------------------------------------------- extraction

  async onAgentFinished(event: AgentFinishedEvent, context: ExtensionContext): Promise<void> {
    if (event.aborted) return;
    this.cacheAgentProfiles(context);
    const agentId = await this.agentIdFor(context);
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
    const agentId = event.task?.agentProfileId ?? (await this.agentIdFor(context));
    const cfg = resolveConfig(this.config, agentId);
    if (!cfg.enabled) return;

    if (cfg.extraction.enabled && cfg.extraction.trigger === 'task_end') {
      const taskContext = context.getTaskContext();
      if (taskContext) {
        void (async () => {
          try {
            const messages = await taskContext.getContextMessages();
            this.queueExtraction(context, messages, agentId);
          } catch {
            /* task already torn down */
          }
        })();
      }
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
  private queueExtraction(
    context: ExtensionContext,
    messages: ContextMessage[],
    agentId?: string | null,
  ): void {
    const projectDir = context.getProjectDir();
    if (!messages?.length) return;
    if (this.inFlight.has(projectDir)) {
      logger.debug('extraction already running for this project - queued');
    }
    this.inFlight.add(projectDir);

    const taskId = context.getTaskContext()?.data?.id ?? '';

    this.enqueue(projectDir, async () => {
      const cfg = resolveConfig(this.config, agentId);
      logger.setConfig(cfg.logging);
      if (hasAgentOverride(this.config, agentId)) {
        logger.debug(`extraction using agent profile "${agentId}" overrides`);
      }
      try {
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
        if (report.failure) logger.debug(`extraction report: ${report.failure}`);
        this.countRound(projectDir);
      } catch (error) {
        logger.warn(`extraction error: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        this.inFlight.delete(projectDir);
        logger.setConfig(this.config.logging);
      }
    });
  }

  private maybeConsolidate(context: ExtensionContext, agentId?: string | null): void {
    const cfg = resolveConfig(this.config, agentId);
    if (!cfg.enabled || !cfg.consolidation.enabled) return;
    const projectDir = context.getProjectDir();
    const stats = projectStats(this.state, projectDir);
    if (stats.tasksSinceConsolidation < Math.max(2, cfg.consolidation.interval_tasks)) return;

    this.enqueue(projectDir, async () => {
      const effective = resolveConfig(this.config, agentId);
      logger.setConfig(effective.logging);
      try {
        if (!effective.enabled || !effective.consolidation.enabled) {
          logger.debug('consolidation skipped - disabled for this agent');
          return;
        }
        await runConsolidation({
          context,
          projectDir,
          config: effective,
          state: this.state,
          statePath: this.statePath,
          signal: this.abortController.signal,
          taskContext: context.getTaskContext(),
        });
      } catch (error) {
        logger.warn(`consolidation error: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        logger.setConfig(this.config.logging);
      }
    });
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
          const messages = await taskContext.getContextMessages();
          say(ctx, 'info', `[Memory] extraction started (background, agent ${agentId ?? 'global'})`);
          this.queueExtraction(ctx, messages, agentId);
        },
      },
      {
        name: 'memory:consolidate',
        description: 'Engram: consolidate project + global memories (dedupe, merge, resolve conflicts)',
        arguments: [{ description: 'force - run even when fewer than 2 memories' }],
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
              taskContext: ctx.getTaskContext(),
            });
            const line = report.failure
              ? `consolidation failed: ${report.failure}`
              : `consolidated ${report.scanned} memories -> merged ${report.merged}, updated ${report.updated}, deleted ${report.deleted}, kept ${report.kept}`;
            say(ctx, 'info', `[Memory] ${line}`);
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
            `[Memory] project counters: extractions ${stats.extractions}, stored ${stats.stored}, updated ${stats.updated}, duplicates ${stats.duplicates}, obsolete ${stats.obsolete}`,
            `[Memory] totals: LLM calls ${this.state.totals.llmCalls} (failures ${this.state.totals.llmFailures}), stored ${this.state.totals.stored}, updated ${this.state.totals.updated}, deleted ${this.state.totals.deleted}`,
            `[Memory] extraction: enabled=${cfg.extraction.enabled} trigger=${cfg.extraction.trigger} min_importance=${cfg.extraction.min_importance}`,
            `[Memory] retrieval: enabled=${cfg.retrieval.enabled} max_memories=${cfg.retrieval.max_memories} min_importance=${cfg.retrieval.min_importance}`,
            `[Memory] consolidation: ${stats.tasksSinceConsolidation}/${cfg.consolidation.interval_tasks} rounds, safe_mode=${cfg.consolidation.safe_mode}`,
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
          say(ctx, 
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
          saveState(this.statePath, this.state);
          say(ctx, 'info', `[Memory] cleared ${deleted} project memories`);
        },
      },
    ];
  }

  // ---------------------------------------------------------------- settings

  getConfigComponent(_context: ExtensionContext): string | undefined {
    return readConfigComponent();
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
    return profiles.length ? { ...config, _agents: profiles } : config;
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
    return merged;
  }
}
