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

import { loadConfig, saveConfig, mergeConfig, type EngramConfig } from './src/config';
import { logger } from './src/logger';
import { loadState, saveState, projectStats, type EngramState } from './src/state';
import { runExtraction } from './src/extraction';
import { runConsolidation } from './src/consolidation';
import { retrieveForPrompt, wrapBlock } from './src/retrieval';
import {
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

export default class EngramMemoryExtension implements Extension {
  static metadata = {
    name: 'Engram Memory',
    version: '1.0.0',
    description:
      'Automatic long-term memory: extracts durable facts from conversations with a secondary local OpenAI-compatible LLM, dedupes/updates/resolves conflicts in AiderDesk Memory, consolidates periodically, and injects only relevant memories.',
    author: 'local',
    capabilities: ['commands', 'settings', 'events'],
  };

  private config: EngramConfig = loadConfig(CONFIG_PATH);
  private state: EngramState = loadState(STATE_PATH);
  private readonly abortController = new AbortController();
  /** Serialized work queue per project, so memory writes never interleave. */
  private readonly queues = new Map<string, Promise<void>>();
  private readonly inFlight = new Set<string>();
  /** Last user prompt per task, so retrieval has a query on the reminder hook. */
  private readonly lastPrompt = new Map<string, string>();

  // ---------------------------------------------------------------- lifecycle

  async onLoad(context: ExtensionContext): Promise<void> {
    logger.bind(context);
    logger.setConfig(this.config.logging);

    logger.info(
      `loaded (secondary LLM: ${this.config.secondary_llm.model} @ ${this.config.secondary_llm.base_url})`,
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
    saveState(STATE_PATH, this.state);
  }

  // ---------------------------------------------------------------- retrieval

  /**
   * Remember the current prompt so retrieval can run on the reminder hook,
   * which is the point where content actually reaches the main model.
   */
  async onPromptStarted(event: PromptStartedEvent, context: ExtensionContext): Promise<void> {
    if (!this.config.enabled || !this.config.retrieval.enabled) return;
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
    if (!this.config.enabled || !this.config.retrieval.enabled) return;

    const taskContext = context.getTaskContext();
    const taskId = taskContext?.data?.id;
    const prompt = (taskId && this.lastPrompt.get(taskId)) || '';
    if (!prompt) return;

    try {
      const { block } = await retrieveForPrompt(context, context.getProjectDir(), prompt, this.config);
      if (!block) return;

      return {
        remindersContent: `${event.remindersContent}\n\n${wrapBlock(block)}`.trim(),
      };
    } catch (error) {
      logger.warn(`retrieval failed: ${error instanceof Error ? error.message : String(error)}`);
      return;
    }
  }

  // -------------------------------------------------------------- extraction

  async onAgentFinished(event: AgentFinishedEvent, context: ExtensionContext): Promise<void> {
    if (!this.config.enabled || !this.config.extraction.enabled) return;
    if (this.config.extraction.trigger !== 'agent_end') return;
    if (event.aborted) return;

    this.queueExtraction(context, event.contextMessages);
  }

  async onPromptFinished(event: PromptFinishedEvent, context: ExtensionContext): Promise<void> {
    if (!this.config.enabled || !this.config.extraction.enabled) return;
    if (this.config.extraction.trigger !== 'prompt_end') return;

    const taskContext = context.getTaskContext();
    if (!taskContext) return;

    void (async () => {
      try {
        const messages = await taskContext.getContextMessages();
        this.queueExtraction(context, messages);
      } catch (error) {
        logger.warn(`cannot read conversation: ${error instanceof Error ? error.message : String(error)}`);
      }
    })();
  }

  async onTaskClosed(event: TaskClosedEvent, context: ExtensionContext): Promise<void> {
    if (!this.config.enabled) return;

    if (this.config.extraction.enabled && this.config.extraction.trigger === 'task_end') {
      const taskContext = context.getTaskContext();
      if (taskContext) {
        void (async () => {
          try {
            const messages = await taskContext.getContextMessages();
            this.queueExtraction(context, messages);
          } catch {
            /* task already torn down */
          }
        })();
      }
    }

    // Consolidation is driven by extraction rounds, not by wall-clock, so the
    // interval means the same thing for every trigger mode.
    this.maybeConsolidate(context);
  }

  // ------------------------------------------------------------- scheduling

  /**
   * Fire-and-forget. The handler returns immediately, so the main agent is
   * never blocked by the (potentially slow) secondary LLM. Work is serialized
   * per project so two runs cannot race on the same memories.
   */
  private queueExtraction(context: ExtensionContext, messages: ContextMessage[]): void {
    const projectDir = context.getProjectDir();
    if (!messages?.length) return;
    if (this.inFlight.has(projectDir)) {
      logger.debug('extraction already running for this project - queued');
    }
    this.inFlight.add(projectDir);

    const taskId = context.getTaskContext()?.data?.id ?? '';

    this.enqueue(projectDir, async () => {
      try {
        const report = await runExtraction({
          context,
          messages,
          projectDir,
          taskId,
          config: this.config,
          state: this.state,
          statePath: STATE_PATH,
          signal: this.abortController.signal,
          taskContext: context.getTaskContext(),
        });
        if (report.failure) logger.debug(`extraction report: ${report.failure}`);
        this.countRound(projectDir);
      } catch (error) {
        logger.warn(`extraction error: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        this.inFlight.delete(projectDir);
      }
    });
  }

  private maybeConsolidate(context: ExtensionContext): void {
    if (!this.config.enabled || !this.config.consolidation.enabled) return;
    const projectDir = context.getProjectDir();
    const stats = projectStats(this.state, projectDir);
    if (stats.tasksSinceConsolidation < Math.max(2, this.config.consolidation.interval_tasks)) return;

    this.enqueue(projectDir, async () => {
      try {
        await runConsolidation({
          context,
          projectDir,
          config: this.config,
          state: this.state,
          statePath: STATE_PATH,
          signal: this.abortController.signal,
          taskContext: context.getTaskContext(),
        });
      } catch (error) {
        logger.warn(`consolidation error: ${error instanceof Error ? error.message : String(error)}`);
      }
    });
  }

  private countRound(projectDir: string): void {
    const stats = projectStats(this.state, projectDir);
    stats.tasksSinceConsolidation += 1;
    saveState(STATE_PATH, this.state);
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

  // ---------------------------------------------------------------- commands

  getCommands(context: ExtensionContext): CommandDefinition[] {
    return [
      {
        name: 'memory:extract',
        description: 'Engram: extract memories from this conversation now (secondary LLM, runs in background)',
        execute: async (_args, ctx) => {
          const taskContext = ctx.getTaskContext();
          if (!taskContext) {
            ctx.log('[Memory] no task context - open a task first', 'warn');
            return;
          }
          const messages = await taskContext.getContextMessages();
          taskContext.addLogMessage('info', '[Memory] extraction started (background)');
          this.queueExtraction(ctx, messages);
        },
      },
      {
        name: 'memory:consolidate',
        description: 'Engram: consolidate project + global memories (dedupe, merge, resolve conflicts)',
        arguments: [{ description: 'force - run even when fewer than 2 memories' }],
        execute: async (args, ctx) => {
          const projectDir = ctx.getProjectDir();
          ctx.getTaskContext()?.addLogMessage('info', '[Memory] consolidation started (background)');
          this.enqueue(projectDir, async () => {
            const report = await runConsolidation({
              context: ctx,
              projectDir,
              config: {
                ...this.config,
                consolidation: {
                  ...this.config.consolidation,
                  safe_mode: args.includes('force') ? false : this.config.consolidation.safe_mode,
                },
              },
              state: this.state,
              statePath: STATE_PATH,
              taskContext: ctx.getTaskContext(),
            });
            const line = report.failure
              ? `consolidation failed: ${report.failure}`
              : `consolidated ${report.scanned} memories -> merged ${report.merged}, updated ${report.updated}, deleted ${report.deleted}, kept ${report.kept}`;
            ctx.getTaskContext()?.addLogMessage('info', `[Memory] ${line}`);
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
            ctx.getTaskContext()?.addLogMessage('warning', '[Memory] AiderDesk Memory is disabled/unavailable');
            return;
          }
          const all = await listAll(memory);
          const managed = all.filter((entry) => entry?.id && isManaged(entry));
          const mine = managed.filter((entry) => (entry.projectId ?? '') === projectDir);
          const global = managed.filter((entry) => (entry.projectId ?? '') === '');
          const native = all.filter((entry) => entry?.id && !isManaged(entry));
          const stats = projectStats(this.state, projectDir);

          const lines = [
            `[Memory] AiderDesk entries: ${all.length} (Engram-managed ${managed.length}, native ${native.length})`,
            `[Memory] this project: ${mine.length} | global: ${global.length}`,
            `[Memory] project counters: extractions ${stats.extractions}, stored ${stats.stored}, updated ${stats.updated}, duplicates ${stats.duplicates}, obsolete ${stats.obsolete}`,
            `[Memory] totals: LLM calls ${this.state.totals.llmCalls} (failures ${this.state.totals.llmFailures}), stored ${this.state.totals.stored}, updated ${this.state.totals.updated}, deleted ${this.state.totals.deleted}`,
            `[Memory] consolidation: ${stats.tasksSinceConsolidation}/${this.config.consolidation.interval_tasks} rounds, safe_mode=${this.config.consolidation.safe_mode}`,
            `[Memory] secondary LLM: ${this.config.secondary_llm.model} @ ${this.config.secondary_llm.base_url}`,
          ];
          for (const line of lines) ctx.getTaskContext()?.addLogMessage('info', line);

          const result = await probe(this.config.secondary_llm);
          ctx
            .getTaskContext()
            ?.addLogMessage(
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
            ctx.getTaskContext()?.addLogMessage('warning', '[Memory] usage: /memory:forget <text>');
            return;
          }
          const memory = getMemoryContextSafely(ctx);
          if (!memory) {
            ctx.getTaskContext()?.addLogMessage('warning', '[Memory] AiderDesk Memory is disabled/unavailable');
            return;
          }
          const candidates = await retrieveScoped(memory, ctx.getProjectDir(), query, 3, true);
          if (!candidates.length) {
            ctx.getTaskContext()?.addLogMessage('info', '[Memory] no matching memory found');
            return;
          }
          const target = candidates[0];
          const decoded = decodeMemory(target.content);
          const ok = await remove(memory, target.id);
          ctx.getTaskContext()?.addLogMessage(
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
            ctx
              .getTaskContext()
              ?.addLogMessage('warning', '[Memory] refusing to clear without the explicit "confirm" argument');
            return;
          }
          const projectDir = ctx.getProjectDir();
          const memory = getMemoryContextSafely(ctx);
          if (!memory) {
            ctx.getTaskContext()?.addLogMessage('warning', '[Memory] AiderDesk Memory is disabled/unavailable');
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
          saveState(STATE_PATH, this.state);
          ctx.getTaskContext()?.addLogMessage('info', `[Memory] cleared ${deleted} project memories`);
        },
      },
    ];
  }

  // ---------------------------------------------------------------- settings

  getConfigComponent(_context: ExtensionContext): string | undefined {
    return readConfigComponent();
  }

  async getConfigData(_context: ExtensionContext): Promise<unknown> {
    return loadConfig(CONFIG_PATH);
  }

  async saveConfigData(configData: unknown, _context: ExtensionContext): Promise<unknown> {
    const merged = mergeConfig(configData);
    saveConfig(CONFIG_PATH, merged);
    this.config = merged;
    logger.setConfig(merged.logging);
    logger.info(`configuration saved (secondary LLM: ${merged.secondary_llm.model} @ ${merged.secondary_llm.base_url})`);
    return merged;
  }
}
