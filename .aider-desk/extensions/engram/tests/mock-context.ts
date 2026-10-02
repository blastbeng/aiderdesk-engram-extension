/**
 * Minimal stand-ins for AiderDesk's ExtensionContext / TaskContext, used only
 * by the offline test harness (tests/run.ts).
 *
 * Only the members the Engram code paths actually touch are implemented:
 *   ExtensionContext: getMemoryContext, getProjectDir, log, getTaskContext,
 *                     addDisposable, getOpenProjectDirs, getProjectContext
 *   ProjectContext:   getAgentProfiles (per-agent config: the settings panel's
 *                     list of AiderDesk agents)
 *   TaskContext:      data.id, data.agentProfileId, getTaskAgentProfile
 *                     (per-agent config: which agent a task runs as),
 *                     getContextMessages, addLogMessage,
 *                     generateText (used by the 'aiderdesk' transport)
 * Everything else would throw - the harness never calls it.
 *
 * The casts to ExtensionContext / TaskContext are deliberate: implementing the
 * full interfaces would mean stubbing ~40 unrelated Electron/UI methods that
 * the memory pipeline never reaches.
 */
import type {
  AgentProfile,
  ContextMessage,
  ExtensionContext,
  MemoryContext,
  TaskContext,
  TaskData,
} from '@aiderdesk/extensions';

export interface LogSink {
  lines: { level: string; message: string }[];
}

export function createLogSink(): LogSink {
  return { lines: [] };
}

/**
 * The subset of AgentProfile the Engram code reads: the profile id (the key of
 * the per-agent overrides) plus the fields the settings panel displays.
 */
export interface MockAgentProfile {
  id: string;
  name?: string;
  provider?: string;
  model?: string;
  isSubagent?: boolean;
}

export interface MockContextOptions {
  taskContext?: TaskContext | null;
  sink?: LogSink;
  /**
   * Agents this project exposes. When set, getProjectContext() answers with
   * them; when omitted it throws, exactly like the real API outside a project
   * context - which is the path index.ts's agent-list lookup must survive.
   */
  agentProfiles?: MockAgentProfile[];
}

export function mockExtensionContext(
  memory: MemoryContext | null,
  projectDir: string,
  options: MockContextOptions = {},
): ExtensionContext {
  const sink = options.sink;
  const profiles = (options.agentProfiles ?? []) as unknown as AgentProfile[];
  const ctx = {
    addDisposable: (_setup: () => unknown): void => {},
    log: (message: string, type: 'info' | 'error' | 'warn' | 'debug' = 'info'): void => {
      sink?.lines.push({ level: type, message });
    },
    getProjectDir: (): string => projectDir,
    getOpenProjectDirs: (): string[] => [projectDir],
    getTaskContext: (): TaskContext | null => options.taskContext ?? null,
    getProjectContext: () => {
      if (!options.agentProfiles) throw new Error('ProjectContext is not available (mock)');
      return { getAgentProfiles: (): AgentProfile[] => profiles };
    },
    getMemoryContext: (): MemoryContext => {
      // Mirrors the real MemoryManager: throws when the memory system is off.
      if (!memory) throw new Error('MemoryManager is not available (mock)');
      return memory;
    },
  };
  return ctx as unknown as ExtensionContext;
}

export interface MockTaskOptions {
  id?: string;
  generateText?: (modelId: string, systemPrompt: string, prompt: string) => Promise<string | null>;
  /** Written to TaskData.agentProfileId - the raw task field index.ts falls back to. */
  agentProfileId?: string | null;
  /** What getTaskAgentProfile() resolves to - the documented per-task resolver. */
  agentProfile?: MockAgentProfile | null;
  /** Returned by getContextMessages() (prompt_end / task_end / command paths). */
  contextMessages?: ContextMessage[];
  /** Command output (addLogMessage) lands here, so tests can assert on it. */
  logMessages?: string[];
}

export function mockTaskContext(options: MockTaskOptions = {}): TaskContext {
  const tc = {
    data: {
      id: options.id ?? 'task-mock',
      name: 'Harness task',
      agentProfileId: options.agentProfileId ?? undefined,
    } as unknown as TaskData,
    generateText: options.generateText,
    getTaskAgentProfile: async (): Promise<AgentProfile | null> =>
      options.agentProfile ? (options.agentProfile as unknown as AgentProfile) : null,
    getContextMessages: async (): Promise<ContextMessage[]> => options.contextMessages ?? [],
    addLogMessage: (level: string, message: string): void => {
      options.logMessages?.push(`[${level}] ${message}`);
    },
  };
  return tc as unknown as TaskContext;
}
