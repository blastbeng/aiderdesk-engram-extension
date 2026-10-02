/**
 * Minimal stand-ins for AiderDesk's ExtensionContext / TaskContext, used only
 * by the offline test harness (tests/run.ts).
 *
 * Only the members the Engram code paths actually touch are implemented:
 *   ExtensionContext: getMemoryContext, getProjectDir, log, getTaskContext,
 *                     addDisposable, getOpenProjectDirs
 *   TaskContext:      data.id, generateText (used by the 'aiderdesk' transport)
 * Everything else would throw - the harness never calls it.
 *
 * The casts to ExtensionContext / TaskContext are deliberate: implementing the
 * full interfaces would mean stubbing ~40 unrelated Electron/UI methods that
 * the memory pipeline never reaches.
 */
import type { ExtensionContext, MemoryContext, TaskContext, TaskData } from '@aiderdesk/extensions';

export interface LogSink {
  lines: { level: string; message: string }[];
}

export function createLogSink(): LogSink {
  return { lines: [] };
}

export interface MockContextOptions {
  taskContext?: TaskContext | null;
  sink?: LogSink;
}

export function mockExtensionContext(
  memory: MemoryContext | null,
  projectDir: string,
  options: MockContextOptions = {},
): ExtensionContext {
  const sink = options.sink;
  const ctx = {
    addDisposable: (_setup: () => unknown): void => {},
    log: (message: string, type: 'info' | 'error' | 'warn' | 'debug' = 'info'): void => {
      sink?.lines.push({ level: type, message });
    },
    getProjectDir: (): string => projectDir,
    getOpenProjectDirs: (): string[] => [projectDir],
    getTaskContext: (): TaskContext | null => options.taskContext ?? null,
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
}

export function mockTaskContext(options: MockTaskOptions = {}): TaskContext {
  const tc = {
    data: { id: options.id ?? 'task-mock', name: 'Harness task' } as unknown as TaskData,
    generateText: options.generateText,
  };
  return tc as unknown as TaskContext;
}
