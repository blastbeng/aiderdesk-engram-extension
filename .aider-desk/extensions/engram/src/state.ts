/**
 * Persisted counters, stored as `state.json` inside the extension directory.
 * Used for the consolidation interval and for `memory:stats`.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

export interface ProjectStats {
  tasksSinceConsolidation: number;
  extractions: number;
  stored: number;
  updated: number;
  deleted: number;
  duplicates: number;
  obsolete: number;
  lastExtractionAt?: number;
  lastConsolidationAt?: number;
}

export interface EngramState {
  projects: Record<string, ProjectStats>;
  totals: {
    llmCalls: number;
    llmFailures: number;
    stored: number;
    updated: number;
    deleted: number;
    duplicatesSkipped: number;
  };
}

const EMPTY: EngramState = {
  projects: {},
  totals: { llmCalls: 0, llmFailures: 0, stored: 0, updated: 0, deleted: 0, duplicatesSkipped: 0 },
};

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function loadState(path: string): EngramState {
  try {
    if (!existsSync(path)) return structuredClone(EMPTY);
    const parsed = JSON.parse(readFileSync(path, 'utf-8'));
    if (!isPlainObject(parsed)) return structuredClone(EMPTY);
    const state = structuredClone(EMPTY);
    if (isPlainObject(parsed.projects)) {
      for (const [key, value] of Object.entries(parsed.projects)) {
        if (!isPlainObject(value)) continue;
        state.projects[key] = {
          tasksSinceConsolidation: Number(value.tasksSinceConsolidation) || 0,
          extractions: Number(value.extractions) || 0,
          stored: Number(value.stored) || 0,
          updated: Number(value.updated) || 0,
          deleted: Number(value.deleted) || 0,
          duplicates: Number(value.duplicates) || 0,
          obsolete: Number(value.obsolete) || 0,
          lastExtractionAt: Number(value.lastExtractionAt) || undefined,
          lastConsolidationAt: Number(value.lastConsolidationAt) || undefined,
        };
      }
    }
    if (isPlainObject(parsed.totals)) {
      const t = parsed.totals as Record<string, unknown>;
      state.totals = {
        llmCalls: Number(t.llmCalls) || 0,
        llmFailures: Number(t.llmFailures) || 0,
        stored: Number(t.stored) || 0,
        updated: Number(t.updated) || 0,
        deleted: Number(t.deleted) || 0,
        duplicatesSkipped: Number(t.duplicatesSkipped) || 0,
      };
    }
    return state;
  } catch {
    return structuredClone(EMPTY);
  }
}

export function saveState(path: string, state: EngramState): void {
  try {
    writeFileSync(path, JSON.stringify(state, null, 2), 'utf-8');
  } catch {
    /* state is best-effort; never break the agent over it */
  }
}

export function projectStats(state: EngramState, projectDir: string): ProjectStats {
  const key = projectDir || '__global__';
  if (!state.projects[key]) {
    state.projects[key] = {
      tasksSinceConsolidation: 0,
      extractions: 0,
      stored: 0,
      updated: 0,
      deleted: 0,
      duplicates: 0,
      obsolete: 0,
    };
  }
  return state.projects[key];
}
