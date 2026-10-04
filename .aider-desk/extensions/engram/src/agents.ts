/**
 * Filesystem discovery of AiderDesk agent profiles.
 *
 * Why this exists: the extension settings dialog always runs in a GLOBAL
 * context (Settings > Extensions passes no project), so
 * `context.getProjectContext()` throws and the authoritative API path
 * (`ProjectContext.getAgentProfiles()`) is unavailable exactly when the
 * settings panel needs the agent list. Agent profiles are plain JSON files on
 * disk, so they can be read directly:
 *
 *   - global:  `${AIDER_DESK_HOME_DIR:-~/.aider-desk}/agents/<dir>/config.json`
 *   - project: `<projectDir>/${AIDER_DESK_DIR:-.aider-desk}/agents/<dir>/config.json`
 *
 * Ordering follows the `order.json` file AiderDesk keeps next to the profiles
 * (profile id -> sort index); entries without an index sort by name, then id.
 *
 * Only a fallback: the API path stays authoritative because it also sees
 * extension-provided profiles that exist only in memory.
 */

import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

export interface AgentMeta {
  id: string;
  name?: string;
  provider?: string;
  model?: string;
  isSubagent?: boolean;
}

export interface AgentProfileSourceOptions {
  /** Overrides the AiderDesk home directory (default: $AIDER_DESK_HOME_DIR or ~/.aider-desk). */
  homeDir?: string;
  /** Overrides the per-project AiderDesk directory name (default: $AIDER_DESK_DIR or .aider-desk). */
  aiderDeskDir?: string;
  /** Currently open projects whose project-level agent profiles should be included. */
  openProjectDirs?: string[];
}

function profileFromRaw(raw: unknown): AgentMeta | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const r = raw as Record<string, unknown>;
  const id = typeof r.id === 'string' ? r.id.trim() : '';
  if (!id) return null;
  const subagent = typeof r.subagent === 'object' && r.subagent !== null ? (r.subagent as Record<string, unknown>) : null;
  return {
    id,
    name: typeof r.name === 'string' && r.name.trim() ? r.name : undefined,
    provider: typeof r.provider === 'string' ? r.provider : undefined,
    model: typeof r.model === 'string' ? r.model : undefined,
    isSubagent: subagent ? subagent.enabled === true : r.isSubagent === true,
  };
}

/** Best-effort read of `order.json` (profile id -> sort index). */
function readOrder(agentsDir: string): Map<string, number> {
  const order = new Map<string, number>();
  try {
    const raw = JSON.parse(readFileSync(join(agentsDir, 'order.json'), 'utf-8'));
    if (typeof raw === 'object' && raw !== null) {
      for (const [id, index] of Object.entries(raw as Record<string, unknown>)) {
        if (typeof index === 'number' && Number.isFinite(index)) order.set(id, index);
      }
    }
  } catch {
    /* no order file or unreadable - fall back to name ordering */
  }
  return order;
}

/** All `<dir>/config.json` profiles in one agents directory. Invalid files are skipped. */
function scanAgentsDir(agentsDir: string): AgentMeta[] {
  if (!agentsDir) return [];
  let entries: string[];
  try {
    entries = readdirSync(agentsDir, { withFileTypes: true }).map((e) => e.name);
  } catch {
    return [];
  }
  const profiles: AgentMeta[] = [];
  for (const entry of entries) {
    const configPath = join(agentsDir, entry, 'config.json');
    if (!existsSync(configPath)) continue;
    try {
      const profile = profileFromRaw(JSON.parse(readFileSync(configPath, 'utf-8')));
      if (profile) profiles.push(profile);
    } catch {
      /* unreadable profile - skip it, the remaining tabs still render */
    }
  }
  return profiles;
}

/**
 * Agent profiles visible on disk: the global agents directory plus the agents
 * directory of every open project. Project-level definitions win over global
 * ones with the same id (matching AiderDesk's own load order), and the result
 * is ordered by `order.json`, then by name, then by id.
 */
export function listAgentProfiles(options: AgentProfileSourceOptions = {}): AgentMeta[] {
  const home = options.homeDir ?? process.env.AIDER_DESK_HOME_DIR?.trim() ?? join(homedir(), '.aider-desk');
  const appDir = options.aiderDeskDir ?? process.env.AIDER_DESK_DIR?.trim() ?? '.aider-desk';
  const projectDirs = (options.openProjectDirs ?? []).filter((d) => typeof d === 'string' && d.trim());

  const agentsDirs = [join(home, 'agents'), ...projectDirs.map((dir) => join(dir, appDir, 'agents'))];

  // Later directories overwrite earlier ones with the same id, so project-level
  // profiles win over global ones.
  const byId = new Map<string, AgentMeta>();
  for (const agentsDir of agentsDirs) {
    for (const profile of scanAgentsDir(agentsDir)) byId.set(profile.id, profile);
  }
  if (byId.size === 0) return [];

  const order = readOrder(agentsDirs[0]);
  const rank = (profile: AgentMeta): number => order.get(profile.id) ?? Number.MAX_SAFE_INTEGER;
  return Array.from(byId.values()).sort((a, b) => {
    const byOrder = rank(a) - rank(b);
    if (byOrder !== 0) return byOrder;
    const byName = (a.name ?? a.id).localeCompare(b.name ?? b.id);
    return byName !== 0 ? byName : a.id.localeCompare(b.id);
  });
}
