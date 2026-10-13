/**
 * Single-instance arbitration.
 *
 * Why this exists: AiderDesk loads the global extensions
 * (`${AIDER_DESK_HOME_DIR}/extensions/`) AND the project extensions
 * (`<project>/${AIDER_DESK_DIR}/extensions/`) into the SAME process. The
 * registry keys them by file path and never dedupes by name, so an extension
 * installed in both places runs as two live instances. For Engram that was
 * visible as two "Engram" badges in the task status bar, and invisibly as the
 * same commands twice, a second settings entry, and every conversation
 * extracted twice (one round per copy).
 *
 * The rule, as a product decision rather than a heuristic:
 *
 *   the project-local install wins for its own project; the global install is
 *   deactivated for that project and stays active for every other project.
 *
 * How two copies can agree: they are loaded in one Node process, so they share
 * `globalThis`. Each copy seats itself on a shared bus, and asks the bus, per
 * project, whether it owns that project. The loser registers no badge, no
 * commands and handles no events.
 *
 * The loser still answers the UI/settings calls, and this is the part that is
 * easy to get wrong: AiderDesk resolves `getUIExtensionData`,
 * `executeUIExtensionAction` and the settings panel by extension id, and both
 * copies have the SAME id - the host derives it from the directory name, so
 * `~/.aider-desk/extensions/engram/index.ts` and
 * `<project>/.aider-desk/extensions/engram/index.ts` are both "engram". It
 * returns the FIRST copy that answers, with no fall-through on `undefined`. So
 * whichever copy the host happens to pick must be able to answer FOR the owner:
 * `ownerApi()` hands over the owner's live methods and the request is served by
 * the copy that owns the project.
 *
 * Nothing is persisted: the bus lives exactly as long as the instances it
 * arbitrates, and a copy releases its seat when AiderDesk unloads it (which is
 * also what hands a project back to the global install when the local one is
 * removed).
 */

import { homedir } from 'node:os';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';

import type { ExtensionContext } from '@aiderdesk/extensions';

/** Key of the shared bus on `globalThis`. */
const BUS_KEY = '__engramInstanceBus_v1';

/** The AiderDesk home directory: `$AIDER_DESK_HOME_DIR`, else `~/.aider-desk`. */
export function homeDir(): string {
  const env = process.env.AIDER_DESK_HOME_DIR?.trim();
  if (env) return resolve(env);
  return resolve(join(homedir(), '.aider-desk'));
}

/** The per-project AiderDesk directory name: `$AIDER_DESK_DIR`, else `.aider-desk`. */
export function appDirName(): string {
  const env = process.env.AIDER_DESK_DIR?.trim();
  return env || '.aider-desk';
}

function isInside(child: string, parent: string): boolean {
  const rel = relative(parent, child);
  return !!rel && !rel.startsWith('..') && !isAbsolute(rel);
}

/**
 * Which project an install directory belongs to.
 *
 * `null` means "global": the copy sits under the home directory's own
 * extensions folder, so it serves every project that has no local install.
 * A project-local copy returns its project directory, so it can claim that
 * project away from the global copy.
 *
 * An unrecognised layout is treated as global on purpose: a copy whose scope
 * cannot be determined must never silence itself into a project it was asked to
 * serve.
 */
export function scopeOfInstall(installDir: string): string | null {
  if (!installDir) return null;
  const dir = resolve(installDir);
  if (isInside(dir, join(homeDir(), 'extensions'))) return null;

  const app = appDirName();
  const segments = dir.split(sep);
  // `<project>/<appDir>/extensions/<name>` - the project is everything before
  // the AiderDesk directory. Index 0 is the leading separator on POSIX.
  for (let i = 1; i + 2 < segments.length; i++) {
    if (segments[i] === app && segments[i + 1] === 'extensions') {
      const prefix = segments.slice(0, i).join(sep);
      return prefix ? resolve(prefix) : null;
    }
  }
  return null;
}

/**
 * The live surface the host resolves by extension id, and which therefore has
 * to be reachable from whichever copy the host picked.
 */
export interface EngramInstanceApi {
  uiData: (componentId: string, context: ExtensionContext) => unknown;
  uiAction: (componentId: string, action: string, args: unknown[], context: ExtensionContext) => unknown;
  configComponent: (context: ExtensionContext) => string | undefined;
  configData: (context: ExtensionContext) => unknown;
  saveConfigData: (configData: unknown, context: ExtensionContext) => unknown;
}

interface BusRecord {
  readonly installDir: string;
  readonly projectDir: string | null;
  readonly seq: number;
  readonly api: EngramInstanceApi;
}

interface InstanceBus {
  records: Map<string, BusRecord>;
  seq: number;
}

function bus(): InstanceBus {
  const g = globalThis as Record<string, unknown>;
  const existing = g[BUS_KEY] as InstanceBus | undefined;
  if (existing && typeof existing === 'object' && existing.records instanceof Map) return existing;
  const created: InstanceBus = { records: new Map(), seq: 0 };
  g[BUS_KEY] = created;
  return created;
}

/** Deterministic owner for a project: the local install, else the global one. */
function pickOwner(records: BusRecord[], projectDir?: string): BusRecord | null {
  if (!records.length) return null;
  const project = projectDir && projectDir.trim() ? resolve(projectDir) : '';
  const scoped = project ? records.filter((r) => r.projectDir && resolve(r.projectDir) === project) : [];
  const pool = scoped.length ? scoped : records.filter((r) => !r.projectDir);
  if (!pool.length) return null;
  // Lexicographic install dir, then seat order: every copy computes the same
  // winner from the same bus, whatever order AiderDesk loaded them in.
  return pool.sort((a, b) => (a.installDir === b.installDir ? a.seq - b.seq : a.installDir.localeCompare(b.installDir)))[0];
}

/**
 * One copy's seat on the shared bus. Constructed with the copy, activated in
 * `onLoad`, released in `onUnload`.
 */
export class InstanceSeat {
  readonly installDir: string;
  /** The project this copy is installed in, `null` for a global install. */
  readonly projectDir: string | null;

  private readonly api: EngramInstanceApi;
  private record: BusRecord | null = null;
  private readonly idleNotices = new Set<string>();

  constructor(installDir: string, api: EngramInstanceApi) {
    this.installDir = resolve(installDir || '.');
    this.projectDir = scopeOfInstall(this.installDir);
    this.api = api;
  }

  /** Claim the seat for this install path (a reload replaces the old claim). */
  activate(): void {
    const b = bus();
    this.record = { installDir: this.installDir, projectDir: this.projectDir, seq: ++b.seq, api: this.api };
    b.records.set(this.installDir, this.record);
  }

  /** Give up the seat. Only ever removes OUR record, never a newer one. */
  release(): void {
    const b = bus();
    if (this.record && b.records.get(this.installDir) === this.record) b.records.delete(this.installDir);
    this.record = null;
  }

  /** True when this copy must serve `projectDir` itself. */
  isActiveFor(projectDir?: string): boolean {
    const owner = pickOwner(Array.from(bus().records.values()), projectDir);
    // No claimant: a lone install, or a copy whose scope is another project.
    // Serving is the safe answer - silence is only ever the documented outcome
    // of a local install beating a global one in that same project.
    if (!owner) return true;
    return owner.installDir === this.installDir;
  }

  /** True when this copy must stay out of `projectDir`. */
  isIdleIn(projectDir?: string): boolean {
    return !this.isActiveFor(projectDir);
  }

  /**
   * True exactly once per project: the gate that lets the loser say WHY it is
   * silent, without repeating the line on every status-bar refresh.
   */
  firstIdleNotice(projectDir?: string): boolean {
    const key = projectDir && projectDir.trim() ? resolve(projectDir) : '(no project)';
    if (this.idleNotices.has(key)) return false;
    this.idleNotices.add(key);
    return true;
  }

  /**
   * The owner's live methods, when the owner is another copy; `null` when this
   * copy serves the request itself.
   */
  ownerApi(projectDir?: string): EngramInstanceApi | null {
    const owner = pickOwner(Array.from(bus().records.values()), projectDir);
    if (!owner || owner.installDir === this.installDir) return null;
    return owner.api;
  }
}
