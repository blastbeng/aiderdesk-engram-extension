/**
 * Configuration for the Engram memory extension.
 *
 * Config is persisted as `config.json` inside the extension directory and is
 * edited through the extension Settings dialog (ConfigComponent.jsx).
 *
 * The secondary LLM is a plain OpenAI-compatible HTTP endpoint (llama-server,
 * Ollama /v1, LiteLLM, OpenRouter, ...). It is reached directly over HTTP by
 * this extension and is NEVER the AiderDesk main model.
 *
 * Per-agent configuration: `agents` maps an AiderDesk agent profile id
 * (AgentProfile.id, as returned by ExtensionContext.getAgentProfiles() /
 * getTaskAgentProfile()) to a partial override of the global config.
 * Resolution is global-first: every unset field inherits from the global
 * sections; only explicitly set fields override. See resolveConfig().
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

export type MemoryCategory =
  | 'project'
  | 'preference'
  | 'configuration'
  | 'decision'
  | 'constraint'
  | 'problem'
  | 'solution'
  | 'todo'
  | 'environment'
  | 'convention'
  | 'api'
  | 'other';

export type MemoryScope = 'global' | 'project';

export type ExtractionTrigger = 'agent_end' | 'task_end' | 'prompt_end';

/**
 * How the secondary LLM is reached.
 *
 * 'http'      - direct OpenAI-compatible HTTP call made by this extension.
 *               Zero dependencies, nothing to register in AiderDesk, works with
 *               llama-server, Ollama /v1, LiteLLM, OpenRouter, vLLM, LM Studio.
 *               This is the default and the recommended path.
 *
 * 'aiderdesk' - uses the native Extension API
 *               TaskContext.generateText(modelId, systemPrompt, prompt), which
 *               routes through AiderDesk's own ModelManager. Requires the
 *               endpoint to be registered as an AiderDesk provider (Settings ->
 *               Providers -> "OpenAI-compatible") and a model with that id.
 *               Kept as an option because it is the documented native way to
 *               make a secondary-model call from an extension.
 *
 * Neither transport uses the main agent model: 'http' bypasses AiderDesk
 * entirely, 'aiderdesk' targets the explicit `model_id` you configure.
 */
export type LlmTransport = 'http' | 'aiderdesk';

export interface SecondaryLlmConfig {
  transport: LlmTransport;
  /**
   * AiderDesk model id in "provider/model" form, used only when
   * transport === 'aiderdesk'. Example: "openai-compatible/engram-secondary".
   */
  model_id: string;
  /** OpenAI-compatible base URL, e.g. http://192.168.1.13:4000/v1 */
  base_url: string;
  /** API key. llama-server / Ollama usually accept any non-empty string, e.g. "local". */
  api_key: string;
  /** Model id served by the endpoint. */
  model: string;
  temperature: number;
  max_tokens: number;
  timeout_ms: number;
  /** Optional extra HTTP headers (e.g. auth headers for some gateways). */
  headers?: Record<string, string>;
}

export interface ExtractionConfig {
  enabled: boolean;
  trigger: ExtractionTrigger;
  max_messages: number;
  min_importance: number;
  max_input_tokens: number;
  /** Hard cap on how many candidate facts one extraction round may produce. */
  max_candidates: number;
  /** Max existing memories fetched per candidate for dedup/update/conflict classification. */
  max_existing_for_dedup: number;
}

export interface RetrievalConfig {
  enabled: boolean;
  max_memories: number;
  /**
   * Desired minimum relevance (0..1). AiderDesk's native vector search applies a
   * GLOBAL `memory.maxDistance` setting, not a per-call threshold, so this value is
   * used as a best-effort client-side hint (see retrieval.ts). It never blocks retrieval.
   */
  min_relevance: number;
  /** Include global-scope memories (projectId === '') alongside project memories. */
  include_global: boolean;
}

export interface ConsolidationConfig {
  enabled: boolean;
  interval_tasks: number;
  /** When true, consolidation never deletes; it only merges/updates. */
  safe_mode: boolean;
}

export interface PrivacyConfig {
  redact_secrets: boolean;
}

export interface LoggingConfig {
  enabled: boolean;
  level: 'debug' | 'info' | 'warn' | 'error';
}

/**
 * Per-agent override block, keyed by AiderDesk AgentProfile.id.
 *
 * Every field is optional and inherits from the global section when unset.
 * Only the fields you set are applied on top of the global config, so an
 * agent can, for example, use a different secondary model while inheriting
 * everything else, or disable memory entirely.
 */
export interface AgentOverrides {
  /** Master switch for this agent. Unset = inherit the global `enabled`. */
  enabled?: boolean;
  secondary_llm?: Partial<SecondaryLlmConfig>;
  extraction?: Partial<ExtractionConfig>;
  retrieval?: Partial<RetrievalConfig>;
  consolidation?: Partial<ConsolidationConfig>;
  privacy?: Partial<PrivacyConfig>;
  logging?: Partial<LoggingConfig>;
}

export interface EngramConfig {
  enabled: boolean;
  secondary_llm: SecondaryLlmConfig;
  extraction: ExtractionConfig;
  retrieval: RetrievalConfig;
  consolidation: ConsolidationConfig;
  privacy: PrivacyConfig;
  logging: LoggingConfig;
  /**
   * Per-agent overrides keyed by AiderDesk agent profile id
   * (e.g. "local", "intesa", ...). Empty object = global config for all agents.
   */
  agents: Record<string, AgentOverrides>;
}

export const DEFAULT_CONFIG: EngramConfig = {
  enabled: true,
  secondary_llm: {
    transport: 'http',
    model_id: 'openai-compatible/engram-secondary',
    base_url: 'http://192.168.1.13:4000/v1',
    api_key: 'local',
    model: 'synthetic/syn:small:text',
    temperature: 0.1,
    max_tokens: 8192,
    timeout_ms: 30000,
    headers: {},
  },
  extraction: {
    enabled: true,
    trigger: 'agent_end',
    max_messages: 30,
    min_importance: 2,
    max_input_tokens: 12000,
    max_candidates: 20,
    max_existing_for_dedup: 12,
  },
  retrieval: {
    enabled: true,
    max_memories: 8,
    min_relevance: 0.65,
    include_global: true,
  },
  consolidation: {
    enabled: true,
    interval_tasks: 20,
    safe_mode: true,
  },
  privacy: {
    redact_secrets: true,
  },
  logging: {
    enabled: true,
    level: 'info',
  },
  agents: {},
};

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

const isString = (v: unknown): boolean => typeof v === 'string';
const isNumber = (v: unknown): boolean => typeof v === 'number' && Number.isFinite(v);
const isBoolean = (v: unknown): boolean => typeof v === 'boolean';
const isTransport = (v: unknown): boolean => v === 'http' || v === 'aiderdesk';
const isTrigger = (v: unknown): boolean => v === 'agent_end' || v === 'task_end' || v === 'prompt_end';
const isLogLevel = (v: unknown): boolean => v === 'debug' || v === 'info' || v === 'warn' || v === 'error';

/**
 * The config sections that can be overridden per agent. Keyed validators so
 * the global merge and the per-agent merge accept exactly the same fields
 * with exactly the same types.
 */
const SECTION_KEYS = {
  secondary_llm: {
    transport: isTransport,
    model_id: isString,
    base_url: isString,
    api_key: isString,
    model: isString,
    temperature: isNumber,
    max_tokens: isNumber,
    timeout_ms: isNumber,
    headers: isPlainObject,
  },
  extraction: {
    enabled: isBoolean,
    trigger: isTrigger,
    max_messages: isNumber,
    min_importance: isNumber,
    max_input_tokens: isNumber,
    max_candidates: isNumber,
    max_existing_for_dedup: isNumber,
  },
  retrieval: {
    enabled: isBoolean,
    max_memories: isNumber,
    min_relevance: isNumber,
    include_global: isBoolean,
  },
  consolidation: {
    enabled: isBoolean,
    interval_tasks: isNumber,
    safe_mode: isBoolean,
  },
  privacy: {
    redact_secrets: isBoolean,
  },
  logging: {
    enabled: isBoolean,
    level: isLogLevel,
  },
} as const;

type SectionName = keyof typeof SECTION_KEYS;
const SECTION_NAMES = Object.keys(SECTION_KEYS) as SectionName[];

/** Copy every field from `src` that passes its validator onto `target`. */
function applySection(target: Record<string, unknown>, src: Record<string, unknown>, key: SectionName): void {
  for (const [field, valid] of Object.entries(SECTION_KEYS[key])) {
    const value = src[field];
    if (valid(value)) target[field] = value;
  }
}

/** Validate one per-agent override block. Drops unknown agents' junk silently. */
function mergeAgentOverride(raw: unknown): AgentOverrides | null {
  if (!isPlainObject(raw)) return null;
  const ov: Record<string, unknown> = {};
  if (typeof raw.enabled === 'boolean') ov.enabled = raw.enabled;
  for (const section of SECTION_NAMES) {
    const src = raw[section];
    if (!isPlainObject(src)) continue;
    const target: Record<string, unknown> = {};
    applySection(target, src, section);
    if (Object.keys(target).length) ov[section] = target;
  }
  return Object.keys(ov).length ? (ov as AgentOverrides) : null;
}

/**
 * Deep-merge a partial, user-supplied config onto the defaults.
 * Guarantees every field exists with a sane type, so the rest of the
 * extension can rely on a fully-populated config object.
 */
export function mergeConfig(raw: unknown): EngramConfig {
  const base: EngramConfig = structuredClone(DEFAULT_CONFIG);
  if (!isPlainObject(raw)) return base;

  const r = raw as Record<string, unknown>;

  if (typeof r.enabled === 'boolean') base.enabled = r.enabled;

  for (const section of SECTION_NAMES) {
    const src = r[section];
    if (isPlainObject(src)) {
      applySection(base[section] as unknown as Record<string, unknown>, src, section);
    }
  }

  if (isPlainObject(r.agents)) {
    for (const [rawId, rawOverride] of Object.entries(r.agents as Record<string, unknown>)) {
      const id = rawId.trim();
      if (!id) continue;
      const ov = mergeAgentOverride(rawOverride);
      if (ov) base.agents[id] = ov;
    }
  }

  return base;
}

/**
 * Resolve the effective config for one agent.
 *
 * Global-first inheritance: the returned object is the global config with the
 * agent's validated overrides applied on top, field by field. An unknown or
 * null agent id yields the global config unchanged.
 *
 * Returns the shared config object when there is no override for the agent -
 * callers treat config as read-only (they never mutate it in place).
 */
export function resolveConfig(config: EngramConfig, agentId?: string | null): EngramConfig {
  const override = agentId ? config.agents?.[agentId] : undefined;
  if (!override) return config;

  const out: EngramConfig = structuredClone(config);
  if (typeof override.enabled === 'boolean') out.enabled = override.enabled;
  for (const section of SECTION_NAMES) {
    const src = override[section];
    if (isPlainObject(src)) {
      Object.assign(out[section] as unknown as Record<string, unknown>, src);
    }
  }
  return out;
}

/** True when this agent id has any override configured. */
export function hasAgentOverride(config: EngramConfig, agentId?: string | null): boolean {
  return Boolean(agentId && config.agents?.[agentId]);
}

export function loadConfig(configPath: string): EngramConfig {
  try {
    if (!existsSync(configPath)) return structuredClone(DEFAULT_CONFIG);
    const text = readFileSync(configPath, 'utf-8');
    const parsed = JSON.parse(text);
    return mergeConfig(parsed);
  } catch {
    return structuredClone(DEFAULT_CONFIG);
  }
}

export function saveConfig(configPath: string, config: EngramConfig): void {
  const merged = mergeConfig(config);
  writeFileSync(configPath, JSON.stringify(merged, null, 2), 'utf-8');
}
