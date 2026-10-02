/**
 * Configuration for the Engram memory extension.
 *
 * Config is persisted as `config.json` inside the extension directory and is
 * edited through the extension Settings dialog (ConfigComponent.jsx).
 *
 * The secondary LLM is a plain OpenAI-compatible HTTP endpoint (llama-server,
 * Ollama /v1, LiteLLM, OpenRouter, ...). It is reached directly over HTTP by
 * this extension and is NEVER the AiderDesk main model.
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

export interface EngramConfig {
  enabled: boolean;
  secondary_llm: SecondaryLlmConfig;
  extraction: ExtractionConfig;
  retrieval: RetrievalConfig;
  consolidation: ConsolidationConfig;
  privacy: PrivacyConfig;
  logging: LoggingConfig;
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
};

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
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

  if (isPlainObject(r.secondary_llm)) {
    const s = r.secondary_llm as Record<string, unknown>;
    if (s.transport === 'http' || s.transport === 'aiderdesk') base.secondary_llm.transport = s.transport;
    if (typeof s.model_id === 'string') base.secondary_llm.model_id = s.model_id.trim();
    if (typeof s.base_url === 'string') base.secondary_llm.base_url = s.base_url.trim();
    if (typeof s.api_key === 'string') base.secondary_llm.api_key = s.api_key;
    if (typeof s.model === 'string') base.secondary_llm.model = s.model.trim();
    if (typeof s.temperature === 'number') base.secondary_llm.temperature = s.temperature;
    if (typeof s.max_tokens === 'number') base.secondary_llm.max_tokens = s.max_tokens;
    if (typeof s.timeout_ms === 'number') base.secondary_llm.timeout_ms = s.timeout_ms;
    if (isPlainObject(s.headers)) base.secondary_llm.headers = s.headers as Record<string, string>;
  }

  if (isPlainObject(r.extraction)) {
    const e = r.extraction as Record<string, unknown>;
    if (typeof e.enabled === 'boolean') base.extraction.enabled = e.enabled;
    if (e.trigger === 'agent_end' || e.trigger === 'task_end' || e.trigger === 'prompt_end') {
      base.extraction.trigger = e.trigger;
    }
    if (typeof e.max_messages === 'number') base.extraction.max_messages = e.max_messages;
    if (typeof e.min_importance === 'number') base.extraction.min_importance = e.min_importance;
    if (typeof e.max_input_tokens === 'number') base.extraction.max_input_tokens = e.max_input_tokens;
    if (typeof e.max_candidates === 'number') base.extraction.max_candidates = e.max_candidates;
    if (typeof e.max_existing_for_dedup === 'number') base.extraction.max_existing_for_dedup = e.max_existing_for_dedup;
  }

  if (isPlainObject(r.retrieval)) {
    const c = r.retrieval as Record<string, unknown>;
    if (typeof c.enabled === 'boolean') base.retrieval.enabled = c.enabled;
    if (typeof c.max_memories === 'number') base.retrieval.max_memories = c.max_memories;
    if (typeof c.min_relevance === 'number') base.retrieval.min_relevance = c.min_relevance;
    if (typeof c.include_global === 'boolean') base.retrieval.include_global = c.include_global;
  }

  if (isPlainObject(r.consolidation)) {
    const c = r.consolidation as Record<string, unknown>;
    if (typeof c.enabled === 'boolean') base.consolidation.enabled = c.enabled;
    if (typeof c.interval_tasks === 'number') base.consolidation.interval_tasks = c.interval_tasks;
    if (typeof c.safe_mode === 'boolean') base.consolidation.safe_mode = c.safe_mode;
  }

  if (isPlainObject(r.privacy)) {
    const p = r.privacy as Record<string, unknown>;
    if (typeof p.redact_secrets === 'boolean') base.privacy.redact_secrets = p.redact_secrets;
  }

  if (isPlainObject(r.logging)) {
    const l = r.logging as Record<string, unknown>;
    if (typeof l.enabled === 'boolean') base.logging.enabled = l.enabled;
    if (l.level === 'debug' || l.level === 'info' || l.level === 'warn' || l.level === 'error') {
      base.logging.level = l.level;
    }
  }

  return base;
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
