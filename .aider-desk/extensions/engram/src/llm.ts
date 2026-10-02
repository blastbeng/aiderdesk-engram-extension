/**
 * Minimal, dependency-free client for an OpenAI-compatible chat-completions
 * endpoint (llama-server, Ollama /v1, LiteLLM, OpenRouter, vLLM, ...).
 *
 * Uses Node's global `fetch` (Node >= 18; AiderDesk 0.81.0 ships Node 22+).
 *
 * Design rules:
 *  - NEVER throws. Every failure is returned as a typed result so callers can
 *    degrade gracefully (the main agent must keep working when the secondary
 *    LLM is offline).
 *  - No streaming, no tools, no function calling. The secondary LLM is a pure
 *    text-in / text-out memory worker, never an agent.
 */
import type { TaskContext } from '@aiderdesk/extensions';
import type { SecondaryLlmConfig } from './config';

export type LlmFailureKind =
  | 'disabled'
  | 'timeout'
  | 'unreachable'
  | 'http_error'
  | 'malformed'
  | 'empty'
  | 'aborted';

export interface LlmSuccess {
  ok: true;
  text: string;
  /** Tokens reported by the endpoint when available (llama-server / OpenAI both do). */
  promptTokens?: number;
  completionTokens?: number;
  durationMs: number;
}

export interface LlmFailure {
  ok: false;
  kind: LlmFailureKind;
  /** Short, log-safe message. Never contains request content. */
  message: string;
  status?: number;
  durationMs: number;
}

export type LlmResult = LlmSuccess | LlmFailure;

export interface ChatRequest {
  system: string;
  user: string;
  /** Overrides the configured temperature for this call. */
  temperature?: number;
  /** Overrides the configured max_tokens for this call. */
  maxTokens?: number;
  /** External cancellation (e.g. extension unload). */
  signal?: AbortSignal;
}

function joinUrl(base: string): string {
  const trimmed = base.replace(/\/+$/, '');
  // Accept both `http://host:4000` and `http://host:4000/v1`.
  if (/\/v\d+(\/+)?$/.test(trimmed)) {
    return `${trimmed}/chat/completions`;
  }
  return `${trimmed}/v1/chat/completions`;
}

function isAbortError(err: unknown): boolean {
  return (
    typeof err === 'object' &&
    err !== null &&
    (err as { name?: string }).name === 'AbortError'
  );
}

/**
 * One non-streaming chat completion.
 */
export async function chat(cfg: SecondaryLlmConfig, req: ChatRequest): Promise<LlmResult> {
  const started = Date.now();

  if (!cfg.base_url || !cfg.model) {
    return { ok: false, kind: 'disabled', message: 'secondary_llm.base_url/model not configured', durationMs: 0 };
  }

  const url = joinUrl(cfg.base_url);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort('timeout'), Math.max(1000, cfg.timeout_ms));

  const onExternalAbort = () => controller.abort('cancelled');
  if (req.signal) {
    if (req.signal.aborted) {
      clearTimeout(timeout);
      return { ok: false, kind: 'aborted', message: 'request cancelled', durationMs: 0 };
    }
    req.signal.addEventListener('abort', onExternalAbort, { once: true });
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(cfg.headers ?? {}),
  };
  if (cfg.api_key) headers['Authorization'] = `Bearer ${cfg.api_key}`;

  const body = JSON.stringify({
    model: cfg.model,
    temperature: req.temperature ?? cfg.temperature,
    max_tokens: req.maxTokens ?? cfg.max_tokens,
    stream: false,
    messages: [
      { role: 'system', content: req.system },
      { role: 'user', content: req.user },
    ],
  });

  try {
    const res = await fetch(url, { method: 'POST', headers, body, signal: controller.signal });

    if (!res.ok) {
      // Read a bounded slice of the body for diagnostics; never log it verbatim.
      let snippet = '';
      try {
        snippet = (await res.text()).slice(0, 240);
      } catch {
        snippet = '';
      }
      return {
        ok: false,
        kind: 'http_error',
        status: res.status,
        message: `HTTP ${res.status} from ${new URL(url).host}${snippet ? `: ${snippet}` : ''}`,
        durationMs: Date.now() - started,
      };
    }

    let payload: unknown;
    try {
      payload = await res.json();
    } catch {
      return { ok: false, kind: 'malformed', message: 'response body is not valid JSON', durationMs: Date.now() - started };
    }

    const text = extractCompletion(payload);
    if (text === null) {
      // Reasoning models often burn the whole budget in reasoning_content and
      // return content: null with finish_reason "length" - surface that.
      const finish = (payload as { choices?: { finish_reason?: unknown }[] }).choices?.[0]?.finish_reason;
      const hint = typeof finish === 'string' && finish ? ` (finish_reason=${finish} - reasoning models may need a larger max_tokens)` : '';
      return { ok: false, kind: 'malformed', message: `response has no chat completion content${hint}`, durationMs: Date.now() - started };
    }
    if (!text.trim()) {
      return { ok: false, kind: 'empty', message: 'empty completion', durationMs: Date.now() - started };
    }

    const usage = (payload as { usage?: { prompt_tokens?: number; completion_tokens?: number } }).usage;
    return {
      ok: true,
      text,
      promptTokens: typeof usage?.prompt_tokens === 'number' ? usage.prompt_tokens : undefined,
      completionTokens: typeof usage?.completion_tokens === 'number' ? usage.completion_tokens : undefined,
      durationMs: Date.now() - started,
    };
  } catch (err) {
    if (isAbortError(err)) {
      const reason = String((controller.signal as unknown as { reason?: unknown }).reason ?? '');
      if (reason === 'timeout') {
        return { ok: false, kind: 'timeout', message: `timed out after ${cfg.timeout_ms} ms`, durationMs: Date.now() - started };
      }
      return { ok: false, kind: 'aborted', message: 'request cancelled', durationMs: Date.now() - started };
    }
    const msg = err instanceof Error ? err.message : String(err);
    return { ok: false, kind: 'unreachable', message: `endpoint unreachable: ${msg}`, durationMs: Date.now() - started };
  } finally {
    clearTimeout(timeout);
    if (req.signal) req.signal.removeEventListener('abort', onExternalAbort);
  }
}

/**
 * Tolerant completion extraction. OpenAI-compatible servers differ:
 *  - choices[0].message.content (standard)
 *  - choices[0].text (completions-style)
 *  - message.content (some llama.cpp builds / shims)
 *  - content (Ollama-native shims)
 */
function extractCompletion(payload: unknown): string | null {
  if (typeof payload !== 'object' || payload === null) return null;
  const p = payload as Record<string, unknown>;

  const choices = p.choices;
  if (Array.isArray(choices) && choices.length > 0) {
    const first = choices[0] as Record<string, unknown>;
    const message = first.message as Record<string, unknown> | undefined;
    if (typeof message?.content === 'string') return message.content;
    // Some servers return content parts arrays.
    if (Array.isArray(message?.content)) {
      const parts = (message!.content as unknown[]).filter(
        (part): part is { type: string; text: string } =>
          typeof part === 'object' &&
          part !== null &&
          (part as { type?: string }).type === 'text' &&
          typeof (part as { text?: unknown }).text === 'string',
      );
      if (parts.length) return parts.map((part) => part.text).join('\n');
    }
    if (typeof first.text === 'string') return first.text;
  }

  const direct = p.message as Record<string, unknown> | undefined;
  if (typeof direct?.content === 'string') return direct.content;
  if (typeof p.content === 'string') return p.content;

  return null;
}

/**
 * Cheap health probe used by `memory:stats` and by the startup log.
 * Never blocks longer than `timeout_ms`.
 */
export async function probe(cfg: SecondaryLlmConfig): Promise<LlmResult> {
  if (cfg.transport === 'aiderdesk') {
    return { ok: true, text: 'OK (aiderdesk transport - probe skipped)', durationMs: 0 };
  }
  return chat(cfg, {
    system: 'You are a health probe. Reply with exactly: OK',
    user: 'Reply: OK',
    // Reasoning models (e.g. synthetic/*, DeepSeek-R1 style) spend tokens on
    // reasoning_content before any content appears; too small a cap yields
    // content: null with finish_reason "length". 1024 keeps the probe cheap
    // while leaving real room for the reasoning + a one-word answer (256 was
    // exhausted by reasoning alone against syn:small in live testing).
    maxTokens: 1024,
    temperature: 0,
  });
}

/**
 * Bounded retry for TRANSIENT failures only. An endpoint that is restarting or
 * a connection blip should not kill a whole background extraction; nothing
 * else is worth a second call:
 *   - 'aborted' / 'disabled' are never retried (cancellation / config error);
 *   - 'malformed' / 'empty' are not retried (model behavior, not transport);
 *   - 'http_error' is retried only for 429 and 5xx (4xx client errors are
 *     deterministic).
 * External cancellation between attempts stops the loop immediately.
 */
const MAX_LLM_ATTEMPTS = 3;
const RETRY_BACKOFF_MS = 1000;

function isTransientFailure(result: LlmFailure): boolean {
  if (result.kind === 'unreachable' || result.kind === 'timeout') return true;
  if (result.kind === 'http_error') {
    return result.status === 429 || (result.status !== undefined && result.status >= 500);
  }
  return false;
}

async function withRetries(req: ChatRequest, attempt: () => Promise<LlmResult>): Promise<LlmResult> {
  let result = await attempt();
  for (let n = 1; n < MAX_LLM_ATTEMPTS && !result.ok && isTransientFailure(result); n++) {
    if (req.signal?.aborted) break;
    await new Promise((resolve) => setTimeout(resolve, RETRY_BACKOFF_MS * n));
    if (req.signal?.aborted) break;
    result = await attempt();
  }
  return result;
}

/**
 * Transport dispatcher.
 *
 * 'http'      -> direct OpenAI-compatible call (chat above).
 * 'aiderdesk' -> TaskContext.generateText(modelId, system, prompt), the native
 *                Extension API for "quick LLM calls (e.g. summarization,
 *                classification) within extensions". Requires a task context;
 *                falls back to HTTP when none is available so a background
 *                extraction never dies because a task was closed.
 *
 * Both transports retry transient failures (see withRetries).
 */
export async function chatWithTransport(
  cfg: SecondaryLlmConfig,
  req: ChatRequest,
  taskContext?: TaskContext | null,
): Promise<LlmResult> {
  // HTTP transport, or the degraded aiderdesk-without-task-context fallback
  // (a background extraction whose task was closed in the meantime).
  if (cfg.transport !== 'aiderdesk' || !taskContext || typeof taskContext.generateText !== 'function') {
    return withRetries(req, () => chat(cfg, req));
  }

  const started = Date.now();
  if (!cfg.model_id) {
    return {
      ok: false,
      kind: 'disabled',
      message: 'secondary_llm.model_id is required when transport = "aiderdesk"',
      durationMs: 0,
    };
  }

  if (req.signal?.aborted) {
    return { ok: false, kind: 'aborted', message: 'request cancelled', durationMs: 0 };
  }

  return withRetries(req, async (): Promise<LlmResult> => {
    try {
      const text = await taskContext.generateText(cfg.model_id, req.system, req.user);
      if (text === undefined || text === null) {
        return {
          ok: false,
          kind: 'empty',
          message: `generateText("${cfg.model_id}") returned no text - check that the model id exists in Settings > Models`,
          durationMs: Date.now() - started,
        };
      }
      if (!text.trim()) {
        return { ok: false, kind: 'empty', message: 'empty completion', durationMs: Date.now() - started };
      }
      return { ok: true, text, durationMs: Date.now() - started };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { ok: false, kind: 'unreachable', message: `generateText failed: ${msg}`, durationMs: Date.now() - started };
    }
  });
}
