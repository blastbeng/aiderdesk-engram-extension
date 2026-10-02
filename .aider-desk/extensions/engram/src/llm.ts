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
      return { ok: false, kind: 'malformed', message: 'response has no chat completion content', durationMs: Date.now() - started };
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
    maxTokens: 8,
    temperature: 0,
  });
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
 */
export async function chatWithTransport(
  cfg: SecondaryLlmConfig,
  req: ChatRequest,
  taskContext?: TaskContext | null,
): Promise<LlmResult> {
  if (cfg.transport !== 'aiderdesk') return chat(cfg, req);

  const started = Date.now();
  if (!taskContext || typeof taskContext.generateText !== 'function') {
    return chat(cfg, req);
  }
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
}
