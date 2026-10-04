/**
 * A real, local OpenAI-compatible chat-completions server for the offline test
 * harness. It exists so the tests exercise the extension's actual HTTP path
 * (fetch, headers, JSON body, usage parsing, timeouts) rather than stubbing
 * `chat()`.
 *
 * The "model" is a scripted rule-based responder: it reads the prompt the
 * extension built and returns the JSON a competent small model would return.
 * That is a stand-in for llama-server, not a claim about any real model.
 */
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';

export interface ChatBody {
  model?: string;
  temperature?: number;
  max_tokens?: number;
  messages?: { role: string; content: string }[];
}

export type Responder = (body: ChatBody) => string | { status: number; message: string };

export interface MockServer {
  baseUrl: string;
  requests: ChatBody[];
  close: () => Promise<void>;
}

export async function startMockLlm(
  responder: Responder,
  /** `delayMs` holds every response open past the client's timeout, so the
   *  harness can assert timeout classification (tests/run.ts scenario 12). */
  opts: { delayMs?: number } = {},
): Promise<MockServer> {
  const requests: ChatBody[] = [];

  const server: Server = createServer((req, res) => {
    if (!req.url?.endsWith('/chat/completions')) {
      res.writeHead(404).end('{}');
      return;
    }

    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
    });
    req.on('end', () => {
      let body: ChatBody = {};
      try {
        body = JSON.parse(raw) as ChatBody;
      } catch {
        res.writeHead(400, { 'Content-Type': 'application/json' }).end('{}');
        return;
      }
      requests.push(body);

      const respond = (): void => {
        const out = responder(body);
        if (typeof out === 'object') {
          res.writeHead(out.status, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: { message: out.message } }));
          return;
        }

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify({
            id: 'chatcmpl-mock',
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: body.model ?? 'mock',
            choices: [
              {
                index: 0,
                message: { role: 'assistant', content: out },
                finish_reason: 'stop',
              },
            ],
            usage: {
              prompt_tokens: Math.ceil((body.messages ?? []).reduce((n, m) => n + m.content.length, 0) / 4),
              completion_tokens: Math.ceil(out.length / 4),
              total_tokens: 0,
            },
          }),
        );
      };

      if (opts.delayMs && opts.delayMs > 0) setTimeout(respond, opts.delayMs);
      else respond();
    });
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as AddressInfo).port;

  return {
    baseUrl: `http://127.0.0.1:${port}/v1`,
    requests,
    close: () =>
      new Promise<void>((resolve) => {
        // Destroy keep-alive sockets and any in-flight (delayed) response so
        // close() resolves promptly between scenarios.
        (server as unknown as { closeAllConnections?: () => void }).closeAllConnections?.();
        server.close(() => resolve());
      }),
  };
}

/**
 * Rule-based "small model". Deterministic, so the six scenarios assert real
 * behaviour instead of a canned string.
 */
export function scriptedResponder(): Responder {
  return (body) => {
    const system = body.messages?.find((m) => m.role === 'system')?.content ?? '';
    const user = body.messages?.find((m) => m.role === 'user')?.content ?? '';

    if (/repair|valid JSON/i.test(system)) return repair(user);
    if (/classify|verdict|DUPLICATE/i.test(system)) return classify(user);
    if (/consolidat/i.test(system)) return consolidate(user);
    return extract(user);
  };
}

function extract(user: string): string {
  const memories: Record<string, unknown>[] = [];

  const push = (content: string, category: string, importance: number, scope: string, confidence: number) =>
    memories.push({ content, category, importance, scope, confidence });

  if (/llama\.cpp/i.test(user) && !/switched from llama\.cpp/i.test(user)) {
    push(
      'The project uses llama.cpp as the inference backend instead of Ollama.',
      'decision',
      4,
      'project',
      0.9,
    );
  }
  if (/switched from llama\.cpp to Ollama/i.test(user)) {
    push('The project now uses Ollama as the inference backend, replacing llama.cpp.', 'decision', 4, 'project', 0.9);
  }
  if (/prefers local models/i.test(user)) {
    push('The user prefers running models locally over cloud APIs.', 'preference', 4, 'global', 0.85);
  }
  if (/Vulkan0,CUDA0/i.test(user)) {
    push(
      'llama-server must be started with Vulkan0,CUDA0 because the RX 7800 XT and the RTX 3060 are both in use.',
      'environment',
      5,
      'global',
      0.9,
    );
  }
  if (/how do i install/i.test(user)) {
    // Deliberately low-importance noise, to prove the importance floor drops it.
    push('The user asked how to install a package.', 'other', 1, 'project', 0.4);
  }

  return JSON.stringify({ memories });
}

/**
 * Parse the exact prompt `buildClassificationUser()` produces:
 *
 *   ### Candidate 0
 *   fact: <content>
 *   category: decision | importance: 4 | scope: project
 *   existing memories:
 *   #m1 [imp=4 cat=decision scope=project]: <statement>
 *
 * Lines carry short positional aliases (`#m1`), not real UUIDs, so the parse
 * is format-generic: any `#<token> [...]:` line is captured and its handle is
 * echoed back in verdicts; extraction.ts resolves the handle via resolveAlias.
 */
function classify(user: string): string {
  const results: Record<string, unknown>[] = [];

  // split() with a capturing group yields: [prefix, index, block, index, block, ...]
  const parts = user.split(/### Candidate\s+(\d+)/i);
  for (let i = 1; i + 1 < parts.length; i += 2) {
    const index = Number(parts[i]);
    const block = parts[i + 1];

    const factMatch = block.match(/fact:\s*([\s\S]*?)\ncategory:/i);
    const candidate = (factMatch?.[1] ?? block).replace(/\s+/g, ' ').trim();

    const existingSection = block.split(/existing memories:/i)[1] ?? '';
    const existingLines = [...existingSection.matchAll(/#(\S+)\s*\[[^\]]*\]:\s*(.+)/g)].map((m) => ({
      id: m[1],
      text: m[2].trim(),
    }));

    results.push(judge(index, candidate, existingLines));
  }

  return JSON.stringify({ results });
}

function judge(
  index: number,
  candidate: string,
  existing: { id: string; text: string }[],
): Record<string, unknown> {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim();

  for (const e of existing) {
    const a = new Set(norm(candidate).split(' '));
    const b = new Set(norm(e.text).split(' '));
    let shared = 0;
    for (const t of a) if (b.has(t)) shared += 1;
    const score = shared / Math.max(a.size, b.size);

    // A stated switch of the same decision supersedes the old memory.
    const switched =
      /llama\.cpp|ollama/i.test(candidate) &&
      /llama\.cpp|ollama/i.test(e.text) &&
      /now uses|switched|replacing/i.test(candidate);

    if (switched) return { index, verdict: 'UPDATE', targetId: e.id, mergedContent: candidate };
    if (score > 0.55) return { index, verdict: 'DUPLICATE', targetId: e.id };
  }

  return { index, verdict: 'NEW' };
}

function consolidate(user: string): string {
  const lines = [...user.matchAll(/#(\S+)\s*\[[^\]]*\]:\s*(.+)/g)].map((m) => ({
    id: m[1],
    text: m[2].trim(),
  }));

  const actions: Record<string, unknown>[] = [];
  const used = new Set<string>();

  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim();

  for (let i = 0; i < lines.length; i++) {
    if (used.has(lines[i].id)) continue;
    const group = [lines[i]];
    for (let j = i + 1; j < lines.length; j++) {
      if (used.has(lines[j].id)) continue;
      const a = new Set(norm(lines[i].text).split(' '));
      const b = new Set(norm(lines[j].text).split(' '));
      let shared = 0;
      for (const t of a) if (b.has(t)) shared += 1;
      if (shared / Math.min(a.size, b.size) > 0.6) {
        group.push(lines[j]);
        used.add(lines[j].id);
      }
    }
    if (group.length > 1) {
      actions.push({
        action: 'MERGE',
        targetIds: group.map((g) => g.id),
        content: lines[i].text,
        importance: 4,
        reason: 'near-duplicate statements collapsed into one',
      });
    } else {
      actions.push({ action: 'KEEP', targetIds: [lines[i].id] });
    }
  }

  return JSON.stringify({ actions });
}

function repair(user: string): string {
  // The harness never feeds broken JSON through repair by design; scenario 4
  // covers malformed output through the raw responder override.
  const m = user.match(/\{[\s\S]*\}/);
  return m ? m[0] : '{"memories":[]}';
}
