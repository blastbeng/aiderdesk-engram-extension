/**
 * Offline test harness for the Engram memory extension.
 *
 * It drives the REAL extension code paths (extraction, dedup classification,
 * update/conflict resolution, consolidation, retrieval, redaction, LLM-failure
 * handling) against:
 *   - a real local OpenAI-compatible HTTP server (tests/mock-server.ts), so
 *     fetch, headers, JSON bodies, usage parsing and timeouts are exercised
 *     rather than stubbed;
 *   - an in-memory MemoryContext that reproduces the two v0.81.0 native
 *     behaviours the extension relies on (tests/mock-memory.ts).
 *
 * Scenarios 1-6 map 1:1 to the six acceptance tests requested for this
 * extension. Scenario 7 is a bonus check for the native 'aiderdesk' transport.
 * Exit code 0 = every assertion passed.
 *
 * Run (from the extension directory):
 *   ./node_modules/.bin/jiti tests/run.ts
 * or:
 *   node tests/run.mjs
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { ContextMessage } from '@aiderdesk/extensions';

import { mergeConfig, type EngramConfig } from '../src/config';
import { logger } from '../src/logger';
import { runExtraction, type ExtractionReport } from '../src/extraction';
import { runConsolidation } from '../src/consolidation';
import { retrieveForPrompt } from '../src/retrieval';
import { decodeMemory } from '../src/memory-format';
import { importanceOf, metaForNew, statementOf } from '../src/store';
import { loadState, type EngramState } from '../src/state';
import { probe } from '../src/llm';

import { startMockLlm, scriptedResponder, type MockServer, type Responder } from './mock-server';
import { MockMemoryContext } from './mock-memory';
import { createLogSink, mockExtensionContext, mockTaskContext } from './mock-context';

// ------------------------------------------------------------------ helpers

let seq = 0;

function userMsg(text: string): ContextMessage {
  return { id: `m${seq++}`, role: 'user', content: text };
}

function assistantMsg(text: string): ContextMessage {
  return { id: `m${seq++}`, role: 'assistant', content: text, editedFiles: ['src/inference.md'] };
}

function toolMsg(names: string[]): ContextMessage {
  return {
    id: `m${seq++}`,
    role: 'tool',
    // ToolResultPart.output is an object type in the .d.ts; the transcript
    // reducer only reads toolName, so a loose cast is safe here.
    content: names.map((toolName, i) => ({
      type: 'tool-result',
      toolCallId: `c${seq}-${i}`,
      toolName,
      output: { ok: true },
    })),
  } as unknown as ContextMessage;
}

interface Harness {
  server: MockServer;
  memory: MockMemoryContext;
  context: ReturnType<typeof mockExtensionContext>;
  config: EngramConfig;
  state: EngramState;
  projectDir: string;
  statePath: string;
  close: () => Promise<void>;
}

function harnessConfig(baseUrl: string, overrides: Record<string, unknown> = {}): EngramConfig {
  return mergeConfig({
    secondary_llm: {
      base_url: baseUrl,
      api_key: 'harness-key',
      model: 'mock-model',
      temperature: 0,
      timeout_ms: 5000,
    },
    logging: { level: 'debug' },
    ...overrides,
  });
}

async function makeHarness(responder: Responder, overrides: Record<string, unknown> = {}): Promise<Harness> {
  const server = await startMockLlm(responder);
  const memory = new MockMemoryContext();
  const projectDir = `/tmp/engram-harness-${seq++}`;
  const dir = mkdtempSync(join(tmpdir(), 'engram-test-'));
  const statePath = join(dir, 'state.json');
  const sink = createLogSink();
  const context = mockExtensionContext(memory, projectDir, { sink });
  logger.bind(context);
  const config = harnessConfig(server.baseUrl, overrides);
  logger.setConfig(config.logging);
  const state = loadState(statePath);
  return {
    server,
    memory,
    context,
    config,
    state,
    projectDir,
    statePath,
    close: async () => {
      await server.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

async function extract(h: Harness, messages: ContextMessage[], taskId: string): Promise<ExtractionReport> {
  return runExtraction({
    context: h.context,
    messages,
    projectDir: h.projectDir,
    taskId,
    config: h.config,
    state: h.state,
    statePath: h.statePath,
    taskContext: null,
  });
}

const systemOf = (req: { messages?: { role: string; content: string }[] }): string =>
  req.messages?.find((m) => m.role === 'system')?.content ?? '';

const joined = (req: { messages?: { role: string; content: string }[] }): string =>
  (req.messages ?? []).map((m) => m.content).join('\n');

/** The conversation used by scenarios 1-3: one durable decision, nothing else. */
const LLAMA_CONVERSATION = (): ContextMessage[] => [
  userMsg('For this project we decided to use llama.cpp as the inference backend instead of Ollama. llama-server will run on port 4000.'),
  assistantMsg('Got it. I wired the inference calls through llama.cpp (llama-server) and left Ollama out of the stack.'),
  toolMsg(['read_file', 'edit_file']),
  assistantMsg('Done - the llama-server configuration now points at port 4000.'),
];

// ------------------------------------------------------------------- runner

let passed = 0;
const failed: string[] = [];

function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function scenario(name: string, body: () => Promise<void>): Promise<void> {
  const t0 = Date.now();
  try {
    await body();
    passed += 1;
    console.log(`PASS  ${name}  (${Date.now() - t0} ms)`);
  } catch (error) {
    failed.push(name);
    console.error(`FAIL  ${name}  (${Date.now() - t0} ms)\n      ${error instanceof Error ? error.message : String(error)}`);
  }
}

// ---------------------------------------------------------------- scenarios

async function main(): Promise<void> {
  console.log('Engram memory extension - offline acceptance harness\n');

  // -------------------------------------------------------------- scenario 1
  await scenario('1. A stated decision is extracted exactly once, as a project decision memory', async () => {
    const h = await makeHarness(scriptedResponder());
    try {
      const report = await extract(h, LLAMA_CONVERSATION(), 'task-1');

      check(!report.failure, `extraction reported a failure: ${report.failure}`);
      check(report.candidates >= 1, `expected at least 1 candidate, got ${report.candidates}`);
      check(report.stored === 1, `expected exactly 1 stored memory, got ${report.stored} (report: ${JSON.stringify(report)})`);
      check(h.memory.count() === 1, `store must hold exactly 1 entry, holds ${h.memory.count()}`);

      const entry = (await h.memory.getAllMemories())[0];
      const decoded = decodeMemory(entry.content);
      check(decoded !== null, 'stored memory must carry the Engram metadata footer');
      check(decoded!.meta.category === 'decision', `category should be 'decision', got '${decoded!.meta.category}'`);
      check(decoded!.meta.scope === 'project', `scope should be 'project', got '${decoded!.meta.scope}'`);
      check(decoded!.meta.importance >= 3, `importance should be >= 3, got ${decoded!.meta.importance}`);
      check(entry.projectId === h.projectDir, 'a project memory must be stored under the project id');
      check(/llama\.cpp/i.test(decoded!.statement), 'statement must mention llama.cpp');

      // Extraction + dedup classification = exactly 2 LLM calls, both to the
      // configured OpenAI-compatible endpoint.
      check(h.server.requests.length === 2, `expected 2 LLM calls (extract + classify), got ${h.server.requests.length}`);
      check(h.server.requests.every((r) => r.model === 'mock-model'), 'requests must target the configured model');
      check(
        h.server.requests.every((r) => (r.messages ?? []).every((m) => !m.content.includes('harness-key'))),
        'the API key must travel in the Authorization header only, never in the prompt',
      );

      // Retrieval injects the fact, footer stripped.
      const { block } = await retrieveForPrompt(h.context, h.projectDir, 'which inference backend does this project use?', h.config);
      check(block !== null && /llama\.cpp/i.test(block), 'retrieval must inject the llama.cpp fact');
      check(block !== null && !block.includes('[mem '), 'the injected block must not contain the metadata footer');
    } finally {
      await h.close();
    }
  });

  // -------------------------------------------------------------- scenario 2
  await scenario('2. Restating the same fact in a new session stores no duplicate', async () => {
    const h = await makeHarness(scriptedResponder());
    try {
      const first = await extract(h, LLAMA_CONVERSATION(), 'task-2a');
      check(first.stored === 1, `setup: expected 1 stored memory, got ${first.stored}`);

      const again = await extract(
        h,
        [
          userMsg('Quick reminder about our earlier decision: the inference backend of this project is llama.cpp, not Ollama.'),
          assistantMsg('Right - llama-server stays the inference backend for this project.'),
        ],
        'task-2b',
      );

      check(!again.failure, `second extraction reported a failure: ${again.failure}`);
      check(again.duplicates === 1, `expected 1 DUPLICATE verdict, got ${again.duplicates} (report: ${JSON.stringify(again)})`);
      check(again.stored === 0, `expected 0 new memories, got ${again.stored}`);
      check(h.memory.count() === 1, `store must still hold exactly 1 entry, holds ${h.memory.count()}`);
      check(h.state.totals.duplicatesSkipped === 1, 'the duplicate must be counted in the statistics');
    } finally {
      await h.close();
    }
  });

  // -------------------------------------------------------------- scenario 3
  await scenario('3. Switching backends updates the existing memory instead of adding a second one', async () => {
    const h = await makeHarness(scriptedResponder());
    try {
      const first = await extract(h, LLAMA_CONVERSATION(), 'task-3a');
      check(first.stored === 1, `setup: expected 1 stored memory, got ${first.stored}`);

      const switched = await extract(
        h,
        [
          userMsg('Update: we switched from llama.cpp to Ollama. Ollama is now the inference backend for this project.'),
          assistantMsg('Switched the configuration to Ollama; llama-server is retired.'),
        ],
        'task-3b',
      );

      check(!switched.failure, `update extraction reported a failure: ${switched.failure}`);
      check(switched.updated === 1, `expected the old memory to be UPDATED, got updated=${switched.updated} (report: ${JSON.stringify(switched)})`);
      check(switched.stored === 0, `expected no new memory, got ${switched.stored}`);
      check(h.memory.count() === 1, `store must still hold exactly 1 entry, holds ${h.memory.count()}`);

      const entry = (await h.memory.getAllMemories())[0];
      const decoded = decodeMemory(entry.content);
      check(decoded !== null, 'updated memory must still carry the metadata footer');
      check(/ollama/i.test(decoded!.statement) && /llama\.cpp/i.test(decoded!.statement), 'updated statement must record the new state (Ollama) and what it replaced');
      check(decoded!.meta.updatedAt !== undefined, 'updated memory must carry an updatedAt timestamp');

      // Retrieval must now surface the CURRENT fact.
      const { block, count } = await retrieveForPrompt(h.context, h.projectDir, 'which inference backend should I configure for this project?', h.config);
      check(count >= 1 && block !== null && /ollama/i.test(block), 'retrieval must surface the updated Ollama fact');

      // Context-pollution guard: an unrelated prompt injects nothing.
      const none = await retrieveForPrompt(h.context, h.projectDir, 'what is the capital of France?', h.config);
      check(none.block === null && none.count === 0, 'an irrelevant prompt must inject nothing');
    } finally {
      await h.close();
    }
  });

  // -------------------------------------------------------------- scenario 4
  await scenario('4. Secrets are redacted before transmission and never stored; malformed output is repaired, corrupt data never stored', async () => {
    // Phase A: secrets.
    const h = await makeHarness(scriptedResponder());
    try {
      const key = 'sk-harness000000000000deadbeef';
      const password = 'hunter2000';
      const report = await extract(
        h,
        [
          userMsg(`Deploy note: the OpenAI key for this service is ${key} and the admin password=${password}. Configure the client with them.`),
          assistantMsg('I will configure the client, but credentials like these must not be stored anywhere.'),
        ],
        'task-4a',
      );

      check(!report.failure, `extraction reported a failure: ${report.failure}`);
      check(report.stored === 0, `nothing should be stored from a credentials-only exchange, got ${report.stored}`);
      for (const content of h.memory.statements()) {
        check(!content.includes(key) && !content.includes(password), 'a secret leaked into the memory store');
      }

      const extractRequests = h.server.requests.filter((r) => /memory clerk/i.test(systemOf(r)));
      check(extractRequests.length >= 1, 'no extraction request reached the server');
      const sent = extractRequests.map(joined).join('\n');
      check(!sent.includes(key), 'the raw API key must never be sent to the secondary LLM');
      check(!sent.includes(password), 'the raw password must never be sent to the secondary LLM');
      check(sent.includes('[REDACTED]'), 'secrets must be replaced with [REDACTED] before transmission');
    } finally {
      await h.close();
    }

    // Phase B: malformed JSON -> one repair round-trip -> recovered, stored.
    const h2 = await makeHarness((body) => {
      if (/repair|valid JSON/i.test(systemOf(body))) {
        return JSON.stringify({
          memories: [
            { content: 'Project uses Bun as its runtime instead of Node.', category: 'decision', importance: 3, scope: 'project', confidence: 0.8 },
          ],
        });
      }
      // Structurally parseable but schema-invalid (unknown category).
      return '{"memories":[{"content":"Project uses Bun as its runtime instead of Node.","category":"runtime-manager","importance":3,"scope":"project","confidence":0.8}]}';
    });
    try {
      const report = await extract(
        h2,
        [userMsg('We migrated the runtime of this project to Bun, replacing Node.'), assistantMsg('Updated all scripts to run with Bun.')],
        'task-4b',
      );
      check(!report.failure, `repair path should recover the payload, got failure: ${report.failure}`);
      check(h2.memory.count() === 1, `expected the repaired fact to be stored once, got ${h2.memory.count()}`);
      check(h2.memory.statements().some((s) => s.includes('Bun')), 'the recovered statement must be in the store');
      const repairCalls = h2.server.requests.filter((r) => /repair|valid JSON/i.test(systemOf(r)));
      check(repairCalls.length === 1, `expected exactly 1 repair round-trip, got ${repairCalls.length}`);
    } finally {
      await h.close();
    }

    // Phase C: unrecoverable output -> nothing is ever written.
    const h3 = await makeHarness(() => 'this is not json at all --- {{{');
    try {
      const report = await extract(h3, LLAMA_CONVERSATION(), 'task-4c');
      check(h3.memory.count() === 0, 'corrupt output must never be written to the store');
      check(!!report.failure, 'an unrecoverable payload must be reported as a failure');
      const repairCalls = h3.server.requests.filter((r) => /repair|valid JSON/i.test(systemOf(r)));
      check(repairCalls.length === 1, 'the repair path must be attempted exactly once before giving up');
    } finally {
      await h.close();
    }
  });

  // -------------------------------------------------------------- scenario 5
  await scenario('5. Secondary LLM offline: extraction degrades gracefully and the agent keeps working', async () => {
    const h = await makeHarness(scriptedResponder());
    try {
      // Nothing is listening on port 9.
      const deadConfig = mergeConfig({ secondary_llm: { base_url: 'http://127.0.0.1:9/v1', timeout_ms: 2000 } });

      const t0 = Date.now();
      const report = await runExtraction({
        context: h.context,
        messages: LLAMA_CONVERSATION(),
        projectDir: h.projectDir,
        taskId: 'task-5',
        config: deadConfig,
        state: h.state,
        statePath: h.statePath,
        taskContext: null,
      });
      const elapsed = Date.now() - t0;

      check(report.stored === 0, `nothing may be written while the LLM is down, got ${report.stored}`);
      check(h.memory.count() === 0, 'the store must stay empty');
      check(!!report.failure, 'the failure must be reported, not swallowed');
      check(/unreachable|timeout/i.test(report.failure!), `failure should be unreachable/timeout, got: ${report.failure}`);
      check(elapsed < 15000, `extraction must fail fast, took ${elapsed} ms`);

      const health = await probe(deadConfig.secondary_llm);
      check(health.ok === false && health.kind === 'unreachable', `probe must report the endpoint as unreachable, got ${health.ok ? 'ok' : health.kind}`);
      check(h.state.totals.llmFailures >= 1, 'the failure must be counted in the statistics');
    } finally {
      await h.close();
    }
  });

  // -------------------------------------------------------------- scenario 6
  await scenario('6. Consolidation collapses 24 redundant memories into one and keeps the distinct facts', async () => {
    const h = await makeHarness(scriptedResponder());
    try {
      const leads = ['The project', 'This repository', 'The codebase', 'Our project'];
      const tails = [
        'for every install.',
        'for all installs and scripts.',
        'whenever dependencies are installed.',
        'for dependency installation and lockfiles.',
      ];
      const variants = Array.from({ length: 24 }, (_, i) => `${leads[i % 4]} uses pnpm as the package manager ${tails[i % 4]}`);
      const distinct = [
        'Deployment target is Kubernetes in the eu-west-1 region.',
        'The release checklist requires a changelog entry before tagging.',
        'Unit tests run through vitest with coverage enabled.',
        'The API base path is /api/v2 and versioning is header-based.',
      ];

      for (let i = 0; i < variants.length; i++) {
        await h.memory.storeMemory(
          h.projectDir,
          `seed-v${i}`,
          'code-pattern',
          encodeForHarness(variants[i], { category: 'configuration', importance: 3, scope: 'project', confidence: 0.8 }),
        );
      }
      for (let i = 0; i < distinct.length; i++) {
        await h.memory.storeMemory(
          h.projectDir,
          `seed-d${i}`,
          'code-pattern',
          encodeForHarness(distinct[i], { category: 'convention', importance: 4, scope: 'project', confidence: 0.9 }),
        );
      }

      const before = h.memory.count();
      check(before === 28, `setup: expected 28 seeded memories, got ${before}`);

      // Pass 1 - safe mode: redundancy demoted, nothing deleted.
      const safe = await runConsolidation({
        context: h.context,
        projectDir: h.projectDir,
        config: h.config,
        state: h.state,
        statePath: h.statePath,
        taskContext: null,
      });
      check(!safe.failure, `safe consolidation reported a failure: ${safe.failure}`);
      check(safe.scanned === before, `expected to scan ${before} memories, scanned ${safe.scanned}`);
      check(safe.merged >= 1, `expected at least 1 merge group, got ${safe.merged}`);
      check(safe.deleted === 0, 'safe_mode must not delete anything');
      const activeAfterSafe = (await h.memory.getAllMemories()).filter((e) => importanceOf(e) >= 2).length;
      check(activeAfterSafe <= 6, `effective (importance >= 2) memories should collapse from ${before} to <= 6, got ${activeAfterSafe}`);

      // Pass 2 - aggressive: the redundant copies are actually removed.
      const aggressiveConfig: EngramConfig = {
        ...h.config,
        consolidation: { ...h.config.consolidation, safe_mode: false },
      };
      const aggressive = await runConsolidation({
        context: h.context,
        projectDir: h.projectDir,
        config: aggressiveConfig,
        state: h.state,
        statePath: h.statePath,
        taskContext: null,
      });
      check(!aggressive.failure, `aggressive consolidation reported a failure: ${aggressive.failure}`);
      check(aggressive.deleted >= 20, `expected the redundant copies to be deleted, got ${aggressive.deleted}`);

      const after = h.memory.count();
      check(after <= before - 20, `store should shrink from ${before} to <= ${before - 20}, got ${after}`);
      const statements = (await h.memory.getAllMemories()).map((e) => statementOf(e));
      check(statements.some((s) => /pnpm/i.test(s)), 'the merged pnpm fact must survive');
      for (const fact of distinct) {
        check(statements.some((s) => s === fact), `distinct fact must survive consolidation: "${fact}"`);
      }
    } finally {
      await h.close();
    }
  });

  // -------------------------------------------------------------- scenario 7
  await scenario('7. Bonus: the native aiderdesk transport routes every call to the configured secondary model', async () => {
    const h = await makeHarness(scriptedResponder());
    try {
      const calls: string[] = [];
      const taskContext = mockTaskContext({
        id: 'task-7',
        generateText: async (modelId, system, prompt) => {
          calls.push(modelId);
          const out = scriptedResponder()({
            model: modelId,
            messages: [
              { role: 'system', content: system },
              { role: 'user', content: prompt },
            ],
          });
          return typeof out === 'string' ? out : null;
        },
      });
      const context = mockExtensionContext(h.memory, h.projectDir, { taskContext });
      const config = mergeConfig({ secondary_llm: { transport: 'aiderdesk', model_id: 'local-secondary/engram' } });

      const report = await runExtraction({
        context,
        messages: LLAMA_CONVERSATION(),
        projectDir: h.projectDir,
        taskId: 'task-7',
        config,
        state: h.state,
        statePath: h.statePath,
        taskContext,
      });

      check(!report.failure, `extraction reported a failure: ${report.failure}`);
      check(report.stored === 1 && h.memory.count() === 1, `expected 1 stored memory, got ${report.stored} (report: ${JSON.stringify(report)})`);
      check(h.server.requests.length === 0, 'the HTTP endpoint must not be contacted when transport = aiderdesk');
      check(calls.length >= 2, `expected >= 2 generateText calls, got ${calls.length}`);
      check(calls.every((id) => id === 'local-secondary/engram'), 'every call must target the configured secondary model id, never the main model');
    } finally {
      await h.close();
    }
  });

  // ----------------------------------------------------------------- summary
  console.log('');
  if (failed.length) {
    console.error(`${passed} passed, ${failed.length} FAILED:\n  - ${failed.join('\n  - ')}`);
    process.exitCode = 1;
  } else {
    console.log(`${passed} passed, 0 failed.`);
  }
}

/** Local wrapper so the harness does not import store.ts metaForNew types. */
function encodeForHarness(
  statement: string,
  meta: { category: 'configuration' | 'convention'; importance: number; scope: 'project'; confidence: number },
): string {
  // Reuse the extension's own encoder through store.ts's metaForNew.
  const { encodeMemory } = require('../src/memory-format') as typeof import('../src/memory-format');
  return encodeMemory(statement, metaForNew(meta));
}

main()
  .then(() => {
    // The mock HTTP server's keep-alive sockets keep the event loop alive;
    // exit explicitly once the scenarios are done.
    process.exit(process.exitCode ?? 0);
  })
  .catch((error) => {
    console.error('harness crashed:', error);
    process.exit(1);
  });
