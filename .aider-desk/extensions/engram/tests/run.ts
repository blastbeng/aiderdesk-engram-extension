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
 * Scenario 8 drives the extension class itself (through its testPaths seam) to
 * check the per-agent configuration: global config for every agent, an agent
 * override for one agent only. Scenario 9 covers relevance-primary retrieval
 * with the retrieval.min_importance floor and the deterministic dedup command.
 * Scenario 10 covers the settings dialog's global (project-less) context: the
 * agent tab list falls back to the agent profile files on disk and `_agents`
 * never reaches config.json.
 * Scenario 11 proves consolidation round-trips real UUID ids through the
 * positional aliases (m1..mN) and counts unmentioned memories as kept.
 * Scenario 12 proves a hanging endpoint is classified as timeout, not
 * unreachable, and that it never blocks the agent. Scenario 13 covers
 * resolveAlias tolerance for every id format a small model emits and the
 * monotonic per-project stats counters. Scenario 14 covers the council-review
 * regressions: a batch-cap-truncated consolidation must keep the round counter
 * (so the unseen batches are revisited), model-written secrets in
 * mergedContent / action.content must never reach the store, deterministic
 * dedup attributes removals per project, and saveConfigData re-attaches the
 * UI-only _agents list in its return value. Scenario 15 covers the command
 * surface: every command definition satisfies the host's validator, a task
 * context that cannot supply the conversation is reported instead of thrown,
 * the destructive clear is guarded by "confirm" and scoped to the project,
 * memory:forget reports usage and deletes the best match, and onUnload cancels
 * a consolidation that is in flight against an endpoint that never answers.
 * Exit code 0 = every assertion passed.
 *
 * Run (from the extension directory):
 *   ./node_modules/.bin/jiti tests/run.ts
 * or:
 *   node tests/run.mjs
 */
import { randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { AgentFinishedEvent, ContextMessage } from '@aiderdesk/extensions';

import { aliasFor, buildAliasTable, resolveAlias } from '../src/aliases';
import { hasAgentOverride, mergeConfig, resolveConfig, type EngramConfig } from '../src/config';
import { logger } from '../src/logger';
import { runExtraction, type ExtractionReport } from '../src/extraction';
import { runConsolidation } from '../src/consolidation';
import { retrieveForPrompt } from '../src/retrieval';
import { decodeMemory } from '../src/memory-format';
import { deterministicDedup, importanceOf, metaForNew, statementOf } from '../src/store';
import { loadState, projectStats, saveState, type EngramState } from '../src/state';
import { probe } from '../src/llm';

import { startMockLlm, scriptedResponder, type MockServer, type Responder } from './mock-server';
import { MockMemoryContext } from './mock-memory';
import { createLogSink, mockExtensionContext, mockTaskContext } from './mock-context';
import EngramMemoryExtension from '../index';

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
  /** The log sink wired into the mock context, for assertions on log output. */
  sink: ReturnType<typeof createLogSink>;
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
    sink,
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

  // -------------------------------------------------------------- scenario 8
  await scenario('8. Per-agent config: global applies to every agent, an override wins for that agent only', async () => {
    const h = await makeHarness(scriptedResponder());
    const dir = mkdtempSync(join(tmpdir(), 'engram-agent-'));
    const configPath = join(dir, 'config.json');
    const agentStatePath = join(dir, 'state.json');
    const logMessages: string[] = [];

    // Global config: everything on. Agent "sub" is switched off; agent "strict"
    // overrides only one field and must inherit the rest.
    writeFileSync(
      configPath,
      JSON.stringify({ ...h.config, agents: { sub: { enabled: false }, strict: { extraction: { min_importance: 5 } } } }, null, 2),
      'utf-8',
    );

    EngramMemoryExtension.testPaths = { configPath, statePath: agentStatePath };
    try {
      const ext = new EngramMemoryExtension();
      const profiles = [{ id: 'local', name: 'Local' }, { id: 'sub', name: 'Subagent' }, { id: 'strict', name: 'Strict' }];

      const ctxFor = (agentId: string | null): ReturnType<typeof mockExtensionContext> =>
        mockExtensionContext(h.memory, h.projectDir, {
          sink: createLogSink(),
          agentProfiles: profiles,
          taskContext: mockTaskContext({
            id: `task-${agentId ?? 'none'}`,
            agentProfile: agentId ? { id: agentId, name: agentId } : null,
            logMessages,
          }),
        });

      const agentFinished = (messages: ContextMessage[]): AgentFinishedEvent =>
        ({ mode: 'agent', aborted: false, contextMessages: messages, resultMessages: [] }) as AgentFinishedEvent;

      await ext.onLoad(ctxFor('local'));

      // 1. An agent with no override runs the global config and stores a memory.
      await ext.onAgentFinished(agentFinished(LLAMA_CONVERSATION()), ctxFor('local'));
      await ext.drainQueues();
      check(h.memory.count() === 1, `expected 1 memory from the global-config agent, got ${h.memory.count()}`);

      // 2. The same conversation from a disabled agent stores nothing and does
      //    not even contact the secondary LLM.
      const callsBefore = h.server.requests.length;
      check(callsBefore >= 1, 'the global-config agent must have contacted the secondary LLM');
      await ext.onAgentFinished(agentFinished(LLAMA_CONVERSATION()), ctxFor('sub'));
      await ext.drainQueues();
      check(h.memory.count() === 1, `a disabled agent must store nothing, store now has ${h.memory.count()}`);
      check(
        h.server.requests.length === callsBefore,
        `the disabled agent must not contact the secondary LLM (${h.server.requests.length - callsBefore} extra calls)`,
      );

      // 3. The settings panel receives the project's agent profiles, and the
      //    overrides survive the save/load round-trip without leaking _agents.
      const panel = (await ext.getConfigData(ctxFor('local'))) as EngramConfig & {
        _agents?: { id: string }[];
      };
      check(
        Array.isArray(panel._agents) && panel._agents.map((a) => a.id).join(',') === 'local,sub,strict',
        `_agents must list the project agent profiles, got ${JSON.stringify(panel._agents)}`,
      );
      check(panel.agents?.sub?.enabled === false, 'getConfigData lost the per-agent override');

      const saved = (await ext.saveConfigData(
        { ...panel, agents: { ...panel.agents, sub: { extraction: { min_importance: 5 } } } },
        ctxFor('local'),
      )) as EngramConfig;
      check(saved.agents?.sub?.extraction?.min_importance === 5, 'saveConfigData did not persist the per-agent override');
      check(saved.agents?.strict?.extraction?.min_importance === 5, 'saveConfigData dropped an untouched per-agent override');
      // The return value passes _agents back through for the settings UI
      // (mergeConfig strips it from what is persisted, checked next).
      check(
        Array.isArray((saved as unknown as Record<string, unknown>)._agents),
        'saveConfigData must re-attach the UI-only _agents list in its return value',
      );
      const persisted = JSON.parse(readFileSync(configPath, 'utf-8')) as Record<string, unknown>;
      check(persisted._agents === undefined, '_agents leaked into config.json');
      check(
        (persisted.agents as Record<string, { extraction?: { min_importance?: number } }>)?.sub?.extraction
          ?.min_importance === 5,
        'config.json lost the per-agent override',
      );

      // 4. Resolution semantics: set fields win, unset fields inherit.
      const resolved = resolveConfig(saved, 'strict');
      check(resolved.extraction.min_importance === 5, 'resolveConfig must apply the per-agent field');
      check(resolved.extraction.max_messages === saved.extraction.max_messages, 'resolveConfig must inherit unset fields');
      check(
        resolved.secondary_llm.base_url === saved.secondary_llm.base_url,
        'resolveConfig must inherit the whole untouched section',
      );
      check(resolved.enabled === saved.enabled, 'resolveConfig must inherit the enabled switch');
      check(resolveConfig(saved, 'no-such-agent') === saved, 'an unknown agent must yield the global config unchanged');
      check(resolveConfig(saved, null) === saved, 'a null agent must yield the global config unchanged');
      check(hasAgentOverride(saved, 'sub') && !hasAgentOverride(saved, 'local'), 'hasAgentOverride disagrees with the config');

      // 5. The per-agent config is what the commands report.
      await ext.getCommands(ctxFor('strict')).find((c) => c.name === 'memory:stats')!.execute([], ctxFor('strict'));
      check(
        logMessages.some((l) => l.includes('agent: strict') && l.includes('per-agent overrides applied')),
        `memory:stats must report the agent scope, got: ${logMessages.join(' | ')}`,
      );

      await ext.onUnload();
      check(existsSync(agentStatePath), 'onUnload did not persist the per-agent state file');
    } finally {
      EngramMemoryExtension.testPaths = null;
      rmSync(dir, { recursive: true, force: true });
      await h.close();
    }
  });

  await scenario('9. Retrieval is relevance-primary with an importance floor, and deterministic dedup removes exact duplicates', async () => {
    const h = await makeHarness(scriptedResponder());
    try {
      // Part A - deterministic dedup: three copies of one fact (importance
      // 2/4/3) plus one distinct fact; the two weaker copies must go.
      const copies: { statement: string; importance: number }[] = [
        { statement: 'The deployment runs on the green cluster.', importance: 2 },
        { statement: 'The deployment runs on the green cluster.', importance: 4 },
        { statement: 'The deployment runs on the green cluster.', importance: 3 },
      ];
      for (let i = 0; i < copies.length; i++) {
        await h.memory.storeMemory(
          h.projectDir,
          `dup-${i}`,
          'code-pattern',
          encodeForHarness(copies[i].statement, {
            category: 'configuration',
            importance: copies[i].importance,
            scope: 'project',
            confidence: 0.8,
          }),
        );
      }
      await h.memory.storeMemory(
        h.projectDir,
        'distinct-0',
        'code-pattern',
        encodeForHarness('Staging deploys every night at 03:00 UTC.', {
          category: 'configuration',
          importance: 4,
          scope: 'project',
          confidence: 0.9,
        }),
      );

      const report = await deterministicDedup(h.memory);
      check(report.scanned === 4, `dedup must scan 4 managed memories, scanned ${report.scanned}`);
      check(report.removed === 2, `dedup must remove 2 exact duplicates, removed ${report.removed}`);
      check(h.memory.count() === 2, `store must hold 2 memories after dedup, has ${h.memory.count()}`);
      const survivors = h.memory
        .statements()
        .map((content) => decodeMemory(content))
        .filter((m) => m !== null);
      check(
        survivors.some((m) => m.meta.importance === 4 && /green cluster/.test(m.statement)),
        'the highest-importance copy of the duplicated fact must survive',
      );
      check(survivors.some((m) => /Staging deploys/.test(m.statement)), 'the distinct fact must survive');

      const again = await deterministicDedup(h.memory);
      check(again.removed === 0, 'a second dedup pass must remove nothing (idempotence)');

      // Part B - relevance-primary retrieval with the importance floor
      // (retrieval.min_importance defaults to 3 in the harness config).
      h.memory.reset();
      await h.memory.storeMemory(
        h.projectDir,
        'low-0',
        'code-pattern',
        encodeForHarness('The deployment runs on the green cluster.', {
          category: 'configuration',
          importance: 2,
          scope: 'project',
          confidence: 0.9,
        }),
      );
      const none = await retrieveForPrompt(h.context, h.projectDir, 'which cluster does the deployment run on?', h.config);
      check(none.block === null, 'a below-floor memory must not be injected even when it is the best match');

      await h.memory.storeMemory(
        h.projectDir,
        'ok-0',
        'code-pattern',
        encodeForHarness('The deployment runs on the green cluster with blue workers.', {
          category: 'configuration',
          importance: 3,
          scope: 'project',
          confidence: 0.9,
        }),
      );
      const some = await retrieveForPrompt(h.context, h.projectDir, 'which cluster does the deployment run on?', h.config);
      check(some.block !== null && some.count === 1, `an at-floor memory must be injected, got count=${some.count}`);
      check(/green cluster with blue workers/.test(some.block ?? ''), 'the injected statement must not carry the footer');
    } finally {
      await h.close();
    }
  });

  await scenario('10. Settings without a project: agent tabs come from the disk scan, and _agents never persists', async () => {
    // The AiderDesk settings dialog runs the extension in a global context:
    // getProjectContext() throws there, so getConfigData() must fall back to
    // reading the agent profile files (global + open projects).
    const home = mkdtempSync(join(tmpdir(), 'engram-agents-home-'));
    const project = mkdtempSync(join(tmpdir(), 'engram-agents-proj-'));
    const dir = mkdtempSync(join(tmpdir(), 'engram-agents-ext-'));
    const configPath = join(dir, 'config.json');
    const statePath = join(dir, 'state.json');
    const previousHome = process.env.AIDER_DESK_HOME_DIR;
    EngramMemoryExtension.testPaths = { configPath, statePath };
    try {
      // Global profiles: beta (ordered first by order.json, a subagent) and
      // alpha. The project redefines alpha, so its metadata must win.
      const mkProfile = (agentsDir: string, id: string, name: string, extra: Record<string, unknown> = {}) => {
        const profileDir = join(agentsDir, name.toLowerCase().replace(/\s+/g, '-'));
        mkdirSync(profileDir, { recursive: true });
        writeFileSync(join(profileDir, 'config.json'), JSON.stringify({ id, name, provider: 'litellm', model: 'm', ...extra }));
      };
      const globalAgents = join(home, 'agents');
      mkProfile(globalAgents, 'alpha', 'Alpha Global');
      mkProfile(globalAgents, 'beta', 'Beta', { subagent: { enabled: true } });
      writeFileSync(join(globalAgents, 'order.json'), JSON.stringify({ beta: 0, alpha: 1 }));
      mkProfile(join(project, '.aider-desk', 'agents'), 'alpha', 'Alpha Project');
      mkProfile(join(project, '.aider-desk', 'agents'), 'gamma', 'Gamma');

      process.env.AIDER_DESK_HOME_DIR = home;

      const ext = new EngramMemoryExtension();
      // No agentProfiles option: the mock throws on getProjectContext(), exactly
      // like the real API inside the settings dialog.
      const ctx = mockExtensionContext(new MockMemoryContext(), project, {
        sink: createLogSink(),
        openProjectDirs: [project],
      });

      const data = (await ext.getConfigData(ctx)) as Record<string, unknown>;
      const agents = Array.isArray(data._agents) ? (data._agents as { id: string; name?: string; isSubagent?: boolean }[]) : [];
      check(agents.length === 3, `the disk scan must find 3 unique agent profiles, found ${agents.length}`);
      check(
        agents.map((a) => a.id).join(',') === 'beta,alpha,gamma',
        `order.json must order beta first and open-project profiles must be included, got ${agents.map((a) => a.id).join(',')}`,
      );
      check(agents.find((a) => a.id === 'alpha')?.name === 'Alpha Project', 'the project-level alpha profile must win over the global one');
      check(agents.find((a) => a.id === 'beta')?.isSubagent === true, 'subagent.enabled must surface as isSubagent');

      // _agents is UI-only: saving the data the dialog sends back must not
      // persist it - but the RETURN value passes the list back through, so a
      // settings framework that re-renders from the response keeps its agent
      // tabs after the first save.
      const saved = (await ext.saveConfigData(data, ctx)) as Record<string, unknown>;
      check(
        Array.isArray(saved._agents) && (saved._agents as unknown[]).length === 3,
        'saveConfigData must re-attach the UI-only _agents list in its return value',
      );
      const onDisk = JSON.parse(readFileSync(configPath, 'utf-8')) as Record<string, unknown>;
      check(!('_agents' in onDisk), 'config.json must not contain the UI-only _agents key');

      // Profiles seen once through the authoritative API (a project-scoped
      // context, e.g. after an agent run) stay available to later settings
      // dialogs, merged over the disk scan.
      const apiCtx = mockExtensionContext(new MockMemoryContext(), project, {
        sink: createLogSink(),
        agentProfiles: [{ id: 'delta', name: 'Delta', provider: 'litellm', model: 'm' }],
      });
      const viaApi = (await ext.getConfigData(apiCtx)) as Record<string, unknown>;
      check(
        (Array.isArray(viaApi._agents) ? (viaApi._agents as { id: string }[]) : []).map((a) => a.id).join(',') === 'delta',
        'the API path must be authoritative when a project context exists',
      );
      const merged = (await ext.getConfigData(ctx)) as Record<string, unknown>;
      const mergedIds = (Array.isArray(merged._agents) ? (merged._agents as { id: string }[]) : []).map((a) => a.id);
      check(mergedIds.includes('delta') && mergedIds.includes('beta'), `runtime-cached profiles must merge over the disk scan, got ${mergedIds.join(',')}`);
    } finally {
      if (previousHome === undefined) delete process.env.AIDER_DESK_HOME_DIR;
      else process.env.AIDER_DESK_HOME_DIR = previousHome;
      EngramMemoryExtension.testPaths = null;
      rmSync(home, { recursive: true, force: true });
      rmSync(project, { recursive: true, force: true });
      rmSync(dir, { recursive: true, force: true });
    }
  });

  // ------------------------------------------------------------- scenario 11
  await scenario('11. Consolidation round-trips real UUID ids through positional aliases; unmentioned memories count as kept', async () => {
    const h = await makeHarness((body) => {
      const user = body.messages?.find((m) => m.role === 'user')?.content ?? '';
      // Parse the alias-prefixed prompt lines (#m1 [imp=... ...]: statement)
      // exactly like a small model would. The store holds UUID ids the model
      // never sees, so every action below only works if alias -> real-id
      // resolution round-trips.
      const lines = [...user.matchAll(/#(\S+)\s*\[[^\]]*\]:\s*(.+)/g)].map((m) => ({ id: m[1], text: m[2].trim() }));
      const actions: Record<string, unknown>[] = [];
      const cluster = lines.filter((l) => /pnpm/i.test(l.text));
      if (cluster.length > 1) {
        actions.push({
          action: 'MERGE',
          targetIds: cluster.map((l) => l.id),
          content: 'The project uses pnpm as the package manager.',
          importance: 4,
          reason: 'duplicate cluster',
        });
      }
      // The two distinct memories are deliberately NOT mentioned anywhere:
      // the extension must count them as kept, not silently lose them.
      return JSON.stringify({ actions });
    });
    try {
      const uuids: string[] = Array.from({ length: 5 }, () => randomUUID());
      const pnpmVariants = [
        'The project uses pnpm as the package manager for every install.',
        'This repository uses pnpm as the package manager for all installs and scripts.',
        'The codebase uses pnpm as the package manager whenever dependencies are installed.',
      ];
      const distinct = [
        'Deployment target is Kubernetes in the eu-west-1 region.',
        'The release checklist requires a changelog entry before tagging.',
      ];
      for (let i = 0; i < 3; i++) {
        h.memory.seed(uuids[i], h.projectDir, 'code-pattern', encodeForHarness(pnpmVariants[i], { category: 'configuration', importance: 3, scope: 'project', confidence: 0.8 }));
      }
      for (let i = 0; i < 2; i++) {
        h.memory.seed(uuids[3 + i], h.projectDir, 'code-pattern', encodeForHarness(distinct[i], { category: 'configuration', importance: 4, scope: 'project', confidence: 0.9 }));
      }
      check(h.memory.count() === 5, `setup: expected 5 seeded memories, got ${h.memory.count()}`);

      // Pass 1 - safe mode: the pnpm cluster merges into its primary; the
      // unmentioned facts must be counted as kept, and nothing is deleted.
      const safe = await runConsolidation({ context: h.context, projectDir: h.projectDir, config: h.config, state: h.state, statePath: h.statePath, taskContext: null });
      check(!safe.failure, `safe consolidation reported a failure: ${safe.failure}`);
      check(safe.scanned === 5, `expected to scan 5 memories, scanned ${safe.scanned}`);
      check(safe.merged === 1, `expected the 3 pnpm variants to form 1 merge group, got merged=${safe.merged} (report: ${JSON.stringify(safe)})`);
      check(safe.updated === 1, `expected the merge primary to be rewritten, got updated=${safe.updated}`);
      check(safe.kept === 2, `the 2 unmentioned memories must count as kept, got kept=${safe.kept}`);
      check(safe.deleted === 0, 'safe_mode must not delete anything');

      const afterSafe = await h.memory.getAllMemories();
      check(afterSafe.length === 5, `safe mode must keep all 5 entries, has ${afterSafe.length}`);
      const primary = afterSafe.find((e) => e.id === uuids[0]);
      check(!!primary && /uses pnpm as the package manager/.test(statementOf(primary)), 'the merge primary must carry the consolidated statement under its original UUID id');
      const demoted = afterSafe.filter((e) => uuids.slice(1, 3).includes(e.id)).map((e) => importanceOf(e));
      check(demoted.length === 2 && demoted.every((i) => i === 1), `safe mode must demote the merged-away copies to importance 1, got ${demoted.join(',')}`);

      // Pass 2 - aggressive: the demoted copies are removed through their
      // resolved UUID ids; the kept facts survive untouched.
      const aggressiveConfig: EngramConfig = { ...h.config, consolidation: { ...h.config.consolidation, safe_mode: false } };
      const aggressive = await runConsolidation({ context: h.context, projectDir: h.projectDir, config: aggressiveConfig, state: h.state, statePath: h.statePath, taskContext: null });
      check(!aggressive.failure, `aggressive consolidation reported a failure: ${aggressive.failure}`);
      check(aggressive.merged === 1 && aggressive.deleted === 2, `expected 1 merge + 2 deletions, got merged=${aggressive.merged} deleted=${aggressive.deleted} (report: ${JSON.stringify(aggressive)})`);
      check(aggressive.kept === 2, `the unmentioned facts must be kept again, got kept=${aggressive.kept}`);

      const after = await h.memory.getAllMemories();
      check(after.length === 3, `store should hold 3 memories after the aggressive pass, has ${after.length}`);
      for (const id of [uuids[0], uuids[3], uuids[4]]) {
        check(after.some((e) => e.id === id), `survivor must keep its original id ${id}`);
      }
      check(after.some((e) => /Kubernetes in the eu-west-1/.test(statementOf(e))), 'the distinct facts must survive consolidation');
    } finally {
      await h.close();
    }
  });

  // ------------------------------------------------------------- scenario 12
  await scenario('12. A hanging endpoint is classified as timeout, not unreachable, and never blocks the agent', async () => {
    // The mock holds every response for 5 s; the configured timeout is 1.1 s
    // (the client floors timeouts at 1 s), so every call must abort.
    const slow = await startMockLlm(scriptedResponder(), { delayMs: 5000 });
    const h = await makeHarness(scriptedResponder());
    try {
      const cfg = mergeConfig({ secondary_llm: { base_url: slow.baseUrl, api_key: 'k', model: 'm', timeout_ms: 1100 } });

      const t0 = Date.now();
      const health = await probe(cfg.secondary_llm);
      const probeMs = Date.now() - t0;
      check(health.ok === false && health.kind === 'timeout', `probe must classify a hanging endpoint as timeout, got ${health.ok ? 'ok' : health.kind}`);
      check(/timed out after/i.test(health.message), `the timeout message must say so, got: ${health.message}`);
      check(probeMs < 4000, `the probe must fail at the configured timeout, took ${probeMs} ms`);

      const report = await runExtraction({ context: h.context, messages: LLAMA_CONVERSATION(), projectDir: h.projectDir, taskId: 'task-12', config: cfg, state: h.state, statePath: h.statePath, taskContext: null });
      check(h.memory.count() === 0, 'nothing may be written while the endpoint hangs');
      check(!!report.failure, 'the failure must be reported, not swallowed');
      check(/timed out|timeout/i.test(report.failure!) && !/unreachable/i.test(report.failure!), `the extraction failure must name the timeout, got: ${report.failure}`);
      check(h.state.totals.llmFailures >= 1, 'timeouts must be counted in the statistics');
    } finally {
      await h.close();
      await slow.close();
    }
  });

  // ------------------------------------------------------------- scenario 13
  await scenario('13. resolveAlias tolerates every id format a small model emits, and per-project stats increment monotonically', async () => {
    // Part A - alias tolerance. Production models write m3, #m3, (m3), 3...
    const ids = Array.from({ length: 4 }, () => randomUUID());
    const table = buildAliasTable(ids.map((id) => ({ id })));
    check(aliasFor(table, ids[0]) === 'm1' && aliasFor(table, ids[3]) === 'm4', 'aliasFor must seat entries in order');
    check(resolveAlias('m3', table) === ids[2], 'plain handle');
    check(resolveAlias('M3', table) === ids[2], 'case-insensitive handle');
    check(resolveAlias('#m3', table) === ids[2], 'hash-prefixed handle');
    check(resolveAlias('(m3)', table) === ids[2], 'parenthesised handle');
    check(resolveAlias('m 3', table) === ids[2], 'spaced handle');
    check(resolveAlias('m-3', table) === ids[2], 'dashed handle');
    check(resolveAlias(3, table) === ids[2], 'numeric handle (JSON number targetIds)');
    check(resolveAlias('3', table) === ids[2], 'bare digits mean prefix + index');
    check(resolveAlias(ids[2].toUpperCase(), table) === ids[2], 'a full real id echoes back in any case');
    check(resolveAlias('m0', table) === null && resolveAlias('m5', table) === null, 'out-of-range handles resolve to nothing');
    check(resolveAlias('garbage', table) === null && resolveAlias(null, table) === null && resolveAlias(undefined, table) === null, 'junk resolves to null, never to a random id');

    // Part B - per-project counters must INCREMENT across rounds, not be
    // replaced by the latest run (the Object.assign regression).
    const h = await makeHarness(scriptedResponder());
    try {
      await extract(h, LLAMA_CONVERSATION(), 'task-13a');
      check(projectStats(h.state, h.projectDir).extractions === 1, `one run must count one extraction, got ${projectStats(h.state, h.projectDir).extractions}`);
      check(h.state.totals.stored === 1, `one stored memory must be counted, got ${h.state.totals.stored}`);

      await extract(
        h,
        [
          userMsg('Quick reminder about our earlier decision: the inference backend of this project is llama.cpp, not Ollama.'),
          assistantMsg('Right - llama-server stays the inference backend for this project.'),
        ],
        'task-13b',
      );
      check(projectStats(h.state, h.projectDir).extractions === 2, `a second run must increment the counter, got ${projectStats(h.state, h.projectDir).extractions}`);
      check(h.state.totals.stored === 1, 'the duplicate round must not add to the stored counter');
      check(h.state.totals.duplicatesSkipped === 1, 'the duplicate must be counted in the statistics');

      const persisted = loadState(h.statePath);
      check(projectStats(persisted, h.projectDir).extractions === 2, 'the counters must survive the state round-trip on disk');
    } finally {
      await h.close();
    }
  });

  // ------------------------------------------------------------- scenario 15
  await scenario(
    '15. Command surface: host-legal definitions, no throw on a broken task context, guarded destructive clear, unload cancels a running consolidation',
    async () => {
      const h = await makeHarness(scriptedResponder());
      const dir = mkdtempSync(join(tmpdir(), 'engram-cmd-'));
      const configPath = join(dir, 'config.json');
      const statePath = join(dir, 'state.json');
      const logMessages: string[] = [];
      // Endpoint that answers nothing for 30 s: the only thing that can end a
      // consolidation run against it is the unload signal.
      const slow = await startMockLlm(scriptedResponder(), { delayMs: 30_000 });

      writeFileSync(configPath, JSON.stringify(h.config, null, 2), 'utf-8');
      EngramMemoryExtension.testPaths = { configPath, statePath };

      try {
        const ext = new EngramMemoryExtension();
        const taskCtx = mockExtensionContext(h.memory, h.projectDir, {
          sink: createLogSink(),
          agentProfiles: [{ id: 'local' }],
          taskContext: mockTaskContext({
            id: 'task-cmd',
            agentProfile: { id: 'local' },
            logMessages,
            contextMessages: LLAMA_CONVERSATION(),
          }),
        });
        await ext.onLoad(taskCtx);
        const commands = ext.getCommands(taskCtx);

        // ---- A. Every command must satisfy the host's validator. AiderDesk
        // checks each command against /^[a-z][a-z0-9\-_:]*$/ plus a non-empty
        // description and an execute function, and rejects the whole extension
        // registration when one of them fails.
        const expected = [
          'memory:extract',
          'memory:consolidate',
          'memory:dedup',
          'memory:stats',
          'memory:forget',
          'memory:clear-project',
        ];
        check(commands.length === expected.length, `expected ${expected.length} commands, got ${commands.length}`);
        for (const name of expected) {
          check(commands.some((c) => c.name === name), `command ${name} is missing`);
        }
        for (const c of commands) {
          check(
            /^[a-z][a-z0-9\-_:]*$/.test(c.name),
            `command name "${c.name}" is rejected by the AiderDesk host validator`,
          );
          check(typeof c.description === 'string' && c.description.trim().length > 0, `${c.name} has no description`);
          check(typeof c.execute === 'function', `${c.name} has no execute function`);
          check(c.arguments === undefined || Array.isArray(c.arguments), `${c.name} has a non-array arguments list`);
          for (const arg of c.arguments ?? []) {
            check(
              typeof arg.description === 'string' && arg.description.trim().length > 0,
              `${c.name} has an argument without a description`,
            );
          }
        }

        // ---- B. A task context that cannot hand over the conversation must
        // report it and return: the host wraps execute() in a try/catch and
        // surfaces only a generic "Extension command execution failed".
        const brokenTask = mockTaskContext({ id: 'task-broken', logMessages });
        brokenTask.getContextMessages = async () => {
          throw new Error('conversation unavailable');
        };
        const brokenCtx = mockExtensionContext(h.memory, h.projectDir, {
          sink: createLogSink(),
          taskContext: brokenTask,
        });
        const callsBefore = h.server.requests.length;
        await commands.find((c) => c.name === 'memory:extract')!.execute([], brokenCtx);
        await ext.drainQueues();
        check(
          logMessages.some((l) => l.includes('cannot read the conversation')),
          `a failed conversation read must be reported, got: ${logMessages.join(' | ')}`,
        );
        check(
          h.server.requests.length === callsBefore,
          'a failed conversation read must not contact the secondary LLM',
        );

        // No task context: `say()` falls back to the extension log, so this
        // assertion reads the sink, not the task log.
        const noTaskSink = createLogSink();
        const noTaskCtx = mockExtensionContext(h.memory, h.projectDir, { sink: noTaskSink });
        await commands.find((c) => c.name === 'memory:extract')!.execute([], noTaskCtx);
        check(
          noTaskSink.lines.some((l) => l.message.includes('no task context')),
          `memory:extract without a task must say so, got: ${noTaskSink.lines.map((l) => l.message).join(' | ')}`,
        );

        // ---- C. The destructive command is guarded, scoped and non-destructive
        // outside its scope.
        const seedManaged = (statement: string, projectDir: string, scope: 'project' | 'global', importance = 3) =>
          h.memory.storeMemory(
            projectDir,
            'task-seed',
            'code-pattern',
            encodeForHarness(statement, { category: 'configuration', importance, scope, confidence: 0.9 }),
          );
        await seedManaged('The staging cluster listens on port 8443.', h.projectDir, 'project');
        await seedManaged('The API gateway requires mTLS client certificates.', h.projectDir, 'project');
        await seedManaged('The user prefers running models locally.', '', 'global');
        await h.memory.storeMemory(
          h.projectDir,
          'task-native',
          'task',
          'A native memory written by the main agent, not managed by Engram.',
        );
        check(h.memory.count() === 4, `expected 4 seeded entries, got ${h.memory.count()}`);

        const clear = commands.find((c) => c.name === 'memory:clear-project')!;
        await clear.execute([], taskCtx);
        check(h.memory.count() === 4, 'memory:clear-project must refuse without the confirm argument');
        check(
          logMessages.some((l) => l.includes('refusing to clear')),
          'memory:clear-project must explain the refusal',
        );
        await clear.execute(['confirm'], taskCtx);
        const remaining = h.memory.statements();
        check(
          h.memory.count() === 2,
          `memory:clear-project must remove only this project's Engram memories, left: ${remaining.join(' | ')}`,
        );
        check(
          remaining.some((s) => s.includes('prefers running models locally')),
          'memory:clear-project must not touch global memories',
        );
        check(
          remaining.some((s) => s.includes('not managed by Engram')),
          'memory:clear-project must not touch memories it does not manage',
        );

        // ---- D. Forget: usage without an argument, best match deleted with it.
        const forget = commands.find((c) => c.name === 'memory:forget')!;
        await forget.execute([], taskCtx);
        check(
          logMessages.some((l) => l.includes('usage: /memory:forget')),
          `memory:forget without text must print usage, got: ${logMessages.join(' | ')}`,
        );
        await forget.execute(['prefers', 'running', 'models', 'locally'], taskCtx);
        check(h.memory.count() === 1, `memory:forget must delete the best match, ${h.memory.count()} left`);
        check(
          logMessages.some((l) => l.includes('forgot:')),
          'memory:forget must report what it deleted',
        );

        // ---- E. Statistics must report the store, the counters and the probe.
        await commands.find((c) => c.name === 'memory:stats')!.execute([], taskCtx);
        check(
          logMessages.some((l) => l.includes('AiderDesk entries')),
          'memory:stats must report the store',
        );
        check(
          logMessages.some((l) => /secondary LLM (reachable|not reachable|http_error|unreachable|timeout|malformed|empty|aborted|disabled)/.test(l)),
          `memory:stats must report the secondary LLM probe, got: ${logMessages.join(' | ')}`,
        );

        await ext.onUnload();
        check(existsSync(statePath), 'onUnload did not persist the state file');

        // ---- F. Unload must cancel a running consolidation. The forced
        // consolidation path runs up to 6 batches at >=120 s each; without the
        // unload signal `onUnload` awaits a request nothing can cancel, hanging
        // the extension reload for minutes.
        writeFileSync(
          configPath,
          JSON.stringify(
            mergeConfig({
              ...h.config,
              secondary_llm: { ...h.config.secondary_llm, base_url: slow.baseUrl, timeout_ms: 60_000 },
            }),
            null,
            2,
          ),
          'utf-8',
        );
        const slowExt = new EngramMemoryExtension();
        const slowCtx = mockExtensionContext(h.memory, h.projectDir, {
          sink: createLogSink(),
          agentProfiles: [{ id: 'local' }],
          taskContext: mockTaskContext({ id: 'task-slow', agentProfile: { id: 'local' }, logMessages }),
        });
        await slowExt.onLoad(slowCtx);
        await seedManaged('The build cache lives on the NAS share.', h.projectDir, 'project');
        await seedManaged('Release tags are annotated git tags.', h.projectDir, 'project');

        await slowExt
          .getCommands(slowCtx)
          .find((c) => c.name === 'memory:consolidate')!
          .execute([], slowCtx);
        check(
          logMessages.some((l) => l.includes('consolidation started')),
          `memory:consolidate must report the start, got: ${logMessages.join(' | ')}`,
        );

        const t0 = Date.now();
        await slowExt.onUnload();
        const unloadMs = Date.now() - t0;
        check(unloadMs < 10_000, `onUnload must cancel the in-flight consolidation, took ${unloadMs} ms`);
        check(existsSync(join(dir, 'state.json')), 'the cancelled run must still persist its state');
      } finally {
        EngramMemoryExtension.testPaths = null;
        rmSync(dir, { recursive: true, force: true });
        await h.close();
        await slow.close();
      }
    },
  );

  await scenario14();

  // ----------------------------------------------------------------- summary
  console.log('');
  if (failed.length) {
    console.error(`${passed} passed, ${failed.length} FAILED:\n  - ${failed.join('\n  - ')}`);
    process.exitCode = 1;
  } else {
    console.log(`${passed} passed, 0 failed.`);
  }
}

// Scenario 14 is a top-level function (hoisted, so main() above can call it)
// rather than a top-level await: a top-level await runs before main() is
// invoked at the bottom of the file, which printed its PASS line ahead of the
// harness header and ahead of scenarios 1-13.
async function scenario14(): Promise<void> {
  // ------------------------------------------------------------- scenario 14
  await scenario(
    '14. Council regressions: a truncated consolidation keeps the round counter, model-written secrets never land in the store, dedup reports per-project removals',
    async () => {
      const SECRET = 'sk-harness000000000000deadbeef';

      // Part A - consolidation batch cap: a run that cannot see every memory
      // (7 single-memory batches vs MAX_BATCHES = 6) must NOT reset the round
      // counter. Resetting would schedule the next run past the unseen
      // batches forever, and with oldest-first batching the NEWEST memories
      // would never consolidate.
      {
        const big = 'The regression corpus sentence is deliberately long. '.repeat(680); // ~8500 tokens
        const h = await makeHarness((body) => {
          if (/consolidat/i.test(systemOf(body))) {
            const ids = [...joined(body).matchAll(/#(\S+)\s*\[[^\]]*\]:/g)].map((m) => m[1]);
            return JSON.stringify({ actions: ids.map((id) => ({ action: 'KEEP', targetIds: [id] })) });
          }
          return '{"memories":[]}';
        });
        try {
          for (let i = 0; i < 7; i++) {
            await h.memory.storeMemory(
              h.projectDir,
              `big-${i}`,
              'code-pattern',
              encodeForHarness(`${big} variant ${i}.`, {
                category: 'convention',
                importance: 3,
                scope: 'project',
                confidence: 0.9,
              }),
            );
          }
          projectStats(h.state, h.projectDir).tasksSinceConsolidation = 5;
          saveState(h.statePath, h.state);

          const truncated = await runConsolidation({
            context: h.context,
            projectDir: h.projectDir,
            config: h.config,
            state: h.state,
            statePath: h.statePath,
            taskContext: null,
          });
          check(!truncated.failure, `truncated run reported a failure: ${truncated.failure}`);
          check(truncated.kept === 6, `exactly 6 of the 7 memories must be processed, got ${truncated.kept}`);
          check(h.memory.count() === 7, 'nothing may be deleted by KEEP actions');
          const statsAfter = projectStats(h.state, h.projectDir);
          check(
            statsAfter.tasksSinceConsolidation === 5,
            `a truncated run must NOT reset the round counter, got ${statsAfter.tasksSinceConsolidation}`,
          );
          check(
            statsAfter.lastConsolidationAt === undefined,
            'a truncated run must not stamp lastConsolidationAt (the store was not fully processed)',
          );
          check(
            h.sink.lines.some((l) => /beyond the 6-batch cap/.test(l.message)),
            'a truncated run must log the batch-cap warning',
          );

          // Non-truncated path: once every memory fits, the counter resets.
          // Delete the oldest five (the store lists them oldest-first).
          for (let i = 0; i < 5; i++) {
            const all = await h.memory.getAllMemories();
            await h.memory.deleteMemory(all[0].id);
          }
          const full = await runConsolidation({
            context: h.context,
            projectDir: h.projectDir,
            config: h.config,
            state: h.state,
            statePath: h.statePath,
            taskContext: null,
          });
          check(!full.failure, `full run reported a failure: ${full.failure}`);
          const statsFull = projectStats(h.state, h.projectDir);
          check(statsFull.tasksSinceConsolidation === 0, 'a full run must reset the round counter');
          check(typeof statsFull.lastConsolidationAt === 'number', 'a full run must stamp lastConsolidationAt');

          // The < 2 targets early return resets the counter too (it can only
          // ever answer "nothing to do"); otherwise it would loop forever.
          projectStats(h.state, h.projectDir).tasksSinceConsolidation = 3;
          saveState(h.statePath, h.state);
          await h.memory.deleteMemory((await h.memory.getAllMemories())[0].id);
          const idle = await runConsolidation({
            context: h.context,
            projectDir: h.projectDir,
            config: h.config,
            state: h.state,
            statePath: h.statePath,
            taskContext: null,
          });
          check(idle.scanned === 1, `the idle run must scan 1 memory, got ${idle.scanned}`);
          check(
            projectStats(h.state, h.projectDir).tasksSinceConsolidation === 0,
            'the < 2 targets early return must reset the round counter',
          );
        } finally {
          await h.close();
        }
      }

      // Part B - a secret inside the classifier's mergedContent must never be
      // written: the UPDATE falls back to the already-swept candidate text.
      {
        const candidate = 'The staging deploy uses a rotated API key for authentication.';
        const h = await makeHarness((body) => {
          const system = systemOf(body);
          if (/classify|verdict|DUPLICATE/i.test(system)) {
            const alias = /#(\S+)\s*\[[^\]]*\]/.exec(joined(body))?.[1] ?? 'm1';
            return JSON.stringify({
              results: [
                {
                  index: 0,
                  verdict: 'UPDATE',
                  targetId: alias,
                  mergedContent: `The staging key is ${SECRET} and rotation is weekly.`,
                },
              ],
            });
          }
          return JSON.stringify({
            memories: [{ content: candidate, category: 'configuration', importance: 4, scope: 'project', confidence: 0.9 }],
          });
        });
        try {
          const seeded = await h.memory.storeMemory(
            h.projectDir,
            'target-0',
            'code-pattern',
            encodeForHarness(candidate, { category: 'configuration', importance: 4, scope: 'project', confidence: 0.9 }),
          );
          const report = await extract(
            h,
            [userMsg('Reminder: the staging deploy uses a rotated API key for authentication.'), assistantMsg('Noted the deploy key policy.')],
            'task-14b',
          );
          check(report.updated === 1, `the UPDATE verdict must still write, got updated=${report.updated}`);
          const row = await h.memory.getMemory(seeded);
          check(row !== null, 'the updated memory must exist');
          const statement = row ? statementOf(row) : '';
          check(!statement.includes(SECRET), 'a secret in mergedContent leaked into the memory store');
          check(
            statement === candidate,
            `the UPDATE must fall back to the swept candidate text, got "${statement}"`,
          );
        } finally {
          await h.close();
        }
      }

      // Part C - a secret inside the consolidator's action.content must never
      // be written: the MERGE falls back to the primary's stored statement.
      {
        const statement = 'The backup job runs nightly on the primary database host.';
        const h = await makeHarness((body) => {
          if (/consolidat/i.test(systemOf(body))) {
            const ids = [...joined(body).matchAll(/#(\S+)\s*\[[^\]]*\]:/g)].map((m) => m[1]);
            return JSON.stringify({
              actions: [
                {
                  action: 'MERGE',
                  targetIds: ids.slice(0, 2),
                  content: `Merged backup note: ${SECRET} rotate weekly`,
                  importance: 4,
                  reason: 'near-duplicate statements',
                },
              ],
            });
          }
          return '{"memories":[]}';
        });
        try {
          for (let i = 0; i < 2; i++) {
            await h.memory.storeMemory(
              h.projectDir,
              `bak-${i}`,
              'code-pattern',
              encodeForHarness(`${statement} (copy ${i})`, {
                category: 'configuration',
                importance: 3,
                scope: 'project',
                confidence: 0.85,
              }),
            );
          }
          const report = await runConsolidation({
            context: h.context,
            projectDir: h.projectDir,
            config: h.config,
            state: h.state,
            statePath: h.statePath,
            taskContext: null,
          });
          check(!report.failure, `merge run reported a failure: ${report.failure}`);
          check(report.updated === 1, `the MERGE verdict must still write, got updated=${report.updated}`);
          const survivors = (await h.memory.getAllMemories()).map((e) => statementOf(e));
          check(
            survivors.every((s) => !s.includes(SECRET)),
            'a secret in action.content leaked into the memory store',
          );
          check(
            survivors.some((s) => s === `${statement} (copy 0)` || s === `${statement} (copy 1)`),
            'the MERGE must fall back to the primary statement when the model content carries a secret',
          );
        } finally {
          await h.close();
        }
      }

      // Part D - deterministic dedup must report removals per project, so the
      // command handler can book them into the right per-project counter
      // (dedup scans every project in one pass).
      {
        const h = await makeHarness((body) => '{"memories":[]}');
        const other = '/tmp/engram-other-project';
        try {
          for (const [dir, tag] of [
            [h.projectDir, 'a'],
            [other, 'b'],
          ] as const) {
            for (let i = 0; i < 2; i++) {
              await h.memory.storeMemory(
                dir,
                `dup-${tag}-${i}`,
                'code-pattern',
                encodeForHarness(`The ${tag} pipeline caches its docker layers for an hour.`, {
                  category: 'configuration',
                  importance: 3,
                  scope: 'project',
                  confidence: 0.9,
                }),
              );
            }
          }
          const report = await deterministicDedup(h.memory);
          check(report.removed === 2, `dedup must remove 2 duplicates, removed ${report.removed}`);
          check(
            report.removedByProject[h.projectDir] === 1 && report.removedByProject[other] === 1,
            `removals must be attributed per project, got ${JSON.stringify(report.removedByProject)}`,
          );
        } finally {
          await h.close();
        }
      }
    },
  );
}

/** Local wrapper so the harness does not import store.ts metaForNew types. */
function encodeForHarness(
  statement: string,
  meta: { category: 'configuration' | 'convention'; importance: number; scope: 'project' | 'global'; confidence: number },
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
