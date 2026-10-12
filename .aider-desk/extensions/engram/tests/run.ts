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
 * Scenario 16 covers the hardening regressions: a single failed
 * consolidation batch must not abort the run (the surviving batches are
 * reviewed, the round counter is kept so the skipped one is retried), and the
 * secret redaction must not treat pure-hex hashes (git SHA-1/SHA-256) as
 * tokens, while still redacting mixed high-entropy blobs. Scenario 17 covers
 * the activity surface: the status-bar badge and its data/actions, a
 * task-scoped feed that a real extraction round lands in, the task-log dump,
 * a command that must not record itself into the feed it prints, and an
 * endpoint that must reach the viewer with its credentials stripped.
 * Scenario 18 covers "spend tokens only while the task runs": subagent,
 * compaction and conflict-resolution agent runs must not trigger extraction or
 * reach the secondary LLM, while a real task turn still does. Scenario 19
 * covers the consolidation cooldown: a backlog left over the interval must not
 * retry an automatic consolidation on every subsequent round.
 * Scenario 20 covers the "what is the secondary LLM actually putting into my
 * Memory, and what is it reading back out" surface: statements are absent from
 * the feed and from the memories view while `ui.show_content` is off, appear
 * once it is on, and reach the viewer secret-redacted in both cases.
 * Scenario 21 covers the automatic-consolidation backoff: a project with
 * several no-progress runs behind it must not consolidate again inside the
 * grown cooldown (the flat 10 minutes re-ran it every time), while the same
 * 11-minute-old attempt without a streak still does. Scenario 22 covers how
 * that streak is recorded: a run that rewrote nothing persists a stall, and a
 * later productive manual run clears it.
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

import type {
  AgentFinishedEvent,
  ContextMessage,
  ImportantRemindersEvent,
  PromptStartedEvent,
} from '@aiderdesk/extensions';

import { aliasFor, buildAliasTable, resolveAlias } from '../src/aliases';
import { safeEndpoint, sanitizeStatements, type ActivityEntry } from '../src/activity';
import { hasAgentOverride, mergeConfig, resolveConfig, type EngramConfig } from '../src/config';
import { logger } from '../src/logger';
import { runExtraction, type ExtractionReport } from '../src/extraction';
import { runConsolidation } from '../src/consolidation';
import { lexicalOverlap, retrieveForPrompt } from '../src/retrieval';
import { looksSecret, redactSecrets } from '../src/privacy';
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

      // Part C - the lexical overlap gate (retrieval.min_overlap, default 1).
      // The host's global memory.maxDistance (shipped default 1.5 on 0..2
      // cosine) is permissive enough that a function word alone ("which") gets
      // a statement natively retrieved; the gate must cut exactly that.
      check(
        lexicalOverlap('which cluster does the deployment run on?', 'The deployment runs on the green cluster.') >= 2,
        'content words shared by prompt and statement must count towards overlap',
      );
      check(
        lexicalOverlap('which cluster does the deployment run on?', 'Ask which region the account was created in before scheduling.') === 0,
        'a function word ("which") shared alone must never count as overlap',
      );
      check(
        lexicalOverlap('come configurare nginx senza password', 'Nginx gira in container con la porta 8080 esposta.') >= 1,
        'Italian prompts must share topic words with Italian statements',
      );
      await h.memory.storeMemory(
        h.projectDir,
        'off-0',
        'code-pattern',
        encodeForHarness('Ask which region the account was created in before scheduling.', {
          category: 'configuration',
          importance: 5,
          scope: 'project',
          confidence: 0.9,
        }),
      );
      // The off-topic statement is importance 5 and natively retrieved (the
      // mock scores "which" as a shared token), so only the overlap gate can
      // be what drops it.
      const gated = await retrieveForPrompt(h.context, h.projectDir, 'which cluster does the deployment run on?', h.config);
      check(
        gated.block !== null && gated.count === 1,
        `the gate must keep only the statement sharing content words, got count=${gated.count}`,
      );
      check(!/scheduling/.test(gated.block ?? ''), 'the function-word-only match must be dropped');
      const inclusive = { ...h.config, retrieval: { ...h.config.retrieval, min_overlap: 0 } };
      const all = await retrieveForPrompt(h.context, h.projectDir, 'which cluster does the deployment run on?', inclusive);
      check(all.count === 2, `min_overlap = 0 must disable the gate and inject both hits, got count=${all.count}`);
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
          'memory:activity',
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

  // ------------------------------------------------------------- scenario 16
  await scenario(
    '16. A failed consolidation batch is skipped and not fatal, and pure-hex hashes are not treated as secrets',
    async () => {
      // Part B - privacy: the entropy-blob rule must not fire on pure
      // hexadecimal tokens. A 40-64 char hex string is a git/object hash or a
      // digest, which appears in coding transcripts constantly; the pre-fix
      // rule silently dropped every candidate quoting a full commit SHA
      // (looksSecret) and mangled the transcript sent to the secondary LLM.
      {
        const sha1 = '124fc07a46a760d789635e47c278933cf6b28c2b';
        const sha256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
        check(redactSecrets(`fixed in commit ${sha1}`, true).hits === 0, 'a full SHA-1 must not be redacted');
        check(redactSecrets(`the digest is ${sha256}`, true).hits === 0, 'a SHA-256 digest must not be redacted');
        check(
          looksSecret(`The regression was fixed in commit ${sha1}.`, true) === false,
          'a candidate quoting a commit hash must not be dropped as a secret',
        );
        const mixedBlob = 'AE3xKv9pQm2Rt7Ws5Yz1Bd4Cf6Gh8Jk0Lu3Nv2Xc9Zq1';
        check(
          redactSecrets(`the token is ${mixedBlob}`, true).hits >= 1,
          'a mixed-case 40+ char blob must still be redacted',
        );
        check(
          redactSecrets('Authorization: Bearer abcdefghijklmnop1234', true).hits >= 1,
          'bearer credentials must still be redacted',
        );
        check(redactSecrets('sk-1234567890abcdef12', true).hits >= 1, 'OpenAI-style keys must still be redacted');
      }

      // Part A - consolidation hardening: one batch exhausts its retry ladder
      // against a 503, the run must still review the remaining batch instead
      // of aborting, and the round counter must be kept so the skipped batch
      // is retried on the next trigger. The pre-fix behaviour broke the whole
      // run on the first failure - with a high call-failure rate a multi-batch
      // run almost never completed - while still resetting the counter.
      const big = 'The regression corpus sentence is deliberately long. '.repeat(680); // each ~ 10k est. tokens > batch budget
      let consolidationCalls = 0;
      const scripted = scriptedResponder();
      const h = await makeHarness((body) => {
        if (/consolidat/i.test(systemOf(body))) {
          consolidationCalls += 1;
          if (consolidationCalls <= 3) return { status: 503, message: 'upstream overloaded' };
        }
        return scripted(body);
      });
      try {
        await h.memory.storeMemory(
          h.projectDir,
          'big-a',
          'code-pattern',
          encodeForHarness(`${big} variant alpha.`, {
            category: 'configuration',
            importance: 3,
            scope: 'project',
            confidence: 0.9,
          }),
        );
        await h.memory.storeMemory(
          h.projectDir,
          'big-b',
          'code-pattern',
          encodeForHarness(`${big} variant beta.`, {
            category: 'configuration',
            importance: 3,
            scope: 'project',
            confidence: 0.9,
          }),
        );

        const stats = projectStats(h.state, h.projectDir);
        stats.tasksSinceConsolidation = 20;
        saveState(h.statePath, h.state);

        const report = await runConsolidation({
          context: h.context,
          projectDir: h.projectDir,
          config: h.config,
          state: h.state,
          statePath: h.statePath,
          taskContext: null,
        });

        check(report.scanned === 2, `expected to scan 2 memories, scanned ${report.scanned}`);
        check(report.failedBatches === 1, `exactly one batch must be marked failed, got ${report.failedBatches}`);
        check(
          !report.failure,
          `a partially successful run must not be reported as a whole-run failure, got: ${report.failure}`,
        );
        check(report.kept === 1, `the surviving batch's memory must count as kept, got kept=${report.kept}`);
        check(
          consolidationCalls === 4,
          `the failed batch must consume its 3 retry attempts before the next batch runs, saw ${consolidationCalls} consolidation call(s)`,
        );
        check(
          projectStats(h.state, h.projectDir).tasksSinceConsolidation === 20,
          'the round counter must be kept while at least one batch went unreviewed',
        );
        check(
          typeof projectStats(h.state, h.projectDir).lastConsolidationAt === 'number',
          'a run that processed a batch must still stamp lastConsolidationAt',
        );
        check(
          h.state.totals.llmFailures === 1,
          `one exhausted retry ladder must count as one logical failure, got ${h.state.totals.llmFailures}`,
        );
      } finally {
        await h.close();
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

  // ------------------------------------------------------------- scenario 17
  await scenario(
    '17. Activity surface: a per-task feed the badge reads, UI actions, credential-stripped status, and no self-pollution',
    async () => {
      const h = await makeHarness(scriptedResponder());
      const dir = mkdtempSync(join(tmpdir(), 'engram-activity-'));
      const configPath = join(dir, 'config.json');
      const statePath = join(dir, 'state.json');
      const logMessages: string[] = [];

      writeFileSync(configPath, JSON.stringify(h.config, null, 2), 'utf-8');
      EngramMemoryExtension.testPaths = { configPath, statePath };

      const readEntries = async (
        ext: EngramMemoryExtension,
        ctx: ReturnType<typeof mockExtensionContext>,
        opts: Record<string, unknown>,
      ): Promise<ActivityEntry[]> => {
        const result = (await ext.executeUIExtensionAction('engram-activity', 'activity', [opts], ctx)) as {
          entries: ActivityEntry[];
        };
        return result.entries;
      };

      try {
        const ext = new EngramMemoryExtension();
        const taskCtx = mockExtensionContext(h.memory, h.projectDir, {
          sink: createLogSink(),
          agentProfiles: [{ id: 'local' }],
          taskContext: mockTaskContext({
            id: 'task-activity',
            agentProfile: { id: 'local' },
            logMessages,
            contextMessages: LLAMA_CONVERSATION(),
          }),
        });
        await ext.onLoad(taskCtx);

        // ---- A. The badge is registered and configured to load its data.
        const components = ext.getUIComponents(taskCtx);
        check(components.length === 1, `expected 1 UI component, got ${components.length}`);
        const badge = components[0];
        check(badge.id === 'engram-activity', `unexpected component id ${badge.id}`);
        check(badge.placement === 'task-status-bar-right', `unexpected placement ${badge.placement}`);
        check(typeof badge.jsx === 'string' && badge.jsx.length > 0, 'the badge has no JSX');
        check(badge.loadData === true, 'the badge must load its data via getUIExtensionData');

        // ---- B. The status header is credential-free and carries the counters.
        const status = (await ext.getUIExtensionData('engram-activity', taskCtx)) as {
          version: string;
          memoryAvailable: boolean;
          activity: { size: number; warnings: number; errors: number };
          llm: { endpoint: string };
          project: { dir: string };
        };
        check(typeof status.version === 'string' && status.version.length > 0, 'status must carry a version');
        check(status.memoryAvailable === true, 'status must report that AiderDesk Memory is available');
        check(typeof status.activity.size === 'number', 'status must carry the feed size');
        check(status.project.dir === h.projectDir, 'status must carry the project dir');
        check(
          !safeEndpoint('http://user:secret@host:4000/v1').includes('secret'),
          'safeEndpoint must strip credentials embedded in a base URL',
        );
        check(
          (await ext.getUIExtensionData('some-other-component', taskCtx)) === undefined,
          'an unknown component id must not leak data',
        );

        // ---- C. A real extraction round lands in the task-scoped feed.
        const beforeTask = await readEntries(ext, taskCtx, { scope: 'task', taskId: 'task-activity' });
        const started = (await ext.executeUIExtensionAction('engram-activity', 'extract', [{}], taskCtx)) as {
          started?: boolean;
        };
        check(started.started === true, 'the "Extract now" action must start a round');
        await ext.drainQueues();
        const afterTask = await readEntries(ext, taskCtx, { scope: 'task', taskId: 'task-activity' });
        check(afterTask.length > beforeTask.length, 'the extraction round must appear in the task feed');
        check(afterTask.some((e) => e.kind === 'extraction'), 'the feed must contain an extraction entry');
        check(
          afterTask.every((e) => !e.taskId || e.taskId === 'task-activity'),
          'a task-scoped feed must never leak another task',
        );
        check(h.memory.count() >= 1, 'the extraction must have stored at least one memory');

        // ---- D. "To task log" prints a header plus one line per entry.
        const dump = (await ext.executeUIExtensionAction(
          'engram-activity',
          'dump',
          [{ taskId: 'task-activity', scope: 'task' }],
          taskCtx,
        )) as { lines: string[]; count: number };
        check(dump.count > 0, 'the dump must include entries');
        check(
          dump.lines.length === dump.count + 1,
          `the dump must print a header plus one line per entry, got ${dump.lines.length} for ${dump.count}`,
        );
        check(logMessages.some((l) => l.includes('activity:')), 'the dump must reach the task log');

        // ---- E. The command prints the feed WITHOUT recording itself (no loop).
        const feedBeforeCommand = await readEntries(ext, taskCtx, { scope: 'all' });
        await ext.getCommands(taskCtx).find((c) => c.name === 'memory:activity')!.execute([], taskCtx);
        check(logMessages.some((l) => l.includes('activity:')), 'memory:activity must print the feed');
        const feedAfterCommand = await readEntries(ext, taskCtx, { scope: 'all' });
        check(
          feedAfterCommand.length === feedBeforeCommand.length,
          'memory:activity must not add entries about itself to the feed',
        );

        // ---- F. Clear empties the feed.
        const cleared = (await ext.executeUIExtensionAction('engram-activity', 'clear', [{}], taskCtx)) as {
          entries: ActivityEntry[];
        };
        check(Array.isArray(cleared.entries) && cleared.entries.length === 0, 'clear must return an empty feed');
        check((await readEntries(ext, taskCtx, { scope: 'all' })).length === 0, 'the feed must be empty after clear');

        // ---- G. show_activity=false removes the badge entirely.
        writeFileSync(
          configPath,
          JSON.stringify(mergeConfig({ ...h.config, ui: { show_activity: false, activity_limit: 200 } }), null, 2),
          'utf-8',
        );
        const offExt = new EngramMemoryExtension();
        await offExt.onLoad(taskCtx);
        check(offExt.getUIComponents(taskCtx).length === 0, 'show_activity=false must hide the badge');

        // ---- H. A credential-bearing base URL never reaches the viewer.
        writeFileSync(
          configPath,
          JSON.stringify(
            mergeConfig({
              ...h.config,
              secondary_llm: { ...h.config.secondary_llm, base_url: 'http://user:secret@127.0.0.1:9/v1' },
            }),
            null,
            2,
          ),
          'utf-8',
        );
        const credExt = new EngramMemoryExtension();
        await credExt.onLoad(taskCtx);
        const credStatus = (await credExt.getUIExtensionData('engram-activity', taskCtx)) as { llm: { endpoint: string } };
        check(!credStatus.llm.endpoint.includes('secret'), `the endpoint must be credential-free, got ${credStatus.llm.endpoint}`);
      } finally {
        EngramMemoryExtension.testPaths = null;
        rmSync(dir, { recursive: true, force: true });
        await h.close();
      }
    },
  );

  // ------------------------------------------------------------- scenario 18
  await scenario(
    '18. Tokens are spent only on task turns: subagent/compaction/conflict runs never extract',
    async () => {
      const h = await makeHarness(scriptedResponder());
      const dir = mkdtempSync(join(tmpdir(), 'engram-mode-'));
      const configPath = join(dir, 'config.json');
      const statePath = join(dir, 'state.json');
      const logMessages: string[] = [];

      writeFileSync(configPath, JSON.stringify(h.config, null, 2), 'utf-8');
      EngramMemoryExtension.testPaths = { configPath, statePath };

      try {
        const ext = new EngramMemoryExtension();
        const sink = createLogSink();
        const ctx = mockExtensionContext(h.memory, h.projectDir, {
          sink,
          agentProfiles: [{ id: 'local' }],
          taskContext: mockTaskContext({
            id: 'task-mode',
            agentProfile: { id: 'local' },
            logMessages,
            contextMessages: LLAMA_CONVERSATION(),
          }),
        });
        await ext.onLoad(ctx);

        const finished = (mode: string): AgentFinishedEvent =>
          ({ mode, aborted: false, contextMessages: LLAMA_CONVERSATION(), resultMessages: [] }) as AgentFinishedEvent;

        // Every internal run AiderDesk dispatches through onAgentFinished must
        // be ignored: no secondary-LLM call, no stored memory, no queue work.
        const callsBefore = h.server.requests.length;
        for (const mode of ['subagent', 'compact-conversation', 'conflict-resolution']) {
          await ext.onAgentFinished(finished(mode), ctx);
        }
        await ext.drainQueues();
        check(
          h.server.requests.length === callsBefore,
          `internal agent modes must not reach the secondary LLM (${h.server.requests.length - callsBefore} calls)`,
        );
        check(h.memory.count() === 0, `internal agent modes must store nothing, got ${h.memory.count()}`);
        check(
          sink.lines.some((l) => l.message.includes('skipping subagent run')),
          `the skip must be visible in the log, got: ${sink.lines.map((l) => l.message).join(' | ')}`,
        );

        // A real task turn still extracts and stores.
        await ext.onAgentFinished(finished('agent'), ctx);
        await ext.drainQueues();
        check(h.server.requests.length > callsBefore, 'a real task turn must contact the secondary LLM');
        check(h.memory.count() === 1, `a real task turn must store a memory, got ${h.memory.count()}`);

        // Aborted runs are skipped regardless of mode (unchanged behaviour).
        const callsAfterTurn = h.server.requests.length;
        await ext.onAgentFinished(
          { mode: 'agent', aborted: true, contextMessages: LLAMA_CONVERSATION(), resultMessages: [] } as AgentFinishedEvent,
          ctx,
        );
        await ext.drainQueues();
        check(
          h.server.requests.length === callsAfterTurn,
          'an aborted run must not contact the secondary LLM',
        );

        await ext.onUnload();
      } finally {
        EngramMemoryExtension.testPaths = null;
        rmSync(dir, { recursive: true, force: true });
        await h.close();
      }
    },
  );

  // ------------------------------------------------------------- scenario 19
  await scenario(
    '19. Automatic consolidation is rate-limited: a failed backlog cannot retry on every round',
    async () => {
      const failingConsolidation: Responder = (body) => {
        const system = body.messages?.find((m) => m.role === 'system')?.content ?? '';
        if (/consolidat/i.test(system)) return { status: 500, message: 'mock consolidation outage' };
        return scriptedResponder()(body);
      };
      const h = await makeHarness(failingConsolidation);
      const dir = mkdtempSync(join(tmpdir(), 'engram-cooldown-'));
      const configPath = join(dir, 'config.json');
      const statePath = join(dir, 'state.json');
      const logMessages: string[] = [];

      // interval_tasks 2 => threshold 2, so the seeded counter is already over.
      writeFileSync(
        configPath,
        JSON.stringify({ ...h.config, consolidation: { enabled: true, interval_tasks: 2, safe_mode: true } }, null, 2),
        'utf-8',
      );
      EngramMemoryExtension.testPaths = { configPath, statePath };

      const isConsolidationCall = (req: { messages?: { role: string; content: string }[] }): boolean =>
        /consolidat/i.test(systemOf(req));

      try {
        // Two managed memories, so consolidation has real work to send to the
        // LLM (fewer than 2 targets short-circuits without a model call).
        for (let i = 0; i < 2; i++) {
          h.memory.seed(
            randomUUID(),
            h.projectDir,
            'code-pattern',
            encodeForHarness(`The build pipeline ${i} caches its docker layers for an hour.`, {
              category: 'configuration',
              importance: 4,
              scope: 'project',
              confidence: 0.9,
            }),
          );
        }
        // The extension loads its own state at construction, so the backlog
        // counter must be on disk BEFORE it is built.
        const seeded = loadState(statePath);
        projectStats(seeded, h.projectDir).tasksSinceConsolidation = 5;
        saveState(statePath, seeded);

        const ext = new EngramMemoryExtension();
        const ctx = mockExtensionContext(h.memory, h.projectDir, {
          sink: createLogSink(),
          agentProfiles: [{ id: 'local' }],
          taskContext: mockTaskContext({
            id: 'task-cooldown',
            agentProfile: { id: 'local' },
            logMessages,
            contextMessages: LLAMA_CONVERSATION(),
          }),
        });
        await ext.onLoad(ctx);
        const finished = (): AgentFinishedEvent =>
          ({ mode: 'agent', aborted: false, contextMessages: LLAMA_CONVERSATION(), resultMessages: [] }) as AgentFinishedEvent;

        // Round 1: over the interval, so a consolidation is attempted - and
        // every batch fails, which is exactly the state that keeps the counter
        // above the interval so the next trigger retries the backlog.
        await ext.onAgentFinished(finished(), ctx);
        await ext.drainQueues();
        const afterFirst = h.server.requests.filter(isConsolidationCall).length;
        check(afterFirst >= 1, `the first over-threshold round must attempt a consolidation (calls ${h.server.requests.length})`);
        check(
          projectStats(loadState(statePath), h.projectDir).tasksSinceConsolidation >= 2,
          'a run whose batches all failed must keep the round counter above the interval',
        );

        // Round 2: still over the interval, inside the cooldown. Without the
        // cooldown this is where the production retry storm came from.
        await ext.onAgentFinished(finished(), ctx);
        await ext.drainQueues();
        const afterSecond = h.server.requests.filter(isConsolidationCall).length;
        check(
          afterSecond === afterFirst,
          `a second round inside the cooldown must not consolidate again (${afterSecond - afterFirst} extra call(s))`,
        );

        await ext.onUnload();
      } finally {
        EngramMemoryExtension.testPaths = null;
        rmSync(dir, { recursive: true, force: true });
        await h.close();
      }
    },
  );

  // ------------------------------------------------------------- scenario 20
  await scenario(
    '20. Statements surface: what the LLM wrote and what it injected, opt-in and secret-redacted',
    async () => {
      const h = await makeHarness(scriptedResponder());
      const dir = mkdtempSync(join(tmpdir(), 'engram-content-'));
      const configPath = join(dir, 'config.json');
      const statePath = join(dir, 'state.json');
      const logMessages: string[] = [];

      const writeConfig = (ui: Record<string, unknown>): void => {
        writeFileSync(
          configPath,
          JSON.stringify(
            mergeConfig({ ...h.config, ui: { show_activity: true, activity_limit: 200, ...ui } }),
            null,
            2,
          ),
          'utf-8',
        );
      };

      const readEntries = async (
        ext: EngramMemoryExtension,
        ctx: ReturnType<typeof mockExtensionContext>,
        opts: Record<string, unknown>,
      ): Promise<ActivityEntry[]> => {
        const result = (await ext.executeUIExtensionAction('engram-activity', 'activity', [opts], ctx)) as {
          entries: ActivityEntry[];
        };
        return result.entries;
      };

      interface MemoriesRow {
        id: string;
        scope: 'global' | 'project';
        category: string;
        importance: number;
        statement?: string;
      }
      const readMemories = async (
        ext: EngramMemoryExtension,
        ctx: ReturnType<typeof mockExtensionContext>,
      ): Promise<{ items: MemoriesRow[]; total: number; showContent: boolean }> =>
        (await ext.executeUIExtensionAction('engram-activity', 'memories', [{ limit: 200 }], ctx)) as {
          items: MemoriesRow[];
          total: number;
          showContent: boolean;
        };

      /** A secret a model might write into a memory, to prove the viewer redacts it. */
      const SECRET = 'sk-FAKESECRETKEY000';

      /** The conversation the "prefers local models" rule of the mock model reacts to. */
      const prefConversation = (): ContextMessage[] => [
        userMsg('Standing note: the user prefers local models over any cloud API.'),
        assistantMsg('Understood - inference stays on the local llama-server, no cloud calls.'),
        toolMsg(['read_file']),
        assistantMsg('Noted: local models only.'),
      ];

      try {
        // ---- A. Default: the viewer shows counts, never statements.
        writeConfig({});
        EngramMemoryExtension.testPaths = { configPath, statePath };

        const offExt = new EngramMemoryExtension();
        const offCtx = mockExtensionContext(h.memory, h.projectDir, {
          sink: createLogSink(),
          agentProfiles: [{ id: 'local' }],
          taskContext: mockTaskContext({
            id: 'task-content-off',
            agentProfile: { id: 'local' },
            logMessages,
            contextMessages: LLAMA_CONVERSATION(),
          }),
        });
        await offExt.onLoad(offCtx);

        const offStatus = (await offExt.getUIExtensionData('engram-activity', offCtx)) as {
          ui: { showContent: boolean };
        };
        check(offStatus.ui.showContent === false, 'show_content must default to off');

        const offRun = (await offExt.executeUIExtensionAction('engram-activity', 'extract', [{}], offCtx)) as {
          started?: boolean;
        };
        check(offRun.started === true, 'the "Extract now" action must start a round');
        await offExt.drainQueues();
        check(h.memory.count() >= 1, `the round must have stored a memory, got ${h.memory.count()}`);

        const offFeed = await readEntries(offExt, offCtx, { scope: 'task', taskId: 'task-content-off' });
        const offExtraction = offFeed.filter((e) => e.kind === 'extraction');
        check(offExtraction.length >= 1, 'the round must record an extraction entry');
        check(
          offExtraction.every((e) => e.content === undefined),
          'with show_content off the feed must carry no statements',
        );

        const offList = await readMemories(offExt, offCtx);
        check(offList.showContent === false, 'the memories view must report statements as hidden');
        check(offList.total >= 1, `the memories view must list the stored memory, got ${offList.total}`);
        check(
          offList.items.every((item) => item.statement === undefined),
          'with show_content off the memories view must return no statements',
        );

        const offDump = (await offExt.executeUIExtensionAction(
          'engram-activity',
          'dump',
          [{ scope: 'all' }],
          offCtx,
        )) as { lines: string[]; count: number };
        check(
          offDump.lines.length === offDump.count + 1,
          `with show_content off the dump must print one line per entry, got ${offDump.lines.length} for ${offDump.count}`,
        );
        await offExt.onUnload();

        // ---- B. show_content on: the writes, the injections and the store rows are listed.
        writeConfig({ show_content: true });
        const ext = new EngramMemoryExtension();
        const sink = createLogSink();
        const ctx = mockExtensionContext(h.memory, h.projectDir, {
          sink,
          agentProfiles: [{ id: 'local' }],
          taskContext: mockTaskContext({
            id: 'task-content-on',
            agentProfile: { id: 'local' },
            logMessages,
            contextMessages: prefConversation(),
          }),
        });
        await ext.onLoad(ctx);

        const onStatus = (await ext.getUIExtensionData('engram-activity', ctx)) as {
          ui: { showContent: boolean };
        };
        check(onStatus.ui.showContent === true, 'show_content must reach the viewer status');

        const onRun = (await ext.executeUIExtensionAction('engram-activity', 'extract', [{}], ctx)) as {
          started?: boolean;
        };
        check(onRun.started === true, 'the "Extract now" action must start a round');
        await ext.drainQueues();

        const onFeed = await readEntries(ext, ctx, { scope: 'task', taskId: 'task-content-on' });
        const written = onFeed.filter((e) => e.kind === 'extraction' && (e.content?.length ?? 0) > 0);
        check(written.length >= 1, 'an extraction entry must carry the statements it wrote');
        check(
          written.some((e) => (e.content ?? []).some((line) => line.startsWith('stored: '))),
          `the write list must name the stored statement, got ${JSON.stringify(written.map((e) => e.content))}`,
        );
        check(
          written.some((e) => (e.content ?? []).some((line) => /running models locally/i.test(line))),
          'the stored statement must be the fact the model extracted',
        );

        // ---- C. Retrieval reports what it injected - with secrets redacted.
        // A memory that already carries a secret (written before redaction
        // existed, or by another tool): the viewer must never show the raw key.
        h.memory.seed(
          randomUUID(),
          h.projectDir,
          'code-pattern',
          encodeForHarness(`The deploy token is ${SECRET} and the deployment runs on the green cluster.`, {
            category: 'configuration',
            importance: 4,
            scope: 'project',
            confidence: 0.9,
          }),
        );

        await ext.onPromptStarted(
          { prompt: 'which deploy token does the deployment on the green cluster use?', mode: 'agent' } as unknown as PromptStartedEvent,
          ctx,
        );
        const injected = (await ext.onImportantReminders(
          { remindersContent: '', agentProfile: { id: 'local' } } as unknown as ImportantRemindersEvent,
          ctx,
        )) as Partial<ImportantRemindersEvent> | undefined;
        check(
          /<engram-memory-context>/.test(injected?.remindersContent ?? ''),
          'the reminder hook must inject the memory block',
        );

        const afterInject = await readEntries(ext, ctx, { scope: 'task', taskId: 'task-content-on' });
        const injectEntries = afterInject.filter((e) => e.kind === 'retrieval' && (e.content?.length ?? 0) > 0);
        check(injectEntries.length >= 1, 'the retrieval entry must carry the statements it injected');
        check(
          injectEntries.some((e) => (e.content ?? []).some((line) => /green cluster/i.test(line))),
          `the injected statement must be listed, got ${JSON.stringify(injectEntries.map((e) => e.content))}`,
        );
        check(
          injectEntries.some((e) => (e.content ?? []).some((line) => line.includes('[REDACTED]'))),
          'an injected secret must be redacted in the viewer',
        );
        check(
          JSON.stringify(afterInject).includes(SECRET) === false,
          'the raw secret must never reach the activity feed',
        );

        // ---- D. The memories view lists the store rows, statements included.
        const onList = await readMemories(ext, ctx);
        check(onList.showContent === true, 'the memories view must report statements as shown');
        check(onList.total >= 2, `the memories view must list every managed row, got ${onList.total}`);
        check(
          onList.items.every((item) => typeof item.statement === 'string' && item.statement.length > 0),
          'with show_content on every row must carry its statement',
        );
        check(
          onList.items.every((item) => item.scope === 'project' || item.scope === 'global'),
          'rows must be scoped',
        );
        check(
          onList.items.some((item) => (item.statement ?? '').includes('[REDACTED]')),
          'the memories view must redact a stored secret',
        );
        check(
          onList.items.every((item) => (item.statement ?? '').includes(SECRET) === false),
          'the memories view must never return a raw secret',
        );
        check(
          onList.items.every((item) => item.importance >= 1),
          'rows must carry their importance',
        );

        // ---- E. The task-log dump prints the statements too.
        const onDump = (await ext.executeUIExtensionAction(
          'engram-activity',
          'dump',
          [{ scope: 'all' }],
          ctx,
        )) as { lines: string[]; count: number };
        check(
          onDump.lines.length > onDump.count + 1,
          `with show_content on the dump must print the statements as extra lines, got ${onDump.lines.length} for ${onDump.count}`,
        );
        check(
          onDump.lines.some((line) => line.includes('stored: ')),
          'the dump must print the written statements',
        );
        check(
          onDump.lines.every((line) => line.includes(SECRET) === false),
          'the dump must never print a raw secret',
        );

        // ---- F. sanitizeStatements: redaction is not optional, and the caps hold.
        const cleaned = sanitizeStatements([
          `password: hunter2keep it safe`,
          '   spaced   out   statement   ',
          '',
          'x'.repeat(400),
          ...Array.from({ length: 40 }, (_, i) => `statement ${i}`),
        ]);
        check(cleaned.length <= 12, `sanitizeStatements must cap the list, got ${cleaned.length}`);
        check(
          cleaned.every((line) => line.length <= 220),
          `sanitizeStatements must cap each statement, got ${Math.max(...cleaned.map((l) => l.length))}`,
        );
        check(cleaned.every((line) => line.includes('  ') === false), 'statements must be whitespace-collapsed');
        check(cleaned[0].includes('hunter2keep') === false, 'a password assignment must be redacted');

        await ext.onUnload();
      } finally {
        EngramMemoryExtension.testPaths = null;
        rmSync(dir, { recursive: true, force: true });
        await h.close();
      }
    },
  );

  // ------------------------------------------------------------- scenario 21
  await scenario(
    '21. A no-progress backlog grows the consolidation cooldown instead of re-running every 10 minutes',
    async () => {
      const h = await makeHarness(scriptedResponder());
      const dir = mkdtempSync(join(tmpdir(), 'engram-backoff-'));
      const configPath = join(dir, 'config.json');
      const statePath = join(dir, 'state.json');
      const logMessages: string[] = [];

      // interval_tasks 2 => threshold 2, so the seeded counter is already over.
      writeFileSync(
        configPath,
        JSON.stringify({ ...h.config, consolidation: { enabled: true, interval_tasks: 2, safe_mode: true } }, null, 2),
        'utf-8',
      );
      EngramMemoryExtension.testPaths = { configPath, statePath };

      const isConsolidationCall = (req: { messages?: { role: string; content: string }[] }): boolean =>
        /consolidat/i.test(systemOf(req));

      const seedMemories = (): void => {
        for (let i = 0; i < 2; i++) {
          h.memory.seed(
            randomUUID(),
            h.projectDir,
            'code-pattern',
            encodeForHarness(`The staging cluster ${i} keeps its docker images for an hour.`, {
              category: 'configuration',
              importance: 4,
              scope: 'project',
              confidence: 0.9,
            }),
          );
        }
      };

      const build = async (
        taskId: string,
      ): Promise<{ ext: EngramMemoryExtension; ctx: ReturnType<typeof mockExtensionContext> }> => {
        const ext = new EngramMemoryExtension();
        const ctx = mockExtensionContext(h.memory, h.projectDir, {
          sink: createLogSink(),
          agentProfiles: [{ id: 'local' }],
          taskContext: mockTaskContext({
            id: taskId,
            agentProfile: { id: 'local' },
            logMessages,
            contextMessages: LLAMA_CONVERSATION(),
          }),
        });
        await ext.onLoad(ctx);
        return { ext, ctx };
      };

      const finished = (): AgentFinishedEvent =>
        ({ mode: 'agent', aborted: false, contextMessages: LLAMA_CONVERSATION(), resultMessages: [] }) as AgentFinishedEvent;

      const seedBacklog = (stalls: number): void => {
        const state = loadState(statePath);
        const stats = projectStats(state, h.projectDir);
        stats.tasksSinceConsolidation = 5;
        stats.consolidationStalls = stalls;
        // 11 minutes old: past the flat 10-minute base, inside any grown window.
        stats.lastConsolidationAt = Date.now() - 11 * 60 * 1000;
        saveState(statePath, state);
      };

      try {
        seedMemories();

        // ---- A. Four no-progress runs behind it: the window is 10 * 2^4 = 160
        // min, so the 11-minute-old attempt must NOT run. The old flat cooldown
        // consolidated right here - that is the token-wasting retry storm.
        seedBacklog(4);
        const stalled = await build('task-backoff-stalled');
        await stalled.ext.onAgentFinished(finished(), stalled.ctx);
        await stalled.ext.drainQueues();
        check(
          h.server.requests.filter(isConsolidationCall).length === 0,
          'a project with 4 no-progress runs and an 11-minute-old attempt must not consolidate again',
        );
        const stalledStatus = (await stalled.ext.getUIExtensionData('engram-activity', stalled.ctx)) as {
          gating: { consolidationStalls: number };
        };
        check(
          stalledStatus.gating.consolidationStalls === 4,
          `the viewer must report the 4-run stall streak, got ${stalledStatus.gating.consolidationStalls}`,
        );
        await stalled.ext.onUnload();

        // ---- B. Control: same age, no stall streak. This must consolidate, or
        // A would pass for the wrong reason (nothing ever consolidating).
        seedBacklog(0);
        const fresh = await build('task-backoff-fresh');
        await fresh.ext.onAgentFinished(finished(), fresh.ctx);
        await fresh.ext.drainQueues();
        check(
          h.server.requests.filter(isConsolidationCall).length >= 1,
          'without a stall streak the same 11-minute-old attempt must consolidate (control)',
        );
        await fresh.ext.onUnload();
      } finally {
        EngramMemoryExtension.testPaths = null;
        rmSync(dir, { recursive: true, force: true });
        await h.close();
      }
    },
  );

  // ------------------------------------------------------------- scenario 22
  await scenario(
    '22. A no-progress automatic run persists the stall streak, and a productive manual run clears it',
    async () => {
      // Consolidation answers 503 while the endpoint is "down", so a run
      // reviews nothing; once "up" the same two near-identical memories merge.
      let down = true;
      const outages: Responder = (body) => {
        if (/consolidat/i.test(systemOf(body))) {
          return down ? { status: 503, message: 'mock consolidation outage' } : scriptedResponder()(body);
        }
        return scriptedResponder()(body);
      };
      const h = await makeHarness(outages);
      const dir = mkdtempSync(join(tmpdir(), 'engram-stall-'));
      const configPath = join(dir, 'config.json');
      const statePath = join(dir, 'state.json');
      const logMessages: string[] = [];

      writeFileSync(
        configPath,
        JSON.stringify({ ...h.config, consolidation: { enabled: true, interval_tasks: 2, safe_mode: true } }, null, 2),
        'utf-8',
      );
      EngramMemoryExtension.testPaths = { configPath, statePath };

      try {
        for (let i = 0; i < 2; i++) {
          h.memory.seed(
            randomUUID(),
            h.projectDir,
            'code-pattern',
            encodeForHarness(`The release pipeline ${i} tags every build with a git sha.`, {
              category: 'configuration',
              importance: 4,
              scope: 'project',
              confidence: 0.9,
            }),
          );
        }
        const seeded = loadState(statePath);
        projectStats(seeded, h.projectDir).tasksSinceConsolidation = 5;
        saveState(statePath, seeded);

        const ext = new EngramMemoryExtension();
        const ctx = mockExtensionContext(h.memory, h.projectDir, {
          sink: createLogSink(),
          agentProfiles: [{ id: 'local' }],
          taskContext: mockTaskContext({
            id: 'task-stall',
            agentProfile: { id: 'local' },
            logMessages,
            contextMessages: LLAMA_CONVERSATION(),
          }),
        });
        await ext.onLoad(ctx);
        const finished = (): AgentFinishedEvent =>
          ({ mode: 'agent', aborted: false, contextMessages: LLAMA_CONVERSATION(), resultMessages: [] }) as AgentFinishedEvent;

        // ---- A. The automatic round rewrote nothing (every batch failed), so
        // the streak must be persisted - not just held in memory, which an
        // extension reload would drop.
        await ext.onAgentFinished(finished(), ctx);
        await ext.drainQueues();
        check(
          projectStats(loadState(statePath), h.projectDir).consolidationStalls === 1,
          'a no-progress automatic run must persist a stall streak of 1',
        );

        // ---- B. The endpoint recovers; a manual run is never throttled and
        // merges the duplicates, so it must clear the streak for the next
        // automatic round instead of leaving it armed.
        down = false;
        await ext.getCommands(ctx).find((c) => c.name === 'memory:consolidate')!.execute([], ctx);
        await ext.drainQueues();
        check(
          projectStats(loadState(statePath), h.projectDir).consolidationStalls === 0,
          'a manual run that merged memories must clear the stall streak',
        );
        await ext.onUnload();
      } finally {
        EngramMemoryExtension.testPaths = null;
        rmSync(dir, { recursive: true, force: true });
        await h.close();
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
