/**
 * Live end-to-end test against a REAL OpenAI-compatible endpoint.
 *
 * Unlike tests/run.ts (offline, scripted mock server), this file drives the
 * real pipeline against a real LLM:
 *   PART 1  - HTTP transport: probe() + raw chat() JSON round-trip, then full
 *             runExtraction (2 rounds) / retrieveForPrompt / runConsolidation
 *             with real model output. Loose assertions: a real model may
 *             phrase things differently, so only pipeline-level invariants
 *             are asserted (JSON parses, something is stored, nothing fails).
 *   PART 2  - 'aiderdesk' transport: TaskContext.generateText receives a
 *             canned responder (mock-server.ts) and must be called with the
 *             configured model id; extraction must store through it.
 *   PART 3  - command callability: a COLD INSTALL of the extension into a
 *             temp directory (exactly like ~/.aider-desk/extensions/engram),
 *             loaded through jiti, then every /memory:* command executed for
 *             real against the live endpoint, plus the settings round-trip
 *             (global and per-agent), the reminder-injection path and onUnload.
 *
 * Secrets never touch the repository: endpoint, key and model come from
 * environment variables and are written only to a temp config.json that is
 * deleted after the run (kept on failure for debugging).
 *
 * Usage:
 *   ENGRAM_LIVE_URL=http://host:4000/v1 \
 *   ENGRAM_LIVE_KEY=sk-... \
 *   ENGRAM_LIVE_MODEL=synthetic/syn:small:text \
 *   node tests/live.mjs
 */
import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { z } from 'zod';
import type { ContextMessage, ExtensionContext, TaskContext } from '@aiderdesk/extensions';

import { mergeConfig, type EngramConfig } from '../src/config';
import { chat, probe } from '../src/llm';
import { parseStructured } from '../src/json';
import { runExtraction } from '../src/extraction';
import { runConsolidation } from '../src/consolidation';
import { retrieveForPrompt } from '../src/retrieval';
import { loadState } from '../src/state';
import { logger } from '../src/logger';
import { MockMemoryContext } from './mock-memory';
import { createLogSink, mockExtensionContext, mockTaskContext } from './mock-context';
import { scriptedResponder } from './mock-server';

const here = dirname(fileURLToPath(import.meta.url));
const extRoot = join(here, '..');

const LIVE_URL = process.env.ENGRAM_LIVE_URL ?? 'http://192.168.1.13:4000/v1';
const LIVE_MODEL = process.env.ENGRAM_LIVE_MODEL ?? 'synthetic/syn:small:text';
const LIVE_KEY = process.env.ENGRAM_LIVE_KEY ?? '';

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(fn: () => boolean, timeoutMs: number, what: string): Promise<void> {
  const start = Date.now();
  while (!fn()) {
    if (Date.now() - start > timeoutMs) {
      throw new Error(`timeout after ${timeoutMs} ms waiting for: ${what}`);
    }
    await sleep(500);
  }
}

function step(name: string): void {
  console.log(`\n=== ${name} ===`);
}

/** Structural view of the extension class used by PART 3. */
interface LiveExt {
  onLoad(ctx: ExtensionContext): Promise<void>;
  onUnload(): Promise<void>;
  getCommands(ctx: ExtensionContext): {
    name: string;
    execute: (args: string[], ctx: ExtensionContext) => Promise<void>;
  }[];
  onPromptStarted(event: { prompt?: string }, ctx: ExtensionContext): Promise<void>;
  onImportantReminders(
    event: { remindersContent?: string },
    ctx: ExtensionContext,
  ): Promise<{ remindersContent?: string } | void>;
  saveConfigData(data: unknown, ctx: ExtensionContext): Promise<unknown>;
  getConfigData(ctx: ExtensionContext): Promise<unknown>;
  getConfigComponent(ctx: ExtensionContext): string | undefined;
}

function liveConfig(secondaryLlmOverrides: Record<string, unknown> = {}): EngramConfig {
  return mergeConfig({
    secondary_llm: {
      transport: 'http',
      base_url: LIVE_URL,
      api_key: LIVE_KEY,
      model: LIVE_MODEL,
      temperature: 0.1,
      max_tokens: 4096,
      timeout_ms: 90000,
      ...secondaryLlmOverrides,
    },
    logging: { level: 'debug' },
  });
}

let seq = 0;
function userMsg(text: string): ContextMessage {
  return { id: `lm${seq++}`, role: 'user', content: text } as unknown as ContextMessage;
}
function assistantMsg(text: string): ContextMessage {
  return { id: `lm${seq++}`, role: 'assistant', content: text } as unknown as ContextMessage;
}

const TRANSCRIPT_A: ContextMessage[] = [
  userMsg(
    'We switched the inference backend from llama.cpp to LiteLLM at http://192.168.1.13:4000/v1 using the model synthetic/syn:small:text. Please remember this.',
  ),
  assistantMsg('Understood - LiteLLM is now the inference backend for this project.'),
  userMsg('Also note this project always uses pnpm instead of npm for package management.'),
  assistantMsg('Noted: pnpm, not npm.'),
];

const TRANSCRIPT_B: ContextMessage[] = [
  userMsg('The team decided to pin all CI runners to Ubuntu 24.04.'),
  assistantMsg('Recorded: CI runners pinned to Ubuntu 24.04.'),
  userMsg('And the database is PostgreSQL 16 with Drizzle ORM for migrations.'),
  assistantMsg('Got it - PostgreSQL 16 with Drizzle ORM.'),
];

async function main(): Promise<void> {
  if (!LIVE_KEY) {
    console.error('ENGRAM_LIVE_KEY is required (keys are passed via env, never stored in files).');
    process.exit(2);
  }
  console.log('Engram live test');
  console.log(`  endpoint: ${LIVE_URL}`);
  console.log(`  model:    ${LIVE_MODEL}`);
  console.log(`  api key:  ${LIVE_KEY.slice(0, 6)}...${LIVE_KEY.slice(-4)} (len ${LIVE_KEY.length})`);

  // ================================================================== PART 1
  step('PART 1 - HTTP transport: probe + raw chat round-trip');
  const cfg1 = liveConfig();
  const probeResult = await probe(cfg1.secondary_llm);
  if (!probeResult.ok) throw new Error(`probe failed: ${probeResult.kind} - ${probeResult.message}`);
  console.log(`PASS probe ok (${probeResult.durationMs} ms)`);

  const round = await chat(cfg1.secondary_llm, {
    system: 'You output strict JSON only.',
    user: 'Return exactly this JSON object, nothing else: {"ok": true, "tool": "engram"}',
    temperature: 0,
    // Reasoning models spend tokens on reasoning_content before the answer;
    // keep the cap generous so the JSON answer is not truncated away.
    maxTokens: 1024,
  });
  if (!round.ok) throw new Error(`chat failed: ${round.kind} - ${round.message}`);
  const roundParsed = parseStructured(z.object({ ok: z.boolean(), tool: z.string() }), round.text);
  if (!roundParsed.ok) {
    throw new Error(`chat JSON round-trip failed to parse: ${roundParsed.errors} (raw: ${round.text.slice(0, 200)})`);
  }
  if (round.promptTokens === undefined) console.log('  note: endpoint did not report token usage');
  console.log(
    `PASS chat round-trip ok (${round.durationMs} ms, tokens ${round.promptTokens ?? '?'}/${round.completionTokens ?? '?'})`,
  );

  step('PART 1b - full extraction pipeline with the real LLM (2 rounds)');
  const memoryA = new MockMemoryContext();
  const sinkA = createLogSink();
  const projectA = '/tmp/engram-live-project-a';
  const ctxA = mockExtensionContext(memoryA, projectA, { sink: sinkA });
  logger.bind(ctxA);
  logger.setConfig(cfg1.logging);
  const statePathA = join(tmpdir(), `engram-live-state-a-${Date.now()}.json`);
  const stateA = loadState(statePathA);

  const rep1 = await runExtraction({
    context: ctxA,
    messages: TRANSCRIPT_A,
    projectDir: projectA,
    taskId: 'task-live-a1',
    config: cfg1,
    state: stateA,
    statePath: statePathA,
  });
  if (rep1.failure) throw new Error(`extraction round 1 failed: ${rep1.failure}`);
  if (rep1.candidates < 1) throw new Error(`extraction round 1 produced no candidates (report: ${JSON.stringify(rep1)})`);
  if (rep1.stored + rep1.updated < 1) {
    throw new Error(`extraction round 1 stored nothing (report: ${JSON.stringify(rep1)})`);
  }
  console.log(
    `PASS extraction round 1: candidates=${rep1.candidates} stored=${rep1.stored} updated=${rep1.updated} dup=${rep1.duplicates} obsolete=${rep1.obsolete}`,
  );

  const rep2 = await runExtraction({
    context: ctxA,
    messages: TRANSCRIPT_B,
    projectDir: projectA,
    taskId: 'task-live-a2',
    config: cfg1,
    state: stateA,
    statePath: statePathA,
  });
  if (rep2.failure) throw new Error(`extraction round 2 failed: ${rep2.failure}`);
  if (rep2.candidates < 1) throw new Error(`extraction round 2 produced no candidates (report: ${JSON.stringify(rep2)})`);
  if (rep2.stored + rep2.updated === 0) {
    console.log(`  note: round 2 produced only duplicates (report: ${JSON.stringify(rep2)})`);
  } else {
    console.log(
      `PASS extraction round 2: candidates=${rep2.candidates} stored=${rep2.stored} updated=${rep2.updated} dup=${rep2.duplicates}`,
    );
  }

  step('PART 1c - retrieval injection');
  const retrieval = await retrieveForPrompt(
    ctxA,
    projectA,
    'which inference backend and package manager does this project use?',
    cfg1,
  );
  if (!retrieval.block) throw new Error('retrieval returned no block');
  if (retrieval.count < 1) throw new Error('retrieval injected 0 memories');
  console.log(`PASS retrieval injected ${retrieval.count} memory(ies)`);

  step('PART 1d - consolidation with the real LLM');
  const cons = await runConsolidation({
    context: ctxA,
    projectDir: projectA,
    config: cfg1,
    state: stateA,
    statePath: statePathA,
  });
  if (cons.failure) throw new Error(`consolidation failed: ${cons.failure}`);
  console.log(
    `PASS consolidation: scanned=${cons.scanned} merged=${cons.merged} updated=${cons.updated} deleted=${cons.deleted} kept=${cons.kept}`,
  );

  // ================================================================== PART 2
  step("PART 2 - 'aiderdesk' transport routes through TaskContext.generateText");
  const memoryC = new MockMemoryContext();
  const cfgC = liveConfig({ transport: 'aiderdesk', model_id: LIVE_MODEL });
  const calledModels: string[] = [];
  const tcC = mockTaskContext({
    id: 'task-live-ad',
    generateText: async (modelId: string, systemPrompt: string, prompt: string) => {
      calledModels.push(modelId);
      const out = scriptedResponder()({
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: prompt },
        ],
      });
      return typeof out === 'string' ? out : null;
    },
  });
  const ctxC = mockExtensionContext(memoryC, '/tmp/engram-live-project-ad', { taskContext: tcC });
  logger.bind(ctxC);
  logger.setConfig(cfgC.logging);
  const statePathC = join(tmpdir(), `engram-live-state-c-${Date.now()}.json`);
  const repC = await runExtraction({
    context: ctxC,
    messages: TRANSCRIPT_A,
    projectDir: '/tmp/engram-live-project-ad',
    taskId: 'task-live-ad',
    config: cfgC,
    state: loadState(statePathC),
    statePath: statePathC,
    taskContext: tcC,
  });
  if (repC.failure) throw new Error(`aiderdesk-transport extraction failed: ${repC.failure}`);
  if (repC.stored < 1) throw new Error(`aiderdesk-transport extraction stored nothing (report: ${JSON.stringify(repC)})`);
  if (calledModels[0] !== LIVE_MODEL) {
    throw new Error(`generateText was called with "${calledModels[0]}" instead of "${LIVE_MODEL}"`);
  }
  console.log(`PASS aiderdesk transport: generateText(${LIVE_MODEL}) -> stored ${repC.stored}`);

  // ================================================================== PART 3
  step('PART 3 - cold install into temp dir + full command callability');
  const tempDir = mkdtempSync(join(tmpdir(), 'engram-live-install-'));
  console.log(`  install dir: ${tempDir}`);
  cpSync(extRoot, tempDir, {
    recursive: true,
    filter: (src) => {
      const base = src.split(/[\\/]/).pop() ?? '';
      if (
        base === 'node_modules' ||
        base === 'tests' ||
        base === 'config.json' ||
        base === 'state.json' ||
        base.endsWith('.md') ||
        base === 'tsconfig.json'
      ) {
        return false;
      }
      return true;
    },
  });
  writeFileSync(join(tempDir, 'config.json'), JSON.stringify(liveConfig(), null, 2), 'utf-8');

  const jiti = (
    globalThis as unknown as {
      __engramJiti?: { import: (p: string, o?: { default?: boolean }) => Promise<unknown> };
    }
  ).__engramJiti;
  if (!jiti) throw new Error('jiti instance not found on globalThis (use tests/live.mjs as the entry point)');
  const mod = await jiti.import(join(tempDir, 'index.ts'));
  const ExtClass = ((mod as { default?: unknown }).default ?? mod) as unknown as new () => LiveExt;
  const ext = new ExtClass();

  const memoryB = new MockMemoryContext();
  const cmdLog: string[] = [];
  const sinkB = createLogSink();
  const taskContext = {
    data: { id: 'task-live-cmd', name: 'Live commands' },
    addLogMessage: (level: string, msg: string) => {
      cmdLog.push(`[${level}] ${msg}`);
    },
    getContextMessages: async () => TRANSCRIPT_A,
  } as unknown as TaskContext;
  const ctxB = mockExtensionContext(memoryB, '/tmp/engram-live-project-b', { taskContext, sink: sinkB });

  await ext.onLoad(ctxB);
  console.log('PASS onLoad');

  const commands = ext.getCommands(ctxB);
  const names = commands.map((c) => c.name).sort();
  const expected = ['memory:clear-project', 'memory:consolidate', 'memory:extract', 'memory:forget', 'memory:stats'];
  if (JSON.stringify(names) !== JSON.stringify(expected)) {
    throw new Error(`unexpected command set: ${names.join(', ')}`);
  }
  console.log(`PASS getCommands: ${names.join(', ')}`);

  const byName = new Map(commands.map((c) => [c.name, c]));
  const run = (name: string, args: string[] = []): Promise<void> => byName.get(name)!.execute(args, ctxB);

  // memory:extract - real LLM through the extension's background queue.
  await run('memory:extract');
  await waitFor(() => memoryB.count() > 0, 240000, 'memory:extract stored at least one memory');
  await sleep(2000); // let the queue finish writing its state
  if (!cmdLog.some((l) => l.includes('extraction started'))) {
    throw new Error('memory:extract did not log its start');
  }
  console.log(`PASS memory:extract (memories now: ${memoryB.count()})`);

  // memory:stats
  await run('memory:stats');
  if (!cmdLog.some((l) => l.includes('[Memory] AiderDesk entries:'))) {
    throw new Error('memory:stats did not log entry counts');
  }
  if (!cmdLog.some((l) => l.includes('secondary LLM reachable'))) {
    throw new Error(`memory:stats probe line missing (tail: ${cmdLog.slice(-4).join(' | ')})`);
  }
  console.log('PASS memory:stats');

  // Retrieval through the extension event path.
  await ext.onPromptStarted({ prompt: 'what package manager and inference backend do we use here?' }, ctxB);
  const injected = await ext.onImportantReminders({ remindersContent: 'base' }, ctxB);
  const injectedText = injected && typeof injected === 'object' ? injected.remindersContent ?? '' : '';
  if (!injectedText.includes('<engram-memory-context>')) {
    throw new Error(`onImportantReminders did not inject memories (got: ${injectedText.slice(0, 120)})`);
  }
  console.log('PASS onPromptStarted + onImportantReminders injection');

  // memory:forget
  const beforeForget = memoryB.count();
  await run('memory:forget', ['pnpm']);
  if (!cmdLog.some((l) => l.includes('[Memory] forgot:'))) {
    throw new Error(`memory:forget did not delete (tail: ${cmdLog.slice(-3).join(' | ')})`);
  }
  console.log(`PASS memory:forget (${beforeForget} -> ${memoryB.count()})`);

  // memory:consolidate force - real LLM.
  await run('memory:consolidate', ['force']);
  await waitFor(
    () => cmdLog.some((l) => l.includes('[Memory] consolidated ') || l.includes('consolidation failed')),
    240000,
    'memory:consolidate to finish',
  );
  if (cmdLog.some((l) => l.includes('consolidation failed'))) {
    throw new Error(`memory:consolidate failed: ${cmdLog.filter((l) => l.includes('failed')).join(' | ')}`);
  }
  console.log('PASS memory:consolidate force');

  // memory:clear-project must refuse without "confirm".
  const beforeClear = memoryB.count();
  await run('memory:clear-project');
  if (!cmdLog.some((l) => l.includes('refusing to clear'))) {
    throw new Error('memory:clear-project did not refuse without confirm');
  }
  if (memoryB.count() !== beforeClear) {
    throw new Error('memory:clear-project deleted memories without confirm!');
  }
  console.log('PASS memory:clear-project refusal');

  // memory:clear-project with "confirm".
  await run('memory:clear-project', ['confirm']);
  await waitFor(() => memoryB.count() === 0, 30000, 'memory:clear-project confirm emptied the store');
  console.log('PASS memory:clear-project confirm');

  // Settings round-trip.
  const saved = (await ext.saveConfigData(
    { secondary_llm: { base_url: LIVE_URL, model: LIVE_MODEL, api_key: LIVE_KEY } },
    ctxB,
  )) as { secondary_llm?: { base_url?: string; model?: string } };
  if (saved?.secondary_llm?.base_url !== LIVE_URL) throw new Error('saveConfigData did not preserve base_url');
  const reloaded = (await ext.getConfigData(ctxB)) as {
    secondary_llm?: { base_url?: string; model?: string; api_key?: string };
  };
  if (reloaded?.secondary_llm?.base_url !== LIVE_URL || reloaded?.secondary_llm?.model !== LIVE_MODEL) {
    throw new Error('getConfigData round-trip mismatch');
  }
  if (reloaded?.secondary_llm?.api_key !== LIVE_KEY) throw new Error('api_key was lost in the config round-trip');
  console.log('PASS saveConfigData/getConfigData round-trip');

  // Per-agent config: the settings payload carries the project's agent profiles
  // as `_agents`, an override persists, and `_agents` never reaches config.json.
  const ctxAgents = mockExtensionContext(memoryB, '/tmp/engram-live-project-b', {
    taskContext,
    sink: sinkB,
    agentProfiles: [
      { id: 'local', name: 'Local' },
      { id: 'intesa', name: 'Intesa', provider: 'openai-compatible', model: 'engram-secondary' },
    ],
  });
  const panel = (await ext.getConfigData(ctxAgents)) as EngramConfig & { _agents?: { id: string }[] };
  if (!Array.isArray(panel._agents) || panel._agents.map((a) => a.id).join(',') !== 'local,intesa') {
    throw new Error(`_agents missing from the settings payload: ${JSON.stringify(panel._agents)}`);
  }
  // A context with no project context must degrade to the plain global config.
  const plainPanel = (await ext.getConfigData(ctxB)) as EngramConfig & { _agents?: unknown };
  if (plainPanel._agents !== undefined) throw new Error('_agents must be absent when no project context is available');

  const savedAgents = (await ext.saveConfigData(
    { ...panel, agents: { intesa: { extraction: { min_importance: 4 }, secondary_llm: { model: LIVE_MODEL } } } },
    ctxAgents,
  )) as EngramConfig;
  if (savedAgents.agents?.intesa?.extraction?.min_importance !== 4) {
    throw new Error(`the per-agent override was not saved: ${JSON.stringify(savedAgents.agents)}`);
  }
  const persistedConfig = JSON.parse(readFileSync(join(tempDir, 'config.json'), 'utf-8')) as Record<string, unknown>;
  if (persistedConfig._agents !== undefined) throw new Error('_agents leaked into config.json');
  if (
    (persistedConfig.agents as Record<string, { extraction?: { min_importance?: number } }>)?.intesa?.extraction
      ?.min_importance !== 4
  ) {
    throw new Error('config.json lost the per-agent override');
  }
  console.log('PASS per-agent settings (_agents listed, override persisted, _agents not persisted)');

  const jsx = ext.getConfigComponent(ctxB);
  if (typeof jsx !== 'string' || jsx.length < 100) throw new Error('getConfigComponent returned no JSX');
  console.log(`PASS getConfigComponent (${jsx.length} chars)`);

  await ext.onUnload();
  if (!existsSync(join(tempDir, 'state.json'))) throw new Error('onUnload did not persist state.json');
  console.log('PASS onUnload (state.json written)');

  // -------------------------------------------------------------- cleanup
  rmSync(tempDir, { recursive: true, force: true });
  rmSync(statePathA, { force: true });
  rmSync(statePathC, { force: true });
  console.log('\nALL LIVE TESTS PASSED');
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error('\nLIVE TEST FAILED:', err instanceof Error ? err.message : err);
    if (err instanceof Error && err.stack) {
      console.error(err.stack.split('\n').slice(1, 5).join('\n'));
    }
    process.exit(1);
  });
