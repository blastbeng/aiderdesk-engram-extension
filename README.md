# Engram — long-term memory for AiderDesk

**Engram** is an [AiderDesk](https://github.com/hotovo/aider-desk) extension (tested on **v0.81.0**) that adds **automatic long-term memory** to your coding assistant. It extracts durable facts from your conversations — decisions, preferences, constraints, solutions, conventions — stores them in AiderDesk's **native Memory**, and re-injects only the relevant ones into future prompts.

It is powered by a **secondary local LLM** (llama-server, Ollama /v1, LiteLLM, vLLM, LM Studio, OpenRouter — any OpenAI-compatible endpoint) and **never** uses your main model. The main model only ever sees one small block of relevant facts; all the heavy lifting (transcript reading, extraction, deduplication, conflict resolution, consolidation) runs off the critical path on the secondary machine/GPU.

> This is **not** an MCP Memory Server. It is a full AiderDesk extension built on the real 0.81.0 Extension API — every API call used is verified against the `v0.81.0` git tag (see [section 2](#2-verified-extension-api-v0810)).

---

## Features

- **Automatic extraction** at the end of each agent turn (or task/prompt, configurable) — only durable, future-valuable facts are kept ("project uses llama.cpp instead of Ollama" ✅ — "user asked how to install package X" ❌).
- **Structured JSON, validated before writing** — zod schemas + malformed-JSON recovery + one LLM repair round-trip; corrupt data is never stored.
- **LLM deduplication** — each candidate is classified `NEW | DUPLICATE | UPDATE | CONFLICT | OBSOLETE`; updates merge **in place**, never as parallel duplicates.
- **Periodic consolidation** — every N agent turns (default 20), redundant memories are merged; `safe_mode` never deletes.
- **Importance scoring (1–5)** — importance 1 is discarded; higher importance surfaces first.
- **Project vs global scope** — decisions about this project stay with the project; general preferences follow you everywhere.
- **Relevant retrieval only** — ≤ 8 memories injected per prompt, only when relevant to the current prompt; otherwise nothing is injected (zero context pollution).
- **Secret safety** — 17 redaction rules (API keys, AWS, JWT, Bearer, passwords, private keys, high-entropy blobs) applied **before** anything leaves the process.
- **Never blocks the agent** — fire-and-forget serialized queues; if the secondary LLM is offline, AiderDesk keeps working normally.
- **Full UI + commands** — settings panel, `Memory: Extract Now`, `Memory: Consolidate`, `Memory: Show Statistics`, `Memory: Forget`, `Memory: Clear Project Memories`.
- **Dual transport** — direct HTTP to any OpenAI-compatible endpoint (default), or AiderDesk's native `TaskContext.generateText`.
- **Zero runtime dependencies** — plain `fetch`; no build step; TypeScript loaded directly by AiderDesk's jiti loader.

## Repository layout

```
.aider-desk/extensions/engram/   ← the extension itself (copy this folder)
├── index.ts                     Extension: events, commands, settings UI, queues
├── src/                         Pipeline modules (extraction, consolidation, …)
├── tests/                       Offline acceptance harness (7 scenarios)
├── ConfigComponent.jsx          Settings panel
├── package.json / tsconfig.json
└── README.md                    Pointer to this file
```

## Requirements

- AiderDesk **0.81.0 or newer**
- Node.js ≥ 18 (bundled with AiderDesk)
- A secondary LLM behind an OpenAI-compatible endpoint, e.g. on a spare GPU:
  ```bash
  llama-server -m <model>.gguf --host 0.0.0.0 --port 4000 --api-key local
  ```
  Any model works; small instruct models (7–14B) are ideal — temperature 0.1.

## Quick start

```bash
# 1. Copy the extension into AiderDesk's global extensions folder
cp -r .aider-desk/extensions/engram ~/.aider-desk/extensions/engram

# 2. Restart AiderDesk — TS extensions load via jiti, no build, no npm install

# 3. Settings → Extensions → Engram → point base_url at your LLM endpoint
#    (default: http://192.168.1.29:4000/v1, api_key "local", model "small-model")
```

That's it — memory extraction starts at the end of the next agent turn. Full walkthrough: [Installation](#6-installation-10-steps).

---

## 1. What it does, end to end

```
        AiderDesk (main model, e.g. Qwen3.8-Flash-Next)
                    │  sees only the useful memory block
                    ▼
        ┌─────────────────────────────────────────────┐
        │  Engram extension                           │
        │                                             │
        │  Triggers     onAgentFinished / onTaskClosed│
        │  Extraction   bounded transcript + redaction│
        │       │       → validated JSON (zod+repair) │
        │       │       → LLM deduplication           │
        │       ▼                                     │
        │  Native AiderDesk Memory (SQLite + vectors) │
        │       ▲                                     │
        │  Injection    onImportantReminders          │
        │  Consolidation every N turns (merge/dedup,  │
        │               safe_mode)                    │
        └───────────────┬─────────────────────────────┘
                        │ HTTP OpenAI-compatible (transport "http")
                        ▼
        llama-server on the secondary GPU (e.g. RTX 3060 12 GB)
        http://192.168.1.29:4000/v1
```

**Extraction flow** (end of agent turn):

1. Trigger `onAgentFinished` (configurable: `agent_end`, `task_end`, `prompt_end`).
2. Bounded history: the last messages up to `max_input_tokens` (12,000 by default) — **never** the full history. Memory blocks previously injected by Engram are excluded.
3. **Secrets are redacted from the transcript before anything is sent** (17 rules).
4. Nearest existing memories are fetched (dedup corpus, bounded to `max_existing_for_dedup`).
5. Secondary LLM call → JSON `{ memories: [{ content, category, importance, scope, confidence }] }`, zod-validated; malformed JSON gets **one** repair round-trip; if still invalid, nothing is written.
6. Batched LLM classification: `NEW | DUPLICATE | UPDATE | CONFLICT | OBSOLETE` → matching write (UPDATE/CONFLICT merge into the existing memory).
7. Per-project serialized fire-and-forget queue: **the agent never blocks**.

**Token flow vs. plain AiderDesk**:

| Flow | Before | With Engram |
|---|---|---|
| Agent context each turn | Full history + manual repetition | + one small memory block max (`max_memories: 8`, ~1–2 % of context), **only when relevant** |
| Main model spent on memory management | (doesn't exist) | **0 tokens** — everything runs on the secondary LLM |
| Secondary LLM (spare GPU) | idle | ≤ 12k tokens per extraction, called at turn end, off the critical path |
| Long history | re-paid every turn | Durable facts live in native Memory, recovered by vector search |

## 2. Verified Extension API (v0.81.0)

Everything below was verified against the `v0.81.0` git tag of `hotovo/aider-desk` and the installed package (`@aiderdesk/aiderdesk` 0.81.0, `@aiderdesk/extensions` 0.32.0, `dist/index.d.ts`). Nothing is invented.

### APIs the extension uses

| Need | Real API | Reference (d.ts) |
|---|---|---|
| Store / retrieve / edit memories | `context.getMemoryContext()` → `storeMemory(projectId, taskId, type, content)`, `retrieveMemories(projectId, query, limit)`, `getMemory(id)`, `updateMemory(id, content)`, `deleteMemory(id)`, `getAllMemories()` | ~line 2281 |
| Trigger extraction at agent turn end | `onAgentFinished({ mode, aborted, contextMessages, resultMessages })` | ~1322 |
| Trigger extraction at task close | `onTaskClosed({ task })` | ~1260 |
| Trigger extraction at prompt end | `onPromptFinished` | ~1288 |
| Inject relevant memories into the prompt | `onImportantReminders({ agentProfile, remindersContent })` → return `{ remindersContent }`; the returned text is appended to the user message (inside `<ThisIsImportant>`) | ~1351 |
| Bounded conversation history | `context.getTaskContext().getContextMessages()` (typed user/assistant/tool `ContextMessage`s) | ~1492 |
| Manual commands | `getCommands(): CommandDefinition[]` | — |
| Settings panel | `getConfigComponent()` + `getConfigData()` / `saveConfigData()` | — |
| Logging | `context.log(message, 'info' \| 'error' \| 'warn' \| 'debug')` | ~2125 |
| Native secondary-model call (optional) | `TaskContext.generateText(modelId, systemPrompt, prompt): Promise<string \| undefined>` | ~1751 |

### What the API does not offer (and the workarounds used)

1. **`MemoryEntry` has no metadata column** (`{ id, content, type, taskId?, projectId?, timestamp }`, ~line 950). → Attributes (category, importance, scope, confidence, timestamps) are encoded in a **footer** inside `content`: `[mem cat=… imp=… scope=… conf=… ts=… (ut=…)]`. The statement stays first (it dominates the embedding); the footer is stripped before injection and display. Memories whose footer does not decode are **never touched** by consolidation — native AiderDesk memories stay intact.
2. **`retrieveMemories` filters on exact `projectId`** and applies a **global** distance cutoff (`memory.maxDistance`), not a per-call threshold, and returns **no score or distance**. → "Global" scope is implemented with `projectId === ''` (the native Memory default); `min_relevance` is a best-effort client-side hint (importance floor), never a blocker.
3. **No per-message system-prompt hook.** → Injection goes through `onImportantReminders`, the official event-return mechanism; the block is appended to the user message inside a tagged block.
4. **`MemoryEntryType` is not exported** (string enum `'task' | 'user-preference' | 'code-pattern'`). → The native type is derived: `NativeMemoryType = Parameters<MemoryContext['storeMemory']>[2]`, mapped from Engram's 12 categories.
5. **No "agent started" event usable for hot injection.** → The `onImportantReminders` callback covers it; when nothing relevant is retrieved, nothing is injected.

### Pipeline modules

```
engram/
├── index.ts               Extension: events, commands, settings UI, queues
├── src/
│   ├── config.ts          Types + DEFAULT_CONFIG + defensive merge (config.json)
│   ├── llm.ts             Dependency-free OpenAI-compatible client, never throws
│   ├── prompts.ts         Extraction / classification / consolidation / repair prompts
│   ├── extraction.ts      Extraction → validation → dedup → write pipeline
│   ├── consolidation.ts   Periodic consolidation (batches, safe_mode)
│   ├── retrieval.ts       Retrieval + <engram-memory-context> block
│   ├── store.ts           Native Memory access, footers, scopes
│   ├── memory-format.ts   Footer encoding/decoding, native type mapping
│   ├── transcript.ts      Bounded history (token budget)
│   ├── privacy.ts         17 secret-redaction rules
│   ├── json.ts            zod v4 schemas + malformed-JSON recovery
│   ├── state.ts           Statistics (state.json)
│   └── logger.ts          Configurable [Memory] … logging
├── tests/                 Offline harness (7 scenarios, real mock HTTP server)
└── ConfigComponent.jsx    Settings panel
```

## 3. Build

**There is no build step.** AiderDesk 0.81.0 loads TypeScript extensions directly through [jiti](https://github.com/unjs/jiti) (transpiled on the fly); no `tsc`, `esbuild` or bundler is needed to install or run the extension. `package.json` has zero runtime dependencies — the HTTP client is plain `fetch`, and `zod` is provided by AiderDesk's own loader.

For development and verification (on a machine where AiderDesk is installed):

```bash
cd .aider-desk/extensions/engram

# 1) Make zod/jiti resolve for the test harness (mirrors AiderDesk's runtime aliasing).
#    Point the symlink at your AiderDesk install's node_modules:
ln -s /opt/npm/lib/node_modules/@aiderdesk/aiderdesk/node_modules node_modules

# 2) Typecheck (uses AiderDesk's bundled TypeScript; adjust the path to your install):
/opt/npm/lib/node_modules/@aiderdesk/aiderdesk/node_modules/typescript/bin/tsc --noEmit -p tsconfig.json

# 3) Run the offline test harness (7 scenarios, no real LLM needed):
node tests/run.mjs
#    equivalent: ./node_modules/.bin/jiti tests/run.ts
```

Notes:

- `tsconfig.json` and the typecheck command above reference absolute paths from the reference machine (`/opt/npm/lib/node_modules/@aiderdesk/aiderdesk/...`). Adapt them to wherever AiderDesk is installed on your system.
- `node_modules` (the symlink), `config.json` and `state.json` are dev/runtime artifacts and are gitignored.
- The harness runs entirely offline: it starts a real local mock OpenAI-compatible HTTP server (`tests/mock-server.ts`) and a simulated `MemoryContext` (`tests/mock-memory.ts`) with the same `projectId` filtering semantics as the native API.

## 4. Configuration

`config.json` next to the extension, editable via **Settings → Extensions → Engram** (ConfigComponent.jsx). Defaults:

```jsonc
{
  "enabled": true,
  "secondary_llm": {
    "transport": "http",                        // "http" (recommended) or "aiderdesk"
    "model_id": "openai-compatible/engram-secondary", // aiderdesk transport only
    "base_url": "http://192.168.1.29:4000/v1",
    "api_key": "local",                          // llama-server accepts any string
    "model": "small-model",                      // model name served by the endpoint
    "temperature": 0.1,
    "max_tokens": 8192,
    "timeout_ms": 30000
  },
  "extraction": {
    "enabled": true,
    "trigger": "agent_end",       // agent_end | task_end | prompt_end
    "max_messages": 30,           // bound on the analyzed history
    "min_importance": 2,          // importance 1 = not stored
    "max_input_tokens": 12000,
    "max_candidates": 20,
    "max_existing_for_dedup": 12
  },
  "retrieval": {
    "enabled": true,
    "max_memories": 8,
    "min_relevance": 0.65,
    "include_global": true
  },
  "consolidation": {
    "enabled": true,
    "interval_tasks": 20,         // every N agent turns
    "safe_mode": true             // true: never deletes, only merges/updates
  },
  "privacy": { "redact_secrets": true },
  "logging": { "enabled": true, "level": "info" }
}
```

Any OpenAI-compatible URL works: Ollama (`http://host:11434/v1`), LiteLLM, vLLM, LM Studio, OpenRouter. The `aiderdesk` transport goes through `TaskContext.generateText` (model declared in Settings → Providers → OpenAI-compatible); the extension falls back to HTTP when no TaskContext is available.

## 5. Installation (10 steps)

1. **Start llama-server** on the secondary-GPU machine (see Requirements) and verify: `curl http://192.168.1.29:4000/v1/models`.
2. **Copy the extension folder** into AiderDesk's extensions directory:
   ```bash
   cp -r .aider-desk/extensions/engram ~/.aider-desk/extensions/engram
   ```
   (per-project install also works: `<project>/.aider-desk/extensions/engram`).
3. **Restart AiderDesk**. TS extensions are loaded by jiti — no build, no `npm install`.
4. **Check it loaded**: AiderDesk logs show `[Memory] extension loaded` plus a secondary-LLM probe. If Memory is disabled in AiderDesk (Settings → Memory) the extension logs a warning — enable it.
5. **Open Settings → Extensions → Engram** and set `base_url`, `model`, `api_key` if they differ from the defaults.
6. **Pick the extraction trigger** (`agent_end` recommended) and the consolidation interval (`interval_tasks: 20`).
7. **Test the connection**: run the `Memory: Extract Now` command → logs should show `[Memory] extracting memories…` followed by a summary (`N candidates, X new, Y duplicates…`).
8. **Verify injection**: open a task, mention a memorized fact, and look for the `<engram-memory-context>` block in the user message.
9. **Optional**: if you prefer `transport: "aiderdesk"`, declare the endpoint in Settings → Providers (OpenAI-compatible) and set `model_id` as `provider/model`.
10. **Let it run**: consolidation triggers on its own every `interval_tasks` turns; watch `Memory: Show Statistics`.

## 6. Usage

Daily use is automatic — you code, Engram remembers. Manual controls (AiderDesk command palette):

| Command | Effect |
|---|---|
| `Memory: Extract Now` | Immediate extraction on the current task |
| `Memory: Consolidate` | Immediate consolidation (`force` → out of safe_mode, deletion possible) |
| `Memory: Show Statistics` | Per-project stats + totals (LLM calls, failures, stored, updated, duplicates) |
| `Memory: Forget <text>` | Find and delete a specific memory |
| `Memory: Clear Project Memories` confirm | Wipe the current project's memories (confirmation required) |

`[Memory] …` logs: extraction started, candidates found, verdicts (new/duplicate/update), consolidation, LLM failures. No secret ever appears in logs (redaction happens upstream).

## 7. The 6 tests (offline harness included)

The `tests/` folder contains an **offline** harness that runs the real pipeline (extraction, classification, consolidation, redaction, repair, transport) against a **real mock OpenAI-compatible HTTP server** and a simulated **MemoryContext** (same `projectId` filtering semantics as the native API).

```bash
cd .aider-desk/extensions/engram
node tests/run.mjs          # or: ./node_modules/.bin/jiti tests/run.ts
```

| # | Spec test | What the scenario verifies |
|---|---|---|
| 1 | llama.cpp decision → 1 project memory | Exactly 1 memory, category `decision`, scope `project`, attached to the project, no wasted LLM calls, API key never in prompts |
| 2 | Same fact repeated → no duplicate | 2nd extraction: `duplicates: 1`, `stored: 0`, store still at 1 |
| 3 | Switch to Ollama → update | `updated: 1`, no second memory, statement mentions both backends, updatedAt set |
| 4 | Fake API key + malformed JSON | Secret redacted **before** sending (`[REDACTED]`), never stored nor transmitted; invalid JSON → 1 repair → stored; corrupt JSON → never written |
| 5 | Secondary LLM off | Extraction without exception, `unreachable/timeout` failure counted, store empty, agent keeps working |
| 6 | 20+ redundant memories → consolidation | safe_mode pass: merged without deletion; aggressive pass: redundancies deleted, all 4 distinct facts survive |
| 7 | (bonus) native `aiderdesk` transport | Every call routed to `model_id` via `generateText`, 0 HTTP requests |

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `[Memory] secondary LLM probe failed` | llama-server down / wrong URL | `curl http://<host>:4000/v1/models`; fix `base_url` in Settings |
| No extraction in logs | Trigger not reached or extraction disabled | Check `extraction.enabled`, `trigger`, set log level `debug` |
| Memories never injected | Native Memory disabled, or nothing relevant | Check Settings → Memory; `Memory: Extract Now`; raise `retrieval.max_memories` |
| Duplicates persist | Secondary LLM too weak to classify | Raise `max_existing_for_dedup`; run `Memory: Consolidate` |
| `generateText … returned no text` (aiderdesk transport) | `model_id` missing in Settings → Models | Create the provider/model or switch back to `transport: "http"` |
| Extraction timeouts | Slow model on 12 GB | Lower `extraction.max_input_tokens` or raise `timeout_ms` |
| Nothing is written at all | Candidate importance < `min_importance` | Lower `min_importance` to 2, check `debug` logs |

## 9. Known limitations (honesty section)

- **Footer inside `content`**: a direct consequence of the missing metadata column in the 0.81.0 Memory API. Memories created outside Engram are neither read nor modified by consolidation.
- **`min_relevance` is advisory**: the native API returns no score; the effective filter is an importance floor plus AiderDesk's global `memory.maxDistance` setting.
- **Injection via `onImportantReminders`**: the block arrives with the reminders (in the user message, inside `<ThisIsImportant>`), not in the system prompt — the 0.81.0 API offers no alternative.
- **Consolidation in batches**: ~9,000-token budget per batch, oldest memories first; beyond that, remaining batches run on the next cycle.
- **`prompt_end` as trigger**: fires often, so it costs more secondary-LLM calls; `agent_end` is recommended.

## 10. Privacy

- Redaction happens **before** anything is sent to the secondary LLM (17 secret families), plus candidates containing secrets are rejected.
- The secondary LLM stays on the LAN (llama-server): nothing leaves by default. Pointing it at a cloud endpoint (OpenRouter…) is an explicit config choice.
- No secrets in logs; `Memory: Forget` / `Clear Project Memories` for targeted or bulk deletion (confirmation required).

## License

MIT
