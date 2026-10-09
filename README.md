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
- **Importance scoring (1–5)** — importance 1 is discarded at extraction; `retrieval.min_importance` is the injection floor.
- **Project vs global scope** — decisions about this project stay with the project; general preferences follow you everywhere.
- **Relevant retrieval only** — ≤ 8 memories injected per prompt, ranked by vector relevance (most relevant first), gated by an importance floor **and** a lexical word-overlap gate against the current prompt, each statement capped at 300 chars; otherwise nothing is injected (zero context pollution).
- **Secret safety** — 17 redaction rules (API keys, AWS, JWT, Bearer, passwords, private keys, high-entropy blobs) applied **before** anything leaves the process.
- **Never blocks the agent** — fire-and-forget serialized queues; if the secondary LLM is offline, AiderDesk keeps working normally.
- **Full UI + commands** — settings panel, `Memory: Extract Now`, `Memory: Consolidate`, `Memory: Dedup`, `Memory: Show Statistics`, `Memory: Forget`, `Memory: Clear Project Memories`.
- **Dual transport** — direct HTTP to any OpenAI-compatible endpoint (default), or AiderDesk's native `TaskContext.generateText`.
- **Per-agent configuration** — one global config for every AiderDesk agent, plus optional per-agent overrides (different secondary model, different trigger, or memory off for one agent). Untouched fields always inherit the global config.
- **Zero runtime dependencies** — plain `fetch`; no build step; TypeScript loaded directly by AiderDesk's jiti loader.

## Repository layout

```
.aider-desk/extensions/engram/   ← the extension itself (copy this folder)
├── index.ts                     Extension: events, commands, settings UI, queues
├── src/                         Pipeline modules (extraction, consolidation, …)
├── tests/                       Offline acceptance harness (16 scenarios)
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
#    (default: http://192.168.1.13:4000/v1, api_key "local", model "synthetic/syn:small:text")
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
        http://192.168.1.13:4000/v1
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
| List the agents a project can run (per-agent settings tabs) | `ProjectContext.getAgentProfiles(): AgentProfile[]` | ~2044 |
| Settings dialog has no project context (global Settings > Extensions) | `getProjectContext()` throws there, so the agent tabs fall back to reading the profile files on disk: `~/.aider-desk/agents` + every open project's `.aider-desk/agents` | — |
| Resolve which agent a task runs (per-agent config resolution) | `TaskContext.getTaskAgentProfile(): Promise<AgentProfile \| null>`, plus `event.agentProfile?.id` (`onImportantReminders`) and `task.agentProfileId` (`onTaskClosed`) | ~1490 |

### What the API does not offer (and the workarounds used)

1. **`MemoryEntry` has no metadata column** (`{ id, content, type, taskId?, projectId?, timestamp }`, ~line 950). → Attributes (category, importance, scope, confidence, timestamps) are encoded in a **footer** inside `content`: `[mem cat=… imp=… scope=… conf=… ts=… (ut=…)]`. The statement stays first (it dominates the embedding); the footer is stripped before injection and display. Memories whose footer does not decode are **never touched** by consolidation — native AiderDesk memories stay intact.
2. **`retrieveMemories` filters on exact `projectId`** and applies a **global** distance cutoff (`memory.maxDistance`), not a per-call threshold, and returns **no score or distance**. → "Global" scope is implemented with `projectId === ''` (the native Memory default), and filtering is done client-side with `retrieval.min_importance` plus the lexical word-overlap gate `retrieval.min_overlap` (the native API returns no score or distance).
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
├── tests/                 Offline harness (16 scenarios, real mock HTTP server)
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

# 3) Run the offline test harness (16 scenarios, no real LLM needed):
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
    "base_url": "http://192.168.1.13:4000/v1",
    "api_key": "local",                          // llama-server accepts any string
    "model": "synthetic/syn:small:text",         // model name served by the endpoint
    "temperature": 0.1,
    "max_tokens": 16384,       // reasoning models need headroom for reasoning_content
    "timeout_ms": 120000          // 120 s default: small reasoning models think long
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
    "min_importance": 3,           // injection floor: never inject below this importance
    "min_overlap": 1,             // min distinct content words shared with the prompt (0 = gate off)
    "include_global": true
  },
  "consolidation": {
    "enabled": true,
    "interval_tasks": 20,         // every N agent turns
    "safe_mode": true             // true: never deletes, only merges/updates
  },
  "privacy": { "redact_secrets": true },
  "logging": { "enabled": true, "level": "info" },
  "agents": {}                          // per-agent overrides, see below
}
```

Any OpenAI-compatible URL works: Ollama (`http://host:11434/v1`), LiteLLM, vLLM, LM Studio, OpenRouter. The `aiderdesk` transport goes through `TaskContext.generateText` (model declared in Settings → Providers → OpenAI-compatible); the extension falls back to HTTP when no TaskContext is available.

### Per-agent configuration

`agents` maps an **AiderDesk agent profile id** (`AgentProfile.id`, e.g. `"local"`, `"intesa"`) to a partial override of the global config. Every field is optional and **inherits from the global section when unset** — an override only contains what you explicitly changed:

```jsonc
{
  "enabled": true,                      // global: applies to every agent
  "secondary_llm": { "base_url": "http://192.168.1.13:4000/v1", "model": "synthetic/syn:small:text" },
  "agents": {
    "local": {                          // full override for the "local" agent
      "secondary_llm": { "base_url": "http://192.168.1.29:4000/v1", "model": "small-model" },
      "extraction": { "trigger": "task_end" }
    },
    "intesa": { "enabled": false }      // memory completely off for this agent
  }
}
```

Resolution is **global-first** (`resolveConfig()`): the global config is cloned, then the agent's validated overrides are applied section by section, field by field. An unknown or unresolvable agent id yields the global config unchanged, so a task whose agent profile cannot be determined never loses memory.

- **Which agent is used** is resolved per event: `event.agentProfile.id` for prompt-time injection, `task.agentProfileId` at task close, `TaskContext.getTaskAgentProfile()` as the documented fallback for the remaining events.
- **UI**: Settings → Extensions → Engram shows a **Global** tab plus one tab per agent profile. A new agent tab starts in *Inherit global config*; switching it to *Custom* copies the current effective values so you change only what you want, and *Reset to inherit* deletes the override. `Memory: Show Statistics` reports which agent's config was applied.
- **How the tab list is discovered**: the settings dialog runs in a *global* context (no project), where `ProjectContext.getAgentProfiles()` is unavailable. The tabs are therefore listed from three sources, in order: the live API when a project context exists, profiles cached from earlier project-scoped calls in this session, and — always available — the agent profile files on disk (`~/.aider-desk/agents/*/config.json` plus every open project's `.aider-desk/agents/*/config.json`, ordered by `order.json`; project-level profiles win over global ones with the same id). Extension-provided in-memory profiles appear once they have been seen at runtime.
- **Storage**: overrides live in `config.json` under `agents`. The agent list (`_agents`) is injected into the settings UI at read time and is never persisted.

## 5. Installation (10 steps)

1. **Start llama-server** on the secondary-GPU machine (see Requirements) and verify: `curl http://192.168.1.13:4000/v1/models`.
2. **Copy the extension folder** into AiderDesk's extensions directory:
   ```bash
   cp -r .aider-desk/extensions/engram ~/.aider-desk/extensions/engram
   ```
   (per-project install also works: `<project>/.aider-desk/extensions/engram`).

   **Updating an existing global install — keep the live config.** When the extension is installed in the *global* extensions dir, `config.json` there is the real, working configuration for every project (endpoint, key, model, per-agent overrides), and `state.json` is its counters. Never let an update overwrite them:
   ```bash
   rsync -av --exclude config.json --exclude state.json --exclude node_modules \
       .aider-desk/extensions/engram/ ~/.aider-desk/extensions/engram/
   ```
   Both files are gitignored in this repository, so they exist only in the install (and in the dev tree) and are never committed.
3. **Restart AiderDesk**. TS extensions are loaded by jiti — no build, no `npm install`. The extension reads `config.json` **once at startup**, so a restart is also required after any manual edit of that file (changes made through the Settings UI apply immediately, no restart needed).
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
| `Memory: Dedup` | Deterministic exact-duplicate removal across all Engram memories (no LLM involved) |
| `Memory: Show Statistics` | Per-project stats + totals (LLM calls, failures, stored, updated, duplicates) |
| `Memory: Forget <text>` | Find and delete a specific memory |
| `Memory: Clear Project Memories` confirm | Wipe the current project's memories (confirmation required) |

`[Memory] …` logs: extraction started, candidates found, verdicts (new/duplicate/update), consolidation, LLM failures. No secret ever appears in logs (redaction happens upstream).

Settings has one **Global** tab and one tab per AiderDesk agent profile: edit Global to change behaviour everywhere, open an agent tab to give a single agent its own secondary LLM, trigger, retrieval limits — or switch memory off for it. Untouched sections inherit Global; *Reset to inherit* removes an override.

The secondary-LLM API key can also be supplied through the **`ENGRAM_API_KEY`** environment variable, which overrides the value stored in `config.json` — useful for keeping the secret out of the settings file. A per-agent override that sets its own `api_key` still wins.

## 7. The tests (offline harness included)

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
| 8 | Per-agent config | Global config applies to every agent; an override wins for its agent only — different endpoint for one agent, memory off for another, untouched sections inherit |
| 9 | Relevance-primary retrieval + deterministic dedup | Native vector order kept, below-floor memories never injected; an off-topic memory the native search still returns is dropped by the word-overlap gate (`min_overlap: 0` disables it); exact duplicates removed (best copy per scope), idempotent |
| 10 | Settings without a project context | Agent tabs fall back to the profile files on disk (`order.json` ordering, project-level wins), and the UI-only `_agents` key never reaches `config.json` |
| 11 | Alias round-trip in consolidation | Real UUID ids survive positional aliases (m1, m2, …) to merge/delete the right memories; unmentioned memories count as kept |
| 12 | Hanging secondary endpoint | Classified as `timeout` (not unreachable), counted once, never blocks the agent |
| 13 | Alias resolution robustness + stats | `resolveAlias` tolerates every id format a small model emits; per-project stats increment monotonically |
| 14 | Council regression pack | A truncated consolidation keeps the round counter, model-written secrets never land in the store, dedup reports per-project removals |
| 15 | Command surface | Host-legal command definitions, no throw on a broken task context, guarded destructive clear, unload cancels a running consolidation |
| 16 | Batch-failure resilience + hash false positives | A failed consolidation batch is skipped (cycle continues, counter kept, failure reported per batch), and pure-hex hashes (SHA-1/SHA-256) are not treated as secrets |

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `[Memory] secondary LLM probe failed` | llama-server down / wrong URL | `curl http://<host>:4000/v1/models`; fix `base_url` in Settings |
| No extraction in logs | Trigger not reached or extraction disabled | Check `extraction.enabled`, `trigger`, set log level `debug` |
| Memories never injected | Native Memory disabled, nothing relevant, the importance floor — or every hit failed the word-overlap gate | Check Settings → Memory; `Memory: Extract Now`; raise `retrieval.max_memories`, lower `retrieval.min_importance`, or set `retrieval.min_overlap` to 0 to disable the gate |
| Duplicates persist | Secondary LLM too weak to classify | Raise `max_existing_for_dedup`; run `Memory: Consolidate` |
| `generateText … returned no text` (aiderdesk transport) | `model_id` missing in Settings → Models | Create the provider/model or switch back to `transport: "http"` |
| Extraction timeouts | Slow model on 12 GB | Lower `extraction.max_input_tokens` or raise `timeout_ms` |
| Nothing is written at all | Candidate importance < `min_importance` | Lower `min_importance` to 2, check `debug` logs |
| `response has no chat completion content (finish_reason=length)` | Reasoning model spending the whole budget on `reasoning_content` | Raise `secondary_llm.max_tokens` (≥ 8192; 16384 default is comfortable) |
| `Memory: Show Statistics` shows every LLM call failing after you edited `config.json` by hand | Config is read once at startup; the running process still uses the old endpoint | Restart AiderDesk (Settings-UI saves apply immediately — no restart needed) |
| A per-agent override seems ignored | Agent id mismatch (override keyed by `AgentProfile.id`, not display name), or the agent tab is in *Inherit* mode | Open the agent's tab (its id is shown there), switch it to *Custom*, save; `Memory: Show Statistics` prints which agent's config was applied |

## 9. Known limitations (honesty section)

- **Footer inside `content`**: a direct consequence of the missing metadata column in the 0.81.0 Memory API. Memories created outside Engram are neither read nor modified by consolidation.
- **No per-call relevance threshold**: `retrieveMemories` returns no score or distance, and AiderDesk's global `memory.maxDistance` (default 1.5 on the 0–2 cosine scale) is permissive — the effective client-side gates are `retrieval.min_importance` and the lexical word-overlap gate `retrieval.min_overlap`.
- **Injection via `onImportantReminders`**: the block arrives with the reminders (in the user message, inside `<ThisIsImportant>`), not in the system prompt — the 0.81.0 API offers no alternative.
- **Consolidation in batches**: ~4,500-token budget per batch, oldest memories first; a failed batch is skipped and reported while the cycle continues with the next; beyond budget, remaining batches run on the next cycle.
- **`prompt_end` as trigger**: fires often, so it costs more secondary-LLM calls; `agent_end` is recommended.
- **Per-agent overrides need a resolvable agent id**: the id comes from the event payload or `TaskContext.getTaskAgentProfile()`. When neither is available (rare, e.g. a command invoked without a task context) the global config is used — memory degrades to global, never to nothing. The settings tab list comes from the live API when available and from the agent profile files on disk otherwise (see "How the tab list is discovered" above) — including when the dialog is opened from the global Settings > Extensions page, which has no project context.

## 10. Privacy

- Redaction happens **before** anything is sent to the secondary LLM (17 secret families), plus candidates containing secrets are rejected.
- The secondary LLM stays on the LAN (llama-server): nothing leaves by default. Pointing it at a cloud endpoint (OpenRouter…) is an explicit config choice.
- No secrets in logs; `Memory: Forget` / `Clear Project Memories` for targeted or bulk deletion (confirmation required).

## License

MIT
