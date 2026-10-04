({ config, updateConfig, ui }) => {
  const { Input, Select, Checkbox, Button } = ui;
  const { useState } = React;

  const num = (value, fallback) => {
    const n = parseInt(value, 10);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };
  const float = (value, fallback) => {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : fallback;
  };

  // ---- scopes -----------------------------------------------------------------
  // config._agents is injected by getConfigData(): the AiderDesk agent profiles
  // available in this project. config.agents holds the per-agent overrides.
  const agentsMeta = Array.isArray(config?._agents) ? config._agents : [];
  const overrides = config?.agents || {};

  const agentIds = (() => {
    const ids = agentsMeta.map((a) => a?.id).filter(Boolean);
    for (const id of Object.keys(overrides)) if (!ids.includes(id)) ids.push(id);
    return ids;
  })();

  const [tab, setTab] = useState('global');
  const agent = tab === 'global' ? null : tab;
  const ov = agent ? overrides[agent] || {} : {};
  const isCustom = Object.keys(ov).length > 0;

  // Effective settings for the active scope: global values, plus the override.
  const section = (name) => ({
    ...(config?.[name] || {}),
    ...(agent ? ov[name] || {} : {}),
  });
  const eff = {
    enabled: typeof ov.enabled === 'boolean' ? ov.enabled : config?.enabled !== false,
    secondary_llm: section('secondary_llm'),
    extraction: section('extraction'),
    retrieval: section('retrieval'),
    consolidation: section('consolidation'),
    privacy: section('privacy'),
    logging: section('logging'),
  };

  // Write a field to the active scope. Global writes the section; an agent
  // writes the same section into config.agents[agent], so untouched sections
  // keep inheriting from the global config.
  const write = (name, patch) => {
    if (name === 'enabled') {
      if (agent) {
        updateConfig({ ...config, agents: { ...overrides, [agent]: { ...ov, enabled: patch } } });
      } else {
        updateConfig({ ...config, enabled: patch });
      }
      return;
    }
    if (agent) {
      updateConfig({
        ...config,
        agents: { ...overrides, [agent]: { ...ov, [name]: { ...eff[name], ...patch } } },
      });
    } else {
      updateConfig({ ...config, [name]: { ...eff[name], ...patch } });
    }
  };

  const setLlm = (patch) => write('secondary_llm', patch);
  const setExtraction = (patch) => write('extraction', patch);
  const setRetrieval = (patch) => write('retrieval', patch);
  const setConsolidation = (patch) => write('consolidation', patch);
  const setPrivacy = (patch) => write('privacy', patch);
  const setLogging = (patch) => write('logging', patch);

  const startCustom = () => {
    updateConfig({
      ...config,
      agents: { ...overrides, [agent]: { ...ov, enabled: config?.enabled !== false } },
    });
  };
  const resetToGlobal = () => {
    const next = { ...overrides };
    delete next[agent];
    updateConfig({ ...config, agents: next });
  };

  const llm = eff.secondary_llm;
  const extraction = eff.extraction;
  const retrieval = eff.retrieval;
  const consolidation = eff.consolidation;
  const privacy = eff.privacy;
  const logging = eff.logging;
  const isNative = llm.transport === 'aiderdesk';

  const metaFor = (id) => agentsMeta.find((a) => a?.id === id) || {};

  const Section = ({ title }) => (
    <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide pt-2 border-t border-border">
      {title}
    </p>
  );

  const Tab = ({ id, label, active, badge }) => (
    <button
      key={id}
      onClick={() => setTab(id)}
      className={`px-3 py-1.5 rounded-md text-xs border transition-colors cursor-pointer ${
        active
          ? 'bg-accent text-white border-accent'
          : 'bg-bg-secondary text-text-secondary border-border hover:text-text-primary'
      }`}
    >
      {label}
      {badge ? <span className="ml-1 opacity-70">*</span> : null}
    </button>
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-row gap-2 flex-wrap items-center">
        <Tab id="global" label="Global" active={tab === 'global'} badge={Object.keys(overrides).length > 0} />
        {agentIds.map((id) => (
          <Tab
            key={id}
            id={id}
            label={metaFor(id).name || id}
            active={tab === id}
            badge={Boolean(overrides[id])}
          />
        ))}
      </div>

      {agent ? (
        <div className="flex flex-col gap-2 rounded-md border border-border bg-bg-secondary p-3">
          <p className="text-xs text-text-secondary">
            Editing the <span className="font-semibold text-text-primary">{metaFor(agent).name || agent}</span> agent
            (id <span className="font-mono">{agent}</span>
            {metaFor(agent).provider ? `, ${metaFor(agent).provider}/${metaFor(agent).model}` : ''}).
          </p>
          <p className="text-xs text-text-secondary">
            {isCustom
              ? `Overriding: ${Object.keys(ov).join(', ')}. Sections not listed here inherit the global config.`
              : 'Inheriting everything from the Global tab. Choose "Custom" to override single sections.'}
          </p>
          <div className="flex flex-row gap-2 items-center flex-wrap">
            <Select
              label="Config mode"
              value={isCustom ? 'custom' : 'inherit'}
              onChange={(value) => (value === 'custom' ? startCustom() : resetToGlobal())}
              options={[
                { value: 'inherit', label: 'Inherit global config' },
                { value: 'custom', label: 'Custom for this agent' },
              ]}
            />
            {isCustom ? (
              <Button variant="outline" size="xs" onClick={resetToGlobal}>
                Reset to inherit
              </Button>
            ) : null}
          </div>
        </div>
      ) : (
        <p className="text-xs text-text-secondary">
          Global config: applies to every AiderDesk agent. Open an agent tab to give one agent its own secondary
          LLM, trigger, retrieval depth or consolidation interval.
          {agentIds.length === 0
            ? ' No agent tabs right now: none of your AiderDesk agent profiles (Settings > Agents, global or per project) could be listed.'
            : ''}
        </p>
      )}

      {agent && !isCustom ? null : (
        <div className="flex flex-col gap-4">
          <Checkbox
            label="Engram enabled"
            checked={eff.enabled}
            onChange={(checked) => write('enabled', checked)}
          />
          <p className="text-xs text-text-secondary -mt-2">
            When off, no extraction, retrieval or consolidation runs. Existing memories are left untouched.
            {agent ? ' For this agent only; use "Reset to inherit" to follow the global switch again.' : ''}
          </p>

          <Section title="Secondary LLM" />

          <Select
            label="Transport"
            value={llm.transport || 'http'}
            onChange={(value) => setLlm({ transport: value })}
            options={[
              { value: 'http', label: 'OpenAI-compatible HTTP (llama-server, Ollama, LiteLLM, vLLM)' },
              { value: 'aiderdesk', label: 'AiderDesk provider model (TaskContext.generateText)' },
            ]}
          />
          <p className="text-xs text-text-secondary -mt-2">
            {isNative
              ? 'Calls a model registered in AiderDesk Settings > Providers/Models by its "provider/model" id. Never the main agent model.'
              : 'Direct HTTP chat-completions call made by the extension. Nothing to register in AiderDesk; the main model is never used.'}
          </p>

          {isNative ? (
            <Input
              label="AiderDesk model id (provider/model)"
              value={llm.model_id || ''}
              onChange={(e) => setLlm({ model_id: e.target.value })}
              placeholder="openai-compatible/engram-secondary"
            />
          ) : (
            <div className="flex flex-col gap-4">
              <Input
                label="Base URL"
                value={llm.base_url || ''}
                onChange={(e) => setLlm({ base_url: e.target.value })}
                placeholder="http://192.168.1.13:4000/v1"
              />
              <p className="text-xs text-text-secondary -mt-2">
                Accepts .../v1 or a bare host. Works with llama-server, Ollama (/v1), LiteLLM, OpenRouter, vLLM, LM Studio.
              </p>
              <Input
                label="API key"
                type="password"
                value={llm.api_key || ''}
                onChange={(e) => setLlm({ api_key: e.target.value })}
                placeholder="local"
              />
              <p className="text-xs text-text-secondary -mt-2">
                Sent as "Authorization: Bearer &lt;key&gt;". llama-server and Ollama accept any non-empty value, e.g. "local".
              </p>
              <Input
                label="Model"
                value={llm.model || ''}
                onChange={(e) => setLlm({ model: e.target.value })}
                placeholder="synthetic/syn:small:text"
              />
              <p className="text-xs text-text-secondary -mt-2">
                Must match the id the endpoint serves (llama-server: the -m alias; Ollama: e.g. qwen2.5:7b-instruct).
              </p>
            </div>
          )}

          <div className="flex flex-row gap-4">
            <Input
              label="Temperature"
              value={String(llm.temperature ?? 0.1)}
              onChange={(e) => setLlm({ temperature: float(e.target.value, 0.1) })}
              placeholder="0.1"
            />
            <Input
              label="Max tokens"
              value={String(llm.max_tokens ?? 16384)}
              onChange={(e) => setLlm({ max_tokens: num(e.target.value, 16384) })}
              placeholder="16384"
            />
            <Input
              label="Timeout (ms)"
              value={String(llm.timeout_ms ?? 30000)}
              onChange={(e) => setLlm({ timeout_ms: num(e.target.value, 30000) })}
              placeholder="30000"
            />
          </div>
          <p className="text-xs text-text-secondary -mt-2">
            On timeout, connection failure or invalid JSON the extension logs it and skips the round. AiderDesk keeps working.
          </p>

          <Section title="Extraction" />

          <Checkbox
            label="Automatic extraction"
            checked={extraction.enabled !== false}
            onChange={(checked) => setExtraction({ enabled: checked })}
          />
          <Select
            label="Trigger"
            value={extraction.trigger || 'agent_end'}
            onChange={(value) => setExtraction({ trigger: value })}
            options={[
              { value: 'agent_end', label: 'After each agent run (onAgentFinished)' },
              { value: 'prompt_end', label: 'After each prompt (onPromptFinished)' },
              { value: 'task_end', label: 'When the task closes (onTaskClosed)' },
            ]}
          />
          <p className="text-xs text-text-secondary -mt-2">
            Runs in a background queue after the response is returned. The main model never waits for it.
          </p>
          <div className="flex flex-row gap-4">
            <Input
              label="Max messages"
              value={String(extraction.max_messages ?? 30)}
              onChange={(e) => setExtraction({ max_messages: num(e.target.value, 30) })}
              placeholder="30"
            />
            <Input
              label="Max input tokens"
              value={String(extraction.max_input_tokens ?? 12000)}
              onChange={(e) => setExtraction({ max_input_tokens: num(e.target.value, 12000) })}
              placeholder="12000"
            />
            <Input
              label="Min importance"
              value={String(extraction.min_importance ?? 2)}
              onChange={(e) => setExtraction({ min_importance: num(e.target.value, 2) })}
              placeholder="2"
            />
          </div>
          <div className="flex flex-row gap-4">
            <Input
              label="Max candidates"
              value={String(extraction.max_candidates ?? 20)}
              onChange={(e) => setExtraction({ max_candidates: num(e.target.value, 20) })}
              placeholder="20"
            />
            <Input
              label="Existing memories per dedup"
              value={String(extraction.max_existing_for_dedup ?? 12)}
              onChange={(e) => setExtraction({ max_existing_for_dedup: num(e.target.value, 12) })}
              placeholder="12"
            />
          </div>

          <Section title="Retrieval" />

          <Checkbox
            label="Inject relevant memories into the prompt"
            checked={retrieval.enabled !== false}
            onChange={(checked) => setRetrieval({ enabled: checked })}
          />
          <div className="flex flex-row gap-4">
            <Input
              label="Max memories"
              value={String(retrieval.max_memories ?? 8)}
              onChange={(e) => setRetrieval({ max_memories: num(e.target.value, 8) })}
              placeholder="8"
            />
            <Input
              label="Min importance to inject"
              value={String(retrieval.min_importance ?? 3)}
              onChange={(e) => setRetrieval({ min_importance: num(e.target.value, 3) })}
              placeholder="3"
            />
          </div>
          <Checkbox
            label="Include global-scope memories"
            checked={retrieval.include_global !== false}
            onChange={(checked) => setRetrieval({ include_global: checked })}
          />
          <p className="text-xs text-text-secondary -mt-2">
            AiderDesk applies one global similarity threshold (Settings &gt; Memory), not a per-call one, so "Min relevance"
            is only a hint. Memories are injected in vector-search order (most relevant first) and filtered by the
            "Min importance to inject" floor. If nothing qualifies, nothing is added to the context.
          </p>

          <Section title="Consolidation" />

          <Checkbox
            label="Automatic consolidation"
            checked={consolidation.enabled !== false}
            onChange={(checked) => setConsolidation({ enabled: checked })}
          />
          <Input
            label="Interval (extraction rounds)"
            value={String(consolidation.interval_tasks ?? 20)}
            onChange={(e) => setConsolidation({ interval_tasks: num(e.target.value, 20) })}
            placeholder="20"
          />
          <Checkbox
            label="Safe mode (merge and update, never delete)"
            checked={consolidation.safe_mode !== false}
            onChange={(checked) => setConsolidation({ safe_mode: checked })}
          />
          <p className="text-xs text-text-secondary -mt-2">
            In safe mode redundant memories are demoted to importance 1 instead of deleted. Use
            "/memory:consolidate force" to allow deletions.
          </p>

          <Section title="Privacy & Logging" />

          <Checkbox
            label="Redact secrets before sending anything to the secondary LLM"
            checked={privacy.redact_secrets !== false}
            onChange={(checked) => setPrivacy({ redact_secrets: checked })}
          />
          <p className="text-xs text-text-secondary -mt-2">
            Strips API keys (sk-, ghp-, AKIA..., x-api-key), Bearer/session tokens, private keys, password= and
            URL credentials. Candidates that still look secret are dropped and never stored.
          </p>
          <div className="flex flex-row gap-4">
            <Checkbox
              label="Logging"
              checked={logging.enabled !== false}
              onChange={(checked) => setLogging({ enabled: checked })}
            />
            <Select
              label="Log level"
              value={logging.level || 'info'}
              onChange={(value) => setLogging({ level: value })}
              options={[
                { value: 'debug', label: 'debug' },
                { value: 'info', label: 'info' },
                { value: 'warn', label: 'warn' },
                { value: 'error', label: 'error' },
              ]}
            />
          </div>
        </div>
      )}

      <p className="text-xs text-text-secondary -mt-2">
        Commands: /memory:extract · /memory:consolidate [force] · /memory:stats · /memory:forget &lt;text&gt; ·
        /memory:clear-project confirm
      </p>
    </div>
  );
}
