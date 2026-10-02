({ config, updateConfig, ui }) => {
  const { Input, Select, Checkbox } = ui;

  const llm = config?.secondary_llm || {};
  const extraction = config?.extraction || {};
  const retrieval = config?.retrieval || {};
  const consolidation = config?.consolidation || {};
  const privacy = config?.privacy || {};
  const logging = config?.logging || {};

  const setLlm = (patch) => updateConfig({ ...config, secondary_llm: { ...llm, ...patch } });
  const setExtraction = (patch) => updateConfig({ ...config, extraction: { ...extraction, ...patch } });
  const setRetrieval = (patch) => updateConfig({ ...config, retrieval: { ...retrieval, ...patch } });
  const setConsolidation = (patch) => updateConfig({ ...config, consolidation: { ...consolidation, ...patch } });
  const setPrivacy = (patch) => updateConfig({ ...config, privacy: { ...privacy, ...patch } });
  const setLogging = (patch) => updateConfig({ ...config, logging: { ...logging, ...patch } });

  const num = (value, fallback) => {
    const n = parseInt(value, 10);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };
  const float = (value, fallback) => {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : fallback;
  };

  const isNative = llm.transport === 'aiderdesk';

  const Section = ({ title }) => (
    <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide pt-2 border-t border-border">
      {title}
    </p>
  );

  return (
    <div className="flex flex-col gap-4">
      <Checkbox
        label="Engram enabled"
        checked={config?.enabled !== false}
        onChange={(checked) => updateConfig({ ...config, enabled: checked })}
      />
      <p className="text-xs text-text-secondary -mt-2">
        When off, no extraction, retrieval or consolidation runs. Existing memories are left untouched.
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
            placeholder="http://192.168.1.29:4000/v1"
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
            placeholder="small-model"
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
          value={String(llm.max_tokens ?? 8192)}
          onChange={(e) => setLlm({ max_tokens: num(e.target.value, 8192) })}
          placeholder="8192"
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
          value={String(retrieval.min_relevance ?? 0.65)}
          onChange={(e) => setRetrieval({ min_relevance: float(e.target.value, 0.65) })}
          placeholder="0.65"
        />
      </div>
      <Checkbox
        label="Include global-scope memories"
        checked={retrieval.include_global !== false}
        onChange={(checked) => setRetrieval({ include_global: checked })}
      />
      <p className="text-xs text-text-secondary -mt-2">
        AiderDesk applies one global similarity threshold (Settings > Memory), not a per-call one, so "min importance"
        is the client-side floor: importance 4-5 rank first, importance 1 is never injected. If nothing qualifies,
        nothing is added to the context.
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
      <p className="text-xs text-text-secondary -mt-2">
        Commands: /memory:extract · /memory:consolidate [force] · /memory:stats · /memory:forget &lt;text&gt; ·
        /memory:clear-project confirm
      </p>
    </div>
  );
}
