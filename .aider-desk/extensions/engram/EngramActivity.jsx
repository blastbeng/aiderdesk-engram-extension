/**
 * Engram activity viewer.
 *
 * A compact badge in the task status bar that opens a live view of what the
 * extension is doing for THIS task: retrieval hits, extraction rounds, stored
 * /updated/deleted counts, consolidation runs, warnings and errors.
 *
 * Two views inside the modal:
 *  - Feed: the activity stream. When `ui.show_content` is on, each retrieval
 *    and extraction entry also lists the memory statements it moved.
 *  - Memories: the Engram rows currently in AiderDesk Memory for this project
 *    (+ global), with scope, category, importance and — with `ui.show_content`
 *    on — the statement itself.
 *
 * Data flow:
 *  - `props.data` is the status snapshot returned by getUIExtensionData().
 *  - `props.executeExtensionAction(action, args)` calls
 *    executeUIExtensionAction() in index.ts for the live feed ('activity'),
 *    the header ('status') and the action buttons.
 *
 * Everything shown is a count, an id or a short label: the feed never carries
 * conversation content or secrets.
 */
(props) => {
  const { useState, useEffect, useCallback } = React;
  const ui = props.ui || {};
  const { Button, Select, ModalOverlayLayout } = ui;
  const Fi = (props.icons && props.icons.Fi) || {};
  const { FiActivity, FiRefreshCw, FiTrash2, FiZap, FiLayers, FiTerminal, FiX, FiCpu } = Fi;
  const exec = props.executeExtensionAction;

  const initial = props.data || {};
  const [status, setStatus] = useState(initial);
  const [open, setOpen] = useState(false);
  const [scope, setScope] = useState('task');
  const [level, setLevel] = useState('info');
  const [entries, setEntries] = useState([]);
  const [notice, setNotice] = useState('');
  const [pending, setPending] = useState('');
  // 'feed' = what the extension is doing; 'memories' = what is in AiderDesk Memory.
  const [view, setView] = useState('feed');
  const [memories, setMemories] = useState(null);

  const taskId = props.task && props.task.id ? props.task.id : '';
  const projectDir = props.projectDir || (initial.project && initial.project.dir) || '';

  const refreshStatus = useCallback(async () => {
    if (!exec) return;
    try {
      const next = await exec('status', { taskId: taskId, projectDir: projectDir });
      if (next) setStatus(next);
    } catch (e) {
      /* keep the last snapshot */
    }
  }, [exec, taskId, projectDir]);

  const refreshEntries = useCallback(async () => {
    if (!exec) return;
    try {
      const result = await exec('activity', {
        scope: scope,
        minLevel: level,
        taskId: taskId,
        projectDir: projectDir,
        limit: 250,
      });
      if (result && Array.isArray(result.entries)) setEntries(result.entries);
      if (result && result.status) setStatus(result.status);
    } catch (e) {
      /* keep the last feed */
    }
  }, [exec, scope, level, taskId, projectDir]);

  // Collapsed badge: refresh the indicator even while the modal is closed.
  useEffect(() => {
    if (!exec) return undefined;
    refreshStatus();
    const id = setInterval(refreshStatus, open ? 2500 : 5000);
    return () => clearInterval(id);
  }, [exec, open, refreshStatus]);

  // Open modal: poll the feed on a short interval.
  useEffect(() => {
    if (!open || !exec) return undefined;
    refreshEntries();
    const id = setInterval(refreshEntries, 2500);
    return () => clearInterval(id);
  }, [open, exec, refreshEntries]);

  const loadMemories = useCallback(async () => {
    if (!exec) return;
    try {
      const result = await exec('memories', { taskId: taskId, projectDir: projectDir, limit: 60 });
      if (result) setMemories(result);
    } catch (e) {
      /* keep the last list */
    }
  }, [exec, taskId, projectDir]);

  useEffect(() => {
    if (!open || view !== 'memories' || !exec) return undefined;
    loadMemories();
    const id = setInterval(loadMemories, 5000);
    return () => clearInterval(id);
  }, [open, view, exec, loadMemories]);

  const runAction = useCallback(
    async (action, options = {}) => {
      if (!exec) return;
      setPending(action);
      setNotice('');
      try {
        const result = await exec(action, options || {});
        if (result && result.error) {
          setNotice(result.error);
        } else if (action === 'extract') {
          setNotice('Extraction started in the background.');
        } else if (action === 'consolidate') {
          setNotice('Consolidation started in the background.');
        } else if (action === 'clear') {
          setNotice('Feed cleared.');
          setEntries([]);
        } else if (action === 'dump') {
          setNotice('Wrote the feed to the task log.');
        }
        refreshStatus();
        refreshEntries();
        if (view === 'memories') loadMemories();
      } catch (e) {
        setNotice('Action failed: ' + (e && e.message ? e.message : String(e)));
      } finally {
        setPending('');
      }
    },
    [exec, refreshStatus, refreshEntries, view, loadMemories],
  );

  const activityInfo = (status && status.activity) || {};
  const errors = activityInfo.errors || 0;
  const warnings = activityInfo.warnings || 0;
  const busy = Boolean(status && status.busy);
  const off = Boolean(status && status.enabled === false);
  const noMem = Boolean(status && status.memoryAvailable === false);

  const dotClass = off
    ? 'bg-text-muted'
    : errors > 0
      ? 'bg-error'
      : busy
        ? 'bg-accent animate-pulse'
        : warnings > 0
          ? 'bg-warning'
          : 'bg-success';

  const badgeLabel = off ? 'Engram off' : busy ? 'Engram working' : 'Engram';

  const levelClass = (value) => {
    if (value === 'error') return 'text-error';
    if (value === 'warn') return 'text-warning';
    if (value === 'debug') return 'text-text-muted';
    return 'text-text-secondary';
  };

  const fmtTime = (ts) => {
    try {
      return new Date(ts).toLocaleTimeString();
    } catch (e) {
      return '';
    }
  };

  const projectCounts = (status && status.project && status.project.counts) || {};
  const totals = (status && status.totals) || {};
  const llm = (status && status.llm) || {};

  return (
    <div className="relative flex items-center">
      <button
        onClick={() => setOpen(true)}
        title="Engram activity - what the memory extension is doing right now"
        className="flex items-center gap-1.5 px-2 py-0.5 rounded-md border border-border bg-bg-secondary hover:bg-bg-tertiary text-xs text-text-secondary cursor-pointer"
      >
        <span className={'w-2 h-2 rounded-full ' + dotClass} />
        <span className="whitespace-nowrap">{badgeLabel}</span>
        {errors > 0 ? <span className="text-error font-semibold">{errors}</span> : null}
        {errors === 0 && warnings > 0 ? <span className="text-warning font-semibold">{warnings}</span> : null}
      </button>

      {open && ModalOverlayLayout ? (
        <ModalOverlayLayout title="Engram activity" onClose={() => setOpen(false)}>
          <div className="flex flex-1 min-h-0 w-full max-w-[720px] mx-auto flex-col gap-3 overflow-y-auto p-3">
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-3 flex-wrap text-xs text-text-secondary">
                <span className="flex items-center gap-1.5">
                  <span className={'w-2 h-2 rounded-full ' + dotClass} />
                  {badgeLabel}
                </span>
                <span>v{status && status.version ? status.version : '?'}</span>
                <span>
                  agent:{' '}
                  <span className="font-mono">{status && status.agentId ? status.agentId : 'global'}</span>
                  {status && status.perAgent ? ' (with overrides)' : ''}
                </span>
                {off ? <span className="text-warning">extension disabled</span> : null}
                {noMem ? <span className="text-warning">AiderDesk Memory is off</span> : null}
              </div>
              <div className="text-2xs text-text-muted break-all">
                {llm.transport === 'aiderdesk'
                  ? 'model: ' + (llm.modelId || '?')
                  : 'endpoint: ' + (llm.model || '?') + ' @ ' + (llm.endpoint || '?')}
              </div>
              <div className="flex items-center gap-3 flex-wrap text-2xs text-text-muted">
                <span>
                  feed {activityInfo.size || 0}/{activityInfo.limit || 0}
                </span>
                <span>
                  statements:{' '}
                  {status && status.ui && status.ui.showContent ? (
                    <span className="text-success">shown</span>
                  ) : (
                    <span className="text-text-muted">hidden</span>
                  )}
                </span>
                <span className="text-warning">warn {warnings}</span>
                <span className="text-error">error {errors}</span>
                <span>
                  project: extractions {projectCounts.extractions || 0}, stored {projectCounts.stored || 0}, updated{' '}
                  {projectCounts.updated || 0}, deleted {projectCounts.deleted || 0}
                </span>
                <span>
                  totals: LLM {totals.llmCalls || 0} (fail {totals.llmFailures || 0})
                </span>
              </div>
            </div>

            <div className="flex flex-row gap-2 items-center flex-wrap border-t border-border pt-3">
              <Button
                variant={view === 'memories' ? 'contained' : 'outline'}
                size="xs"
                onClick={() => setView(view === 'memories' ? 'feed' : 'memories')}
              >
                <span className="flex items-center gap-1">
                  {FiLayers ? <FiLayers className="w-3 h-3" /> : null} Memories
                </span>
              </Button>
              <Select
                label="Scope"
                value={scope}
                onChange={(value) => setScope(value)}
                options={[
                  { value: 'task', label: 'This task' },
                  { value: 'project', label: 'This project' },
                  { value: 'all', label: 'Everything' },
                ]}
              />
              <Select
                label="Level"
                value={level}
                onChange={(value) => setLevel(value)}
                options={[
                  { value: 'debug', label: 'debug' },
                  { value: 'info', label: 'info' },
                  { value: 'warn', label: 'warn' },
                  { value: 'error', label: 'error' },
                ]}
              />
              <Button variant="outline" size="xs" onClick={() => runAction('refresh')}>
                <span className="flex items-center gap-1">
                  {FiRefreshCw ? <FiRefreshCw className="w-3 h-3" /> : null} Refresh
                </span>
              </Button>
              <Button variant="outline" size="xs" onClick={() => runAction('extract', { taskId: taskId })}>
                <span className="flex items-center gap-1">
                  {FiZap ? <FiZap className="w-3 h-3" /> : null} Extract now
                </span>
              </Button>
              <Button variant="outline" size="xs" onClick={() => runAction('consolidate', { taskId: taskId })}>
                <span className="flex items-center gap-1">
                  {FiLayers ? <FiLayers className="w-3 h-3" /> : null} Consolidate
                </span>
              </Button>
              <Button
                variant="outline"
                size="xs"
                onClick={() => runAction('dump', { taskId: taskId, scope: scope, minLevel: level })}
              >
                <span className="flex items-center gap-1">
                  {FiTerminal ? <FiTerminal className="w-3 h-3" /> : null} To task log
                </span>
              </Button>
              <Button variant="outline" size="xs" color="danger" onClick={() => runAction('clear')}>
                <span className="flex items-center gap-1">
                  {FiTrash2 ? <FiTrash2 className="w-3 h-3" /> : null} Clear
                </span>
              </Button>
            </div>

            {notice ? <div className="text-xs text-text-secondary">{notice}</div> : null}

            {view === 'memories' ? (
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-3 flex-wrap text-2xs text-text-muted">
                  <span>
                    {memories
                      ? `${memories.items ? memories.items.length : 0} of ${memories.total || 0} Engram memories (this project + global)`
                      : 'loading…'}
                  </span>
                  {memories && memories.error ? (
                    <span className="text-warning">{memories.error}</span>
                  ) : null}
                  {memories && memories.showContent === false ? (
                    <span className="text-warning">
                      statements hidden — enable Settings → Extensions → Engram → “Show memory statements”
                    </span>
                  ) : null}
                  <Button variant="outline" size="xs" onClick={loadMemories}>
                    Reload
                  </Button>
                </div>
                <div className="flex flex-col rounded-md border border-border">
                  {!memories || !memories.items || memories.items.length === 0 ? (
                    <p className="text-xs text-text-muted px-2 py-6 text-center">
                      No Engram memories for this project yet. Extraction writes them at the end of an agent turn.
                    </p>
                  ) : (
                    memories.items.map((item) => (
                      <div
                        key={item.id}
                        className="flex flex-col gap-0.5 px-2 py-1.5 border-b border-border last:border-b-0"
                      >
                        <div className="flex items-center gap-2 flex-wrap text-2xs">
                          <span className="px-1 rounded bg-bg-tertiary text-text-muted">{item.scope}</span>
                          <span className="px-1 rounded bg-bg-secondary text-text-secondary">{item.category}</span>
                          <span className="font-semibold text-text-secondary">i{item.importance}</span>
                          <span className="px-1 rounded bg-bg-secondary text-text-muted font-mono">
                            {String(item.id).slice(0, 8)}
                          </span>
                          {item.updatedAt ? <span className="text-text-muted">updated {fmtTime(item.updatedAt)}</span> : null}
                        </div>
                        {item.statement ? (
                          <div className="text-xs text-text-primary break-words">{item.statement}</div>
                        ) : null}
                      </div>
                    ))
                  )}
                </div>
              </div>
            ) : (
              <div className="flex flex-col rounded-md border border-border">
              {entries.length === 0 ? (
                <p className="text-xs text-text-muted px-2 py-6 text-center">
                  Nothing recorded for this scope yet. Activity appears as retrieval, extraction and consolidation run.
                </p>
              ) : (
                entries
                  .slice()
                  .reverse()
                  .map((entry) => (
                    <div
                      key={entry.id}
                      className="flex flex-col gap-0.5 px-2 py-1.5 border-b border-border last:border-b-0"
                    >
                      <div className="flex items-center gap-2 flex-wrap text-2xs">
                        <span className="font-mono text-text-muted">{fmtTime(entry.ts)}</span>
                        <span className={'uppercase font-semibold ' + levelClass(entry.level)}>{entry.level}</span>
                        <span className="px-1 rounded bg-bg-tertiary text-text-muted">{entry.kind}</span>
                        {entry.details
                          ? Object.keys(entry.details).map((key) => (
                              <span key={key} className="px-1 rounded bg-bg-secondary text-text-secondary">
                                {key}={String(entry.details[key])}
                              </span>
                            ))
                          : null}
                      </div>
                      <div className="text-xs text-text-primary break-words">{entry.message}</div>
                      {entry.content && entry.content.length ? (
                        <div className="flex flex-col gap-0.5 pl-2 border-l-2 border-border ml-0.5">
                          {entry.content.map((statement, index) => (
                            <span key={index} className="text-2xs text-text-secondary break-words">
                              {statement}
                            </span>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ))
              )}
              </div>
            )}

            <div className="text-2xs text-text-muted">
              The feed is in-memory and per-session; it resets when AiderDesk restarts. Prefer{' '}
              <span className="font-mono">/memory:activity</span> to print it into the task log. Memory statements
              (what was injected, what was written) appear only with{' '}
              <span className="font-mono">ui.show_content</span> on, and are secret-redacted before display.
            </div>
          </div>
        </ModalOverlayLayout>
      ) : null}
    </div>
  );
}
