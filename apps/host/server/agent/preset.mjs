/**
 * The agent preset the app's chats run under: the Harness's `standard` preset (pinned 57190202,
 * packages/bundle/web-app/presets/standard.patch.yml) with the differences the app needs, and nothing else.
 *
 * A patch replaces the whole preset row, so the list is restated here. `tests/agent-tools.test.mjs` compares this list
 * with the pinned Harness's own file (when STRATA_HARNESS_DIR is set), so a Harness update that adds or renames a
 * plugin fails a test instead of silently dropping a tool.
 *
 * Differences from `standard`, each with its reason:
 *   - `tool-schedule` removed: the app has one scheduler (server/tasks.mjs, the Tasks screen); a second set of
 *     reminder tools would run behind its back.
 *   - `web_search` removed (tool-web with `search:false`): every search backend the Harness has needs a service key
 *     the app does not have, so the tool could only fail. `web_fetch` stays. (`searchProvider` can switch it back on.)
 *   - `tool-ask-user` removed: the app has no place to show the agent's own questions, so the tool could only fail;
 *     the agent asks in plain text instead.
 *   - the `planning` group (Harness plan mode and `exit_plan_mode`) removed: the app's Plan mode is the permission
 *     preset (nothing can change); the Harness's plan mode needs an interactive review the app does not have.
 *   - `tool-bash` / `tool-pwsh`: exactly one, by operating system (the Harness chooses the same at load).
 *   - `tool-plugin-manager` stays disabled, as in `standard`.
 */

/** Plugin ids of the pinned Harness `standard` preset, in its order (the drift check). */
export const STANDARD_PLUGIN_IDS = [
  'persona', 'agent-instructions', 'time-context', 'tool-bash', 'tool-pwsh', 'tool-fs', 'tool-fs-search', 'tool-jobs',
  'tool-schedule', 'skill-filesystem', 'tool-skill', 'command-goal', 'tool-goal', 'planning', 'compaction', 'delegation',
  'tool-ask-user', 'tool-todo', 'tool-web', 'present', 'tool-plugin-manager',
];
/** What we leave out of it. */
export const REMOVED_PLUGIN_IDS = ['tool-schedule', 'tool-ask-user', 'planning'];

/**
 * @param {{platform?: string, search?: boolean}} [options]
 * @returns {object} the `config` of the `preset-standard` row
 */
export function standardPreset({ platform = process.platform, search = false } = {}) {
  const windows = platform === 'win32';
  const plugins = [
    { id: 'persona', name: '@deepseek-ai/dsh-persona', config: { prefix: 'You are a coding agent powered by the {{model}} model.' } },
    { id: 'agent-instructions', name: '@deepseek-ai/dsh-agent-instructions', config: { maxBytes: 65536 } },
    { id: 'time-context', name: '@deepseek-ai/dsh-time-context' },
    windows
      ? { id: 'tool-pwsh', name: '@deepseek-ai/dsh-tool-pwsh' }
      : { id: 'tool-bash', name: '@deepseek-ai/dsh-tool-bash' },
    { id: 'tool-fs', name: '@deepseek-ai/dsh-tool-fs' },
    { id: 'tool-fs-search', name: '@deepseek-ai/dsh-tool-fs-search', config: { sampleOverCapGlobResults: false } },
    { id: 'tool-jobs', name: '@deepseek-ai/dsh-tool-jobs' },
    { id: 'skill-filesystem', name: '@deepseek-ai/dsh-skill-filesystem' },
    { id: 'tool-skill', name: '@deepseek-ai/dsh-tool-skill' },
    { id: 'command-goal', name: '@deepseek-ai/dsh-command-goal' },
    { id: 'tool-goal', name: '@deepseek-ai/dsh-tool-goal' },
    {
      id: 'compaction', name: 'cordis:group', group: true, isolate: { compaction: true, toolResultPruner: true },
      config: [
        { id: 'compaction-basic', name: '@deepseek-ai/dsh-compaction-basic' },
        { id: 'command-compact', name: '@deepseek-ai/dsh-command-compact' },
        { id: 'tool-result-pruner', name: '@deepseek-ai/dsh-compaction-tool-result-pruner', config: { thresholdChars: 8192, headChars: 4096, tailChars: 1024 } },
      ],
    },
    {
      id: 'delegation', name: 'cordis:group', group: true, isolate: { workflowEngine: true },
      config: [
        { id: 'tool-subagent-control', name: '@deepseek-ai/dsh-tool-subagent-control' },
        { id: 'tool-subagent-list-agents', name: '@deepseek-ai/dsh-tool-subagent-control/list-agents' },
        { id: 'tool-subagent', name: '@deepseek-ai/dsh-tool-subagent', config: { provider: 'spawn', toolName: 'subagent', modelSelectionSettings: true } },
        { id: 'tool-subagent-fork', name: '@deepseek-ai/dsh-tool-subagent', config: { provider: 'fork', toolName: 'subagent_fork' } },
        { id: 'workflow-ptc', name: '@deepseek-ai/dsh-workflow-ptc', config: { provider: 'spawn' } },
        { id: 'tool-workflow', name: '@deepseek-ai/dsh-tool-workflow' },
      ],
    },
    { id: 'tool-todo', name: '@deepseek-ai/dsh-tool-todo', config: { allowParallelInProgress: true } },
    { id: 'tool-web', name: '@deepseek-ai/dsh-tool-web', config: { fetch: true, search, searchTimeoutMs: 60000 } },
    { id: 'present', name: '@deepseek-ai/dsh-tool-present' },
    { id: 'tool-plugin-manager', name: '@deepseek-ai/dsh-plugin-manager/tools', disabled: true },
  ];
  return { id: 'standard', order: 1, plugins };
}
