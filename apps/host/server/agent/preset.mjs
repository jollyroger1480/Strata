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
 *   - Prompt size: a local model reads every tool schema and every instruction on every first message (about 9,000 tokens at
 *     the start, 20 s before the first word on a 64K model). The goal tools, the skill tool (no skill ships), and the
 *     workflow tool (3.4 KB of schema on its own) are left out; they are the largest and the least used. Helper agents,
 *     files, shell, search, todo, web fetch and present stay.
 */

/** Plugin ids of the pinned Harness `standard` preset, in its order (the drift check). */
export const STANDARD_PLUGIN_IDS = [
  'persona', 'agent-instructions', 'time-context', 'tool-bash', 'tool-pwsh', 'tool-fs', 'tool-fs-search', 'tool-jobs',
  'tool-schedule', 'skill-filesystem', 'tool-skill', 'command-goal', 'tool-goal', 'planning', 'compaction', 'delegation',
  'tool-ask-user', 'tool-todo', 'tool-web', 'present', 'tool-plugin-manager',
];
/** What we leave out of it. */
export const REMOVED_PLUGIN_IDS = ['tool-schedule', 'tool-ask-user', 'planning', 'skill-filesystem', 'tool-skill', 'command-goal', 'tool-goal'];
/** Plugins removed from inside the `delegation` group (the group stays: helper agents are a feature). */
export const REMOVED_DELEGATION_IDS = ['workflow-ptc', 'tool-workflow', 'tool-subagent-fork'];

/**
 * The whole system prompt of an app chat. The Harness's own text is about 4.8 KB (about 1,200 tokens) of guidance for working on
 * the Harness itself (its checkout, its web page, its plugins): a local model pays for it on every first message and the person
 * can see its words in the chat ("DSH", "sandbox"). This one says what a chat needs and nothing else. The tools state their own rules.
 */
export const AGENT_PROMPT = [
  'You are a helpful assistant that can also work on this computer through your tools. You are running as {{model}}.',
  'Answer questions directly and briefly; use a tool only when the task needs one. The person reads your replies in a chat: use plain words, not tool names or technical jargon.',
  'Look before you change: read a file before you edit or overwrite it, and prefer a small edit to a rewrite. Search with the glob and grep tools and read files with the read tool; use the shell for running programs.',
  'When you start a program or open a folder or file for the person, the command returning without an error means it worked: say plainly that it is open. Do not say you could not confirm it. If a command fails, say what went wrong in one sentence and what you will try next.',
  'Tokens prefixed with @ are files or folders the person pointed at. A failed or refused action changed nothing; say so, and do not claim work you did not do.',
].join('\n\n');

/**
 * @param {{platform?: string, search?: boolean}} [options]
 * @returns {object} the `config` of the `preset-standard` row
 */
export function standardPreset({ platform = process.platform, search = false } = {}) {
  const windows = platform === 'win32';
  const plugins = [
    { id: 'persona', name: '@deepseek-ai/dsh-persona', config: { prefix: AGENT_PROMPT, complete: true, includeRuntimeContext: false } },
    { id: 'agent-instructions', name: '@deepseek-ai/dsh-agent-instructions', config: { maxBytes: 65536 } },
    { id: 'time-context', name: '@deepseek-ai/dsh-time-context' },
    windows
      ? { id: 'tool-pwsh', name: '@deepseek-ai/dsh-tool-pwsh' }
      : { id: 'tool-bash', name: '@deepseek-ai/dsh-tool-bash' },
    { id: 'tool-fs', name: '@deepseek-ai/dsh-tool-fs' },
    { id: 'tool-fs-search', name: '@deepseek-ai/dsh-tool-fs-search', config: { sampleOverCapGlobResults: false } },
    { id: 'tool-jobs', name: '@deepseek-ai/dsh-tool-jobs' },
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
      ],
    },
    { id: 'tool-todo', name: '@deepseek-ai/dsh-tool-todo', config: { allowParallelInProgress: true } },
    { id: 'tool-web', name: '@deepseek-ai/dsh-tool-web', config: { fetch: true, search, searchTimeoutMs: 60000 } },
    { id: 'present', name: '@deepseek-ai/dsh-tool-present' },
    { id: 'tool-plugin-manager', name: '@deepseek-ai/dsh-plugin-manager/tools', disabled: true },
  ];
  return { id: 'standard', order: 1, plugins };
}
