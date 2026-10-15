/**
 * The Strata profile for the Harness, generated from what the host knows right now.
 *
 * The static profile in `integrations/harness/profile/cordis.patch.yml` names one local route on a
 * fixed port. The app needs more than that: whichever local Strata instances are ready, and one route
 * per saved API connection, so that each chat can pick its own model. This module writes the same two
 * Cordis entries (`agent-default-model`, `llm-pi-ai`) for the current set.
 *
 * Rules carried over from the integration notes (docs/work/IMPLEMENTATION_STATUS.md section 1.8):
 *   - `llm-pi-ai` config is `{ providers: { "<route>": {...} } }`; a route key at the top level is
 *     silently ignored.
 *   - the output budget must fit the whole request, so `maxTokens` is derived from the real context.
 *   - no cloud route is declared unless the user saved that connection, so nothing can fall back to a
 *     hosted model by itself.
 *
 * Secrets: a route names a credential by an environment-variable NAME (`apiKeyEnv`). The values are
 * returned separately in `env` for the host to hand to the agent process; they are never written to
 * the profile file, a prompt, or a log.
 */
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { standardPreset } from './preset.mjs';

const PROTOCOL = { 'openai-chat': 'openai-completions', 'openai-responses': 'openai-responses', anthropic: 'anthropic-messages' };

/** A conservative window for an API model whose connection does not report one. */
export const ASSUMED_API_CONTEXT = 128_000;

/**
 * The three chat modes as Harness permission presets (see README, "Modes"). A preset is a
 * sandbox mode plus an approval policy; the Harness switches them with its own `/permission` command.
 *   manual  open sandbox + ask:        the Strata gate (plugins/strata-gate.mjs) asks BEFORE any change runs; nothing is denied first
 *   plan    read-only sandbox + never: nothing can change, and the agent is told not to ask
 *   bypass  danger-full-access + never: the Harness's own full-access, never-ask bundle (its `danger-full-access` preset)
 */
export const MODE_PRESETS = Object.freeze({
  manual: { key: 'strata-manual', sandbox: 'danger-full-access', approval: 'ask', name: 'Manual', description: 'Asks before changing a file or running a command that changes anything.' },
  plan: { key: 'strata-plan', sandbox: 'read-only', approval: 'never', name: 'Plan', description: 'Read-only. Nothing is changed and no approval prompts appear.' },
  bypass: { key: 'strata-bypass', sandbox: 'danger-full-access', approval: 'never', name: 'Bypass', description: 'Full access without approval prompts. Files and commands anywhere on this computer run without asking.' },
});
export const DEFAULT_MODE = 'manual';
/** Helper agents running at once when no local engine sets the limit (the Harness's own default). */
export const DEFAULT_API_HELPERS = 8;
/**
 * Helpers a local chat may have at once: at least the engine's request slots, and never fewer than this. With one slot the engine
 * serves its requests one after the other (its queue is unbounded), so the extra helpers show as waiting for their turn instead
 * of being refused.
 */
export const MIN_LOCAL_HELPERS = 4;
/** The preset for a chat's stored mode. Anything unknown (older values such as `auto`) gets the strictest ask-first mode. */
export const modeOf = value => (Object.hasOwn(MODE_PRESETS, value) ? value : DEFAULT_MODE);

/** The approval gate plugin (plain ESM, loaded by absolute path). */
export const GATE_PLUGIN = fileURLToPath(new URL('./plugins/strata-gate.mjs', import.meta.url));

/**
 * The tool servers (MCP) set up for the model in Settings, as Harness client rows. Their tools appear to the agent as
 * `mcp__<server>__<tool>`. A command or URL goes into the patch as text; a secret (an env value or a header) never does: the row
 * carries `!!js process.env.<NAME>` and the value travels in the agent process's environment (returned in `env`).
 */
export function mcpRows(servers, env) {
  const rows = [];
  let index = 0;
  for (const [name, server] of Object.entries(servers ?? {})) {
    if (!server || server.disabled === true || !/^[A-Za-z0-9_-]{1,32}$/.test(name)) continue;
    const token = (kind, key, value) => {
      const variable = `STRATA_MCP_${index}_${kind}_${slug(key).replace(/-/g, '_').toUpperCase()}`;
      env[variable] = String(value);
      return `@@JS:process.env.${variable}@@`;
    };
    const config = { serverName: name, ...(server.command
      ? { transport: 'stdio', command: String(server.command), args: Array.isArray(server.args) ? server.args.map(String) : [], ...(server.cwd ? { cwd: String(server.cwd) } : {}),
        ...(server.env && Object.keys(server.env).length ? { env: Object.fromEntries(Object.entries(server.env).map(([k, v]) => [k, token('E', k, v)])) } : {}) }
      : { transport: 'streamable-http', url: String(server.url), ...(server.headers && Object.keys(server.headers).length ? { headers: Object.fromEntries(Object.entries(server.headers).map(([k, v]) => [k, token('H', k, v)])) } : {}) }) };
    rows.push({ id: `mcp-${slug(name)}`, name: '@deepseek-ai/dsh-mcp-client', config });
    index += 1;
  }
  return rows;
}

const slug = value => String(value).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 24) || 'x';

/** Identity of one route's wire configuration, secret included by hash only. */
function fingerprint(provider, secret) {
  return createHash('sha256').update(JSON.stringify(provider)).update('|').update(secret ?? '').digest('hex').slice(0, 16);
}

/**
 * Output budget for an API model, whose real limit the app does not know: a quarter of the window, between 1K and 8K.
 * (A local model has its own, much larger budget: `localOutputBudget`.)
 */
/**
 * Output budget for a LOCAL model: nearly the whole window, so a thinking model is never cut off mid-thought by the app's own
 * limit (it was 8192, which stopped big coding answers at exactly 8192 tokens). The engine clamps the request to what is left
 * of the context (`fit_max_tokens`, which the app turns on for the models it loads), so a long prompt never turns this into an error.
 */
export function localOutputBudget(context) {
  const c = Number.isFinite(context) && context > 0 ? context : 32768;
  return Math.max(8192, c - 4096);
}

/**
 * The thinking levels the chat's Reasoning menu offers, as wire values for a local Strata model. `off` sends enable_thinking false;
 * the rest send enable_thinking true plus the effort (the server reads both; "Max" is the highest effort it has).
 */
export const LOCAL_EFFORTS = Object.freeze({ off: null, low: 'low', medium: 'medium', high: 'high', max: 'high' });
/** The menu's words to the Harness's level names. */
export const EFFORT_LEVEL = Object.freeze({ off: 'off', low: 'low', medium: 'medium', high: 'high', max: 'max' });
export const effortOf = value => EFFORT_LEVEL[String(value ?? '').toLowerCase()] ?? null;

export function outputBudget(context) {
  const c = Number.isFinite(context) && context > 0 ? context : 32768;
  return Math.max(1024, Math.min(8192, Math.floor(c / 4)));
}

/**
 * @param {object} input
 * @param {Array<{id:string,label?:string,baseUrl:string,modelId:string,context?:number,key?:string,vision?:boolean,slots?:number}>} input.local
 *        ready local instances; the first is the default route (`strata-local`)
 * @param {Array<{id:string,name:string,baseUrl:string,protocol:string,apiKey?:string,models:Array<{id:string,name?:string,contextMax?:number|null}>}>} input.connections
 * @returns {{patch:string, env:Record<string,string>, routes:Array, defaultRoute:object|null, signature:string}}
 */
export function buildProfile({ local = [], connections = [], gate = true }) {
  const providers = {};
  const env = {};
  const routes = [];
  const mcpEnv = {};
  // Helper agents: at least as many as the engine has request slots (a local model); API models take the Harness's own limit.
  let slots = 0;

  local.forEach((inst, index) => {
    const route = index === 0 ? 'strata-local' : `strata-local-${slug(inst.id)}`;
    const keyEnv = `STRATA_LOCAL_KEY_${index}`;
    // pi-ai's OpenAI-compatible client insists on some key before it sends anything. A worker that
    // does not authenticate gets a fixed placeholder; one that does gets its real key, by reference.
    env[keyEnv] = inst.key || 'strata-local';
    const context = inst.context ?? 32768;
    providers[route] = {
      displayName: inst.label ? `Strata (local) ${inst.label}` : 'Strata (local)',
      api: 'openai-completions',
      baseURL: inst.baseUrl,
      apiKeyEnv: keyEnv,
      // `qwen` thinking format: enable_thinking true/false plus reasoning_effort, which the Strata server reads.
      compat: { supportsStore: false, thinkingFormat: 'qwen', supportsReasoningEffort: true },
      models: [{
        id: inst.modelId, name: inst.label || inst.modelId, contextWindow: context, maxTokens: localOutputBudget(context),
        input: inst.vision ? ['text', 'image'] : ['text'],
        reasoning: true, reasoningEfforts: LOCAL_EFFORTS,
      }],
    };
    slots = Math.max(slots, Number.isFinite(inst.slots) && inst.slots >= 1 ? Math.floor(inst.slots) : 1);
    routes.push({ route, fp: fingerprint(providers[route], env[keyEnv]), kind: 'local', instanceId: inst.id, modelId: inst.modelId, name: inst.label || inst.modelId, contextWindow: context });
  });

  connections.forEach(conn => {
    const protocol = PROTOCOL[conn.protocol];
    if (!protocol) return;
    const route = `api-${slug(conn.id).slice(0, 12)}`;
    const keyEnv = `STRATA_API_KEY_${slug(conn.id).replace(/-/g, '_').toUpperCase().slice(0, 12)}`;
    if (conn.apiKey) env[keyEnv] = conn.apiKey;
    let baseURL = conn.baseUrl;
    if (protocol === 'anthropic-messages') baseURL = baseURL.replace(/\/v1\/?$/, '');
    providers[route] = {
      displayName: conn.name,
      api: protocol,
      baseURL,
      ...(conn.apiKey ? { apiKeyEnv: keyEnv } : {}),
      models: conn.models.map(m => ({
        id: m.id, name: m.name || m.id,
        contextWindow: m.contextMax ?? ASSUMED_API_CONTEXT, maxTokens: outputBudget(m.contextMax ?? ASSUMED_API_CONTEXT),
        input: ['text'],
      })),
    };
    const fp = fingerprint(providers[route], conn.apiKey ?? '');
    for (const m of conn.models) routes.push({ route, fp, kind: 'api', connectionId: conn.id, modelId: m.id, name: m.name || m.id, contextWindow: m.contextMax ?? ASSUMED_API_CONTEXT });
  });

  const defaultRoute = routes.find(r => r.kind === 'local') ?? null;
  // One Harness, one set of tool servers: the default local model's.
  const mcp = mcpRows(local[0]?.mcpServers, mcpEnv);
  Object.assign(env, mcpEnv);
  const entries = [
    ...(defaultRoute ? [{ id: 'agent-default-model', name: '@deepseek-ai/dsh-agent-default-model', config: { provider: defaultRoute.route, model: defaultRoute.modelId } }] : []),
    { id: 'llm-pi-ai', name: '@deepseek-ai/dsh-llm-pi-ai', config: { providers } },
    // The chat modes. Replacing the table also drops the Harness's own three presets: a chat is always in one of ours.
    { id: 'permission', config: { presets: Object.fromEntries(Object.values(MODE_PRESETS).map(({ key, sandbox, approval, name, description }) => [key, { sandbox, approval, name, description }])), defaultPreset: MODE_PRESETS[DEFAULT_MODE].key } },
    { id: 'subagent', config: { maxActiveSubagents: slots > 0 ? Math.max(slots, MIN_LOCAL_HELPERS) : DEFAULT_API_HELPERS } },
    // Manual asks before a change runs, so the person sees one Approve/Deny card and nothing has happened yet.
    ...(gate ? [{ insert: [{ id: 'strata-gate', name: GATE_PLUGIN }] }] : []),
    ...(mcp.length ? [{ insert: mcp }] : []),
    // The reminder tools would be a second scheduler beside the app's Tasks; web search needs a search-service key the app has none of.
    { id: 'preset-standard', name: '@deepseek-ai/dsh-agent-preset', config: standardPreset() },
    // Strata names its chats itself; the Harness's own title request would be an extra model call on
    // every first prompt (and, for an API route, an unrequested one).
    { id: 'session-title-llm', disabled: true },
    // Office documents are not previewed in the app (the converter is not shipped): do not try to start it.
    { id: 'office-to-pdf', disabled: true },
  ];
  // JSON is valid YAML, and a JSON document cannot be mis-quoted by a model id or a display name.
  // JSON is valid YAML. A tool server's secret is a `!!js process.env.NAME` value, swapped in after stringifying.
  const json = JSON.stringify(entries, null, 2).replace(/"@@JS:([A-Za-z0-9_.]+)@@"/g, '!!js $1');
  const patch = `# Generated by the Strata host. Contains credential NAMES only, never values.\n${json}\n`;
  // The signature decides whether the running agent must be restarted. It covers the secret values
  // through a hash so a changed key is noticed, and the hash is never reversible into the key.
  const signature = createHash('sha256').update(patch).update(JSON.stringify(Object.entries(env).sort())).digest('hex').slice(0, 16);
  return { patch, env, routes, defaultRoute, signature };
}

/** Find the route for a chat's model choice. */
export function routeFor(profile, choice) {
  if (!choice) return profile.defaultRoute;
  if (choice.connectionId) return profile.routes.find(r => r.kind === 'api' && r.connectionId === choice.connectionId && r.modelId === choice.modelId) ?? null;
  if (choice.instanceId) return profile.routes.find(r => r.kind === 'local' && r.instanceId === choice.instanceId) ?? null;
  return profile.defaultRoute;
}
