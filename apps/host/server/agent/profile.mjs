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

const PROTOCOL = { 'openai-chat': 'openai-completions', 'openai-responses': 'openai-responses', anthropic: 'anthropic-messages' };

/** A conservative window for an API model whose connection does not report one. */
export const ASSUMED_API_CONTEXT = 128_000;

/**
 * The three chat modes as Harness permission presets (see README, "Modes"). A preset is a
 * sandbox mode plus an approval policy; the Harness switches them with its own `/permission` command.
 *   manual  read-only sandbox + ask:   any change is denied by the sandbox until the user approves a one-shot widening
 *   plan    read-only sandbox + never: nothing can change, and the agent is told not to ask
 *   bypass  workspace-write + never:   changes inside the workspace without prompts; anything wider is refused
 */
export const MODE_PRESETS = Object.freeze({
  manual: { key: 'strata-manual', sandbox: 'read-only', approval: 'ask', name: 'Manual', description: 'Asks before changing a file or running a command that changes anything.' },
  plan: { key: 'strata-plan', sandbox: 'read-only', approval: 'never', name: 'Plan', description: 'Read-only. Nothing is changed and no approval prompts appear.' },
  bypass: { key: 'strata-bypass', sandbox: 'workspace-write', approval: 'never', name: 'Bypass', description: 'Changes inside the workspace without asking. Nothing outside the workspace.' },
});
export const DEFAULT_MODE = 'manual';
/** The preset for a chat's stored mode. Anything unknown (older values such as `auto`) gets the strictest ask-first mode. */
export const modeOf = value => (Object.hasOwn(MODE_PRESETS, value) ? value : DEFAULT_MODE);

const slug = value => String(value).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 24) || 'x';

/** Identity of one route's wire configuration, secret included by hash only. */
function fingerprint(provider, secret) {
  return createHash('sha256').update(JSON.stringify(provider)).update('|').update(secret ?? '').digest('hex').slice(0, 16);
}

/** Output budget that leaves the prompt room: a quarter of the window, between 1K and 8K. */
export function outputBudget(context) {
  const c = Number.isFinite(context) && context > 0 ? context : 32768;
  return Math.max(1024, Math.min(8192, Math.floor(c / 4)));
}

/**
 * @param {object} input
 * @param {Array<{id:string,label?:string,baseUrl:string,modelId:string,context?:number,key?:string,vision?:boolean}>} input.local
 *        ready local instances; the first is the default route (`strata-local`)
 * @param {Array<{id:string,name:string,baseUrl:string,protocol:string,apiKey?:string,models:Array<{id:string,name?:string,contextMax?:number|null}>}>} input.connections
 * @returns {{patch:string, env:Record<string,string>, routes:Array, defaultRoute:object|null, signature:string}}
 */
export function buildProfile({ local = [], connections = [] }) {
  const providers = {};
  const env = {};
  const routes = [];

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
      compat: { supportsStore: false },
      models: [{
        id: inst.modelId, name: inst.label || inst.modelId, contextWindow: context, maxTokens: outputBudget(context),
        input: inst.vision ? ['text', 'image'] : ['text'],
      }],
    };
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
  const entries = [
    ...(defaultRoute ? [{ id: 'agent-default-model', name: '@deepseek-ai/dsh-agent-default-model', config: { provider: defaultRoute.route, model: defaultRoute.modelId } }] : []),
    { id: 'llm-pi-ai', name: '@deepseek-ai/dsh-llm-pi-ai', config: { providers } },
    // The chat modes. Replacing the table also drops the Harness's own three presets: a chat is always in one of ours.
    { id: 'permission', config: { presets: Object.fromEntries(Object.values(MODE_PRESETS).map(({ key, sandbox, approval, name, description }) => [key, { sandbox, approval, name, description }])), defaultPreset: MODE_PRESETS[DEFAULT_MODE].key } },
    // Strata names its chats itself; the Harness's own title request would be an extra model call on
    // every first prompt (and, for an API route, an unrequested one).
    { id: 'session-title-llm', disabled: true },
  ];
  // JSON is valid YAML, and a JSON document cannot be mis-quoted by a model id or a display name.
  const patch = `# Generated by the Strata host. Contains credential NAMES only, never values.\n${JSON.stringify(entries, null, 2)}\n`;
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
