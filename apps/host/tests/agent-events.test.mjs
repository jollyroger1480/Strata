/** Pure parts of the agent bridge: profile generation, environment scrubbing, event projection. No Harness needed. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildProfile, outputBudget, routeFor, ASSUMED_API_CONTEXT } from '../server/agent/profile.mjs';
import { agentEnvironment, redact } from '../server/agent/harness-runtime.mjs';
import { TurnProjector, presentTool, toolScope, friendlyFailure } from '../server/agent/events.mjs';
import { withEarlierConversation } from '../server/agent/agent-bridge.mjs';

const local = [{ id: 'inst-1', label: 'Qwen', baseUrl: 'http://127.0.0.1:8095/v1', modelId: 'qwen', context: 32768, key: 'local-secret-key-1' }];
const conn = { id: 'c0nn-1234-abcd', name: 'Cloud', baseUrl: 'https://api.example.com/v1', protocol: 'openai-chat', apiKey: 'sk-live-ABCDEFGH', models: [{ id: 'm1', name: 'M1', contextMax: 64000 }, { id: 'm2', name: 'M2', contextMax: null }] };

test('the generated profile names credentials, never carries them, and declares only what exists', () => {
  const p = buildProfile({ local, connections: [conn] });
  assert.ok(!p.patch.includes('sk-live-ABCDEFGH') && !p.patch.includes('local-secret-key-1'));
  assert.match(p.patch, /apiKeyEnv/);
  assert.deepEqual(Object.values(p.env).sort(), ['local-secret-key-1', 'sk-live-ABCDEFGH']);
  const entries = JSON.parse(p.patch.replace(/^#.*\n/, ''));
  const providers = entries.find(e => e.id === 'llm-pi-ai').config.providers;
  assert.deepEqual(Object.keys(providers).sort(), ['api-c0nn-1234-ab', 'strata-local']);
  assert.equal(providers['strata-local'].models[0].contextWindow, 32768);
  assert.equal(providers['strata-local'].models[0].maxTokens, 28672, 'a local model may use nearly the whole window (it was 8192, which cut big answers off)');
  assert.equal(providers['strata-local'].models[0].reasoning, true);
  assert.equal(providers['strata-local'].compat.thinkingFormat, 'qwen');
  assert.equal(providers['api-c0nn-1234-ab'].models[1].contextWindow, ASSUMED_API_CONTEXT, 'an unknown API window is a stated assumption, not a catalog default');
  assert.equal(entries.find(e => e.id === 'agent-default-model').config.provider, 'strata-local', 'local stays the default');
  assert.equal(entries.find(e => e.id === 'session-title-llm').disabled, true);
});

test('no local model means no default model, so nothing can quietly fall to a cloud route', () => {
  const p = buildProfile({ local: [], connections: [conn] });
  assert.equal(p.defaultRoute, null);
  assert.ok(!JSON.parse(p.patch.replace(/^#.*\n/, '')).some(e => e.id === 'agent-default-model'));
  assert.equal(routeFor(p, null), null);
});

test('a route fingerprint changes when its key or address changes, and only then', () => {
  const a = buildProfile({ local, connections: [conn] });
  const b = buildProfile({ local, connections: [{ ...conn, apiKey: 'sk-live-OTHER-KEY' }] });
  const c = buildProfile({ local: [{ ...local[0], context: 16384 }], connections: [conn] });
  const fp = (p, kind) => p.routes.find(r => r.kind === kind).fp;
  assert.notEqual(fp(a, 'api'), fp(b, 'api'));
  assert.equal(fp(a, 'local'), fp(b, 'local'));
  assert.notEqual(fp(a, 'local'), fp(c, 'local'));
  assert.equal(fp(a, 'api'), fp(c, 'api'));
});

test('routeFor picks the chat\'s own choice', () => {
  const p = buildProfile({ local, connections: [conn] });
  assert.equal(routeFor(p, { connectionId: conn.id, modelId: 'm2' }).modelId, 'm2');
  assert.equal(routeFor(p, { instanceId: 'inst-1' }).route, 'strata-local');
  assert.equal(routeFor(p, { connectionId: 'gone', modelId: 'm1' }), null);
  assert.equal(outputBudget(2048), 1024); assert.equal(outputBudget(1_000_000), 8192);
});

test('the agent environment keeps what a shell needs and drops ambient credentials', () => {
  const env = agentEnvironment({ PATH: '/bin', HOME: '/h', OPENAI_API_KEY: 'x', GITHUB_TOKEN: 'y', AWS_SECRET_ACCESS_KEY: 'z', SSH_AUTH_SOCK: 's', SystemRoot: 'C:\\Windows' }, { STRATA_API_KEY_X: 'k' });
  assert.deepEqual(Object.keys(env).sort(), ['DSH_PERMISSION_MODE', 'DSH_TELEMETRY_DISABLED', 'HOME', 'PATH', 'STRATA_API_KEY_X', 'SystemRoot']);
});

test('redact removes secret values and launch tokens from log text', () => {
  const out = redact('GET /?token=abcDEF_123-x failed with key sk-live-ABCDEFGH', ['sk-live-ABCDEFGH']);
  assert.ok(!out.includes('sk-live-ABCDEFGH') && !out.includes('abcDEF_123-x'));
});

const ev = (type, data, seq = 1, time = 1000) => ({ type: 'event', event: { type, seq, time, data } });
const chunk = c => ({ type: 'assistant-stream', frame: { type: 'chunk', attemptId: 'a:1', chunk: c } });

test('a tool call walks running -> proposed -> done with its approval, and carries its result', () => {
  const p = new TurnProjector({ startedAt: 0 });
  const call = p.consume(ev('tool/call', { callId: 'c1', name: 'pwsh', arguments: JSON.stringify({ command: 'ls', description: 'List files' }) }, 1, 1000));
  assert.equal(call[0].tool.state, 'running'); assert.equal(call[0].tool.scope, 'shell'); assert.equal(call[0].tool.preview, 'ls');
  const asked = p.consume(ev('approval/asked', { id: 'ap1', callId: 'c1', reason: 'escalate sandbox' }));
  assert.equal(asked[0].tool.state, 'proposed'); assert.match(asked[0].tool.effect, /escalate/);
  const decided = p.consume(ev('approval/decided', { id: 'ap1', outcome: 'allowed-once' }));
  assert.equal(decided[0].tool.state, 'running');
  const done = p.consume(ev('tool/result', { message: { toolCallId: 'c1', content: [{ type: 'text', text: 'ok' }] } }, 3, 1500));
  assert.equal(done[0].tool.state, 'done'); assert.equal(done[0].tool.result, 'ok'); assert.equal(done[0].tool.durationMs, 500);
});

test('a rejected approval ends as denied even though the Harness returns an error result', () => {
  const p = new TurnProjector({ startedAt: 0 });
  p.consume(ev('tool/call', { callId: 'c2', name: 'pwsh', arguments: '{}' }));
  p.consume(ev('approval/asked', { id: 'ap2', callId: 'c2' }));
  assert.equal(p.consume(ev('approval/decided', { id: 'ap2', outcome: 'rejected' }))[0].tool.state, 'denied');
  const res = p.consume(ev('tool/result', { message: { toolCallId: 'c2', isError: true, content: [{ type: 'text', text: 'the user rejected tool' }] } }));
  assert.equal(res[0].tool.state, 'denied');
});

test('live frames become deltas, usage is summed across steps, the turn ends with one terminal frame', () => {
  const p = new TurnProjector({ startedAt: 0 });
  assert.deepEqual(p.consume(chunk({ type: 'text-delta', index: 0, text: 'Hi' })), [{ type: 'delta', text: 'Hi' }]);
  assert.deepEqual(p.consume(chunk({ type: 'reasoning-delta', index: 0, text: 'hmm' })), [{ type: 'reasoning', text: 'hmm' }]);
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 100, outputTokens: 5 } }));
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 120, outputTokens: 7 } }));
  p.consume(ev('turn/end', { turn: 1, reason: { kind: 'completed' } }));
  const [done] = p.finish({ model: 'M', contextMax: 1000 });
  assert.equal(done.type, 'done'); assert.equal(done.tokensIn, 220); assert.equal(done.tokensOut, 12); assert.equal(done.usageSource, 'reported');
});

test('every turn-end reason maps to a frame the chat knows', () => {
  const finish = reason => { const p = new TurnProjector(); p.ended = reason; return p.finish()[0]; };
  assert.equal(finish({ kind: 'aborted', reason: { kind: 'user' } }).cancelled, true);
  assert.equal(finish({ kind: 'max-tokens' }).finishReason, 'length');
  assert.equal(finish({ kind: 'error', error: { code: 'X', message: 'context window exceeded' } }).type, 'error');
  assert.match(finish({ kind: 'error', error: { message: 'request exceeds the context window' } }).message, /no longer fits/);
  assert.equal(finish({ kind: 'blocked' }).code, 'BLOCKED');
  assert.equal(finish(null).code, 'INTERRUPTED');
  assert.equal(finish({ kind: 'completed' }).usageSource, 'unavailable', 'no usage reported means unavailable, not zero');
});

test('a text the live frames missed is completed from the committed message, once', () => {
  const p = new TurnProjector();
  p.consume(chunk({ type: 'text-delta', index: 0, text: 'Hel' }));
  const out = p.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: 'Hello there' }] } }));
  assert.deepEqual(out, [{ type: 'delta', text: 'lo there' }]);
  assert.deepEqual(p.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: 'Hello there' }] } })), []);
});

test('tool scopes and previews', () => {
  assert.equal(toolScope('write'), 'write'); assert.equal(toolScope('grep'), 'read'); assert.equal(toolScope('web_fetch'), 'connected');
  assert.equal(presentTool({ callId: 'x', name: 'write', args: { file_path: 'a.txt' } }).preview, 'a.txt');
  assert.equal(friendlyFailure({ message: 'No API key' }), 'The model rejected the credentials. Check the connection in Settings.');
});

test('a chat that existed before the agent carries a bounded summary into its first agent turn', () => {
  const history = Array.from({ length: 30 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `m${i}`.padEnd(2000, 'x') }));
  const prompt = withEarlierConversation(history, 'NOW');
  assert.match(prompt, /Current message:\nNOW$/);
  assert.ok(prompt.length < 12 * 1600 + 200, 'bounded');
  assert.equal(withEarlierConversation([], 'NOW'), 'NOW');
});
