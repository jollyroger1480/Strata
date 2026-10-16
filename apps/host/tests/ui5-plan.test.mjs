/** Owner round 7: /plan card, plan steps in the sidebar, queued messages. Snapshots from the host replace what was there. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const src = f => fileURLToPath(new URL(`../src/${f}`, import.meta.url));
const stub = fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url));
async function load(file) {
  const result = await build({ entryPoints: [src(`lib/${file}`)], bundle: true, format: 'esm', platform: 'node', write: false, alias: { react: stub }, logLevel: 'silent', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}
globalThis.window ??= globalThis; globalThis.localStorage ??= { getItem: () => null, setItem: () => {} };
globalThis.navigator ??= { language: 'en' };
const P = await load('plan.ts');

const plan = (state, steps) => ({ id: 'p1', goal: 'Pagoda', state, steps });

test('a plan snapshot is read, ordered, with step states; bad parts are dropped', () => {
  const p = P.readPlan(plan('running', [{ id: 'a', text: 'Look', state: 'done' }, { id: 'b', text: 'Write', state: 'active', note: 'one file' }, { text: 'Check' }, { text: '  ' }, 7]));
  assert.equal(p.steps.length, 3);
  assert.deepEqual(p.steps.map(s => s.state), ['done', 'active', 'pending']);
  assert.equal(p.steps[2].id, '3');
  assert.equal(P.stepsDone(p), 1);
  assert.equal(P.readPlan(plan('weird', [{ text: 'x' }])), null);
  assert.equal(P.readPlan(plan('proposed', [])), null);
  assert.equal(P.readPlan(null), null);
});

test('frames replace the plan; null clears it; the same snapshot changes nothing', () => {
  P.applyPlanFrame('c', plan('proposed', [{ text: 'A' }, { text: 'B' }]));
  assert.equal(P.planOf('c').state, 'proposed');
  P.applyPlanFrame('c', plan('running', [{ id: '1', text: 'A', state: 'active' }, { text: 'B' }]));
  assert.equal(P.planOf('c').steps[0].state, 'active');
  assert.equal(P.planOf('other'), null);
  P.applyPlanFrame('c', null);
  assert.equal(P.planOf('c'), null);
});

test('queued messages: a snapshot, the words say which step they wait for, cancel empties', () => {
  P.applyQueueFrame('c', [{ id: 'q1', text: 'Add a moon', after: '2' }, { id: 'q2', text: 'Warmer glow' }, { id: '', text: 'x' }, { id: 'q3' }]);
  const items = P.queueOf('c');
  assert.deepEqual(items.map(q => q.id), ['q1', 'q2']);
  const p = P.readPlan(plan('running', [{ id: '1', text: 'A', state: 'done' }, { id: '2', text: 'B', state: 'active' }]));
  assert.deepEqual(P.queuedWhen(items[0], p), { id: 'Queued — sends when step {0} finishes', args: [2] });
  assert.deepEqual(P.queuedWhen(items[1], p), { id: 'Queued — sends when the assistant is done' });
  P.applyQueueFrame('c', []);
  assert.deepEqual(P.queueOf('c'), []);
});

test('the chat and the sidebar draw the plan from the same store, and the stream feeds it', async () => {
  const [card, chat, store, client] = await Promise.all(['components/PlanCard.tsx', 'screens/Chat.tsx', 'lib/store.ts', 'lib/client.ts'].map(f => readFile(src(f), 'utf8')));
  assert.match(card, /export function PlanCard/); assert.match(card, /export function PlanPanel/); assert.match(card, /export function QueuedMessages/);
  assert.match(card, /Approve/); assert.match(card, /Start as goal/); assert.match(card, /planAction\(chatId, 'reject'\)/);
  assert.match(chat, /<PlanPanel chatId/); assert.match(chat, /plan:\$\{session\.id\}/);
  assert.match(store, /type === 'plan'/); assert.match(store, /type === 'queue'/); assert.match(client, /'plan', 'queue'/);
});
