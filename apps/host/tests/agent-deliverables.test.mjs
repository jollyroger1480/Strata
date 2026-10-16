// Deliverable events and compact activity groups.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DeliverableTracker, ActivityGroups, fileType, groupLine, lineDelta } from '../server/agent/deliverables.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const NL = String.fromCharCode(10), MINUS = String.fromCharCode(0x2212);

test('file types, line deltas and the compact line', () => {
  assert.deepEqual(['a.html', 'b.MD', 'c.pdf', 'd.csv', 'e.PNG', 'f.py', 'g.txt', 'h.zip'].map(fileType), ['html', 'markdown', 'pdf', 'csv', 'image', 'code', 'text', 'other']);
  assert.deepEqual(lineDelta(['a', 'b', 'c'], ['a', 'c', 'd', 'e']), { added: 2, removed: 1 });
  assert.deepEqual(lineDelta([], ['x']), { added: 1, removed: 0 });
  assert.equal(groupLine({ command: 2, read: 1 }, 225, 0), 'Ran 2 commands, read a file +225 ' + MINUS + '0');
  assert.equal(groupLine({ read: 3 }, 0, 0), 'Read 3 files');
  assert.equal(groupLine({ edit: 1, search: 2 }, 5, 2), 'Edited a file, searched 2 searches +5 ' + MINUS + '2');
});

test('a created file, a changed file and a script-made file are reported once each, with sizes and line changes', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-dl-'));
  try {
    writeFileSync(join(dir, 'notes.md'), ['one', 'two', 'three', ''].join(NL));
    const t = new DeliverableTracker(dir); t.snapshot();
    assert.equal(t.check('notes.md'), null, 'unchanged: nothing');
    writeFileSync(join(dir, 'page.html'), ['<h1>x</h1>', '<p>y</p>', ''].join(NL));
    const made = t.check('page.html');
    assert.deepEqual([made.type, made.status, made.kind, made.lines, made.added, made.removed, made.name], ['deliverable', 'created', 'html', 2, 2, 0, 'page.html']);
    assert.equal(made.size, 20); assert.equal(t.check('page.html'), null, 'the same state is not reported twice');
    writeFileSync(join(dir, 'notes.md'), ['one', 'TWO', 'three', 'four', ''].join(NL));
    const ch = t.check('notes.md');
    assert.deepEqual([ch.status, ch.added, ch.removed, ch.lines], ['changed', 2, 1, 4]);
    mkdirSync(join(dir, 'out')); writeFileSync(join(dir, 'out', 'data.csv'), ['a,b', '1,2', ''].join(NL)); writeFileSync(join(dir, 'pic.png'), Buffer.from([137, 80, 78, 71, 0, 1]));
    mkdirSync(join(dir, '.git')); writeFileSync(join(dir, '.git', 'x'), 'x'); mkdirSync(join(dir, 'node_modules')); writeFileSync(join(dir, 'node_modules', 'y.js'), 'y');
    const swept = t.sweep();
    assert.deepEqual(swept.map(d => d.path).sort(), ['out/data.csv', 'pic.png'], 'hidden and dependency folders are not deliverables');
    assert.equal(swept.find(d => d.path === 'pic.png').lines, null); assert.equal(swept.find(d => d.path === 'pic.png').kind, 'image');
    assert.equal(t.check('../outside.txt'), null); assert.equal(t.check('missing.txt'), null);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('activity groups: consecutive tools are one growing group closed by text, with counts, files and the diff stat', () => {
  const out = [];
  const g = new ActivityGroups({ emit: f => out.push(f) });
  g.card({ id: '1', name: 'pwsh', state: 'running' });
  assert.equal(out.length, 0, 'a running card is not counted yet');
  g.card({ id: '1', name: 'pwsh', state: 'done' }); g.card({ id: '2', name: 'bash', state: 'done' }); g.card({ id: '3', name: 'read', state: 'done' });
  g.card({ id: '4', name: 'write', state: 'done' }, { path: 'a.html', name: 'a.html', kind: 'html', status: 'created', added: 225, removed: 0 });
  g.card({ id: '4', name: 'write', state: 'done' });
  g.text();
  const last = out.at(-1);
  assert.equal(last.closed, true); assert.deepEqual(last.counts, { command: 2, read: 1, edit: 1 }); assert.equal(last.added, 225);
  assert.equal(last.line, 'Ran 2 commands, read a file, edited a file +225 ' + MINUS + '0'); assert.deepEqual(last.files.map(f => f.path), ['a.html']);
  assert.equal(new Set(out.map(f => f.id)).size, 1, 'one id while it grows');
  g.card({ id: '5', name: 'grep', state: 'done' }); g.text();
  assert.notEqual(out.at(-1).id, last.id, 'text starts a new group');
  const n = out.length; g.card({ id: '6', name: 'write', state: 'denied' }); g.text();
  assert.equal(out.length, n, 'a refused tool is not an activity');
});

test('a turn sends deliverable and activity-group frames in order, before the end', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-dl2-')); const ws = join(dir, 'ws'); mkdirSync(ws);
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }], connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const fwd = p => p.split(String.fromCharCode(92)).join('/');
  const out = [];
  const content = ['<h1>a</h1>', '<p>b</p>', ''].join(NL);
  await bridge.runTurn({ chatId: 'd1', message: 'SCRIPT:call write ' + JSON.stringify({ file_path: fwd(join(ws, 'page.html')), content }), workspacePath: ws, mode: 'bypass', onFrame: f => out.push(f) });
  const dl = out.find(f => f.type === 'deliverable');
  assert.deepEqual([dl.path, dl.kind, dl.status, dl.lines, dl.added], ['page.html', 'html', 'created', 2, 2]);
  const groups = out.filter(f => f.type === 'activity-group');
  assert.equal(groups.at(-1).closed, true); assert.deepEqual(groups.at(-1).counts, { edit: 1 }); assert.equal(groups.at(-1).line, 'Edited a file +2 ' + MINUS + '0');
  assert.ok(out.indexOf(dl) < out.findIndex(f => f.type === 'done'));
  assert.equal(readFileSync(join(ws, 'page.html'), 'utf8').length, dl.size);
});
