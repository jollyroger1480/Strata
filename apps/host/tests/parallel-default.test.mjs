/**
 * Several requests at once in the installed app: the default, how it is sized, what a person's own number does, and the
 * retry with one when a load with several slots fails. Nothing here starts an engine; Control is a fake that records what
 * the host saved and asked it to start.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { chooseParallel, applyParallel, withoutParallel, slotGb, DEFAULT_SLOTS, PARALLEL_POLICY, FALLBACK_NOTE } from '../server/native-parallel.mjs';
import { buildProfile } from '../server/agent/profile.mjs';

const hw = (gb, n = 1) => ({ gpus: Array.from({ length: n }, (_, i) => ({ index: i, vram_gb: gb })) });
const cfg = (context, extra = {}) => ({ args: ['--max-context', String(context), '--mtp', '/m', '--spec', '4'], gpu: 0, ...extra });

test('the default is 4 request slots, and the policy is one switch', () => {
  assert.equal(DEFAULT_SLOTS, 4);
  assert.ok(['fit', 'fixed'].includes(PARALLEL_POLICY));
  assert.equal(chooseParallel({ config: cfg(32768) }).slots, 4, 'no hardware read: ask for the default, the engine fits what it can');
});

test('sizing: 4 where the card has room, fewer where the slots would crowd out the expert cache, said in plain words', () => {
  // the slot cost was measured on the 0.1.45 engine: 0.92 GiB per slot at 32K
  assert.ok(Math.abs(slotGb({ context: 32768 }) * 0.931 - 0.92) < 0.1);
  if (PARALLEL_POLICY !== 'fit') return;
  assert.equal(chooseParallel({ config: cfg(32768), hardware: hw(24) }).slots, 4);
  assert.equal(chooseParallel({ config: cfg(32768), hardware: hw(16) }).slots, 3);
  const small = chooseParallel({ config: cfg(32768), hardware: hw(12) });
  assert.equal(small.slots, 2);
  assert.match(small.note, /room for 2 requests at once/);
  assert.match(small.note, /smaller context size leaves room for more/);
  const tight = chooseParallel({ config: cfg(65536), hardware: hw(12) });
  assert.equal(tight.slots, 1); assert.match(tight.note, /one at a time/);
  // two cards share the load
  assert.equal(chooseParallel({ config: cfg(65536, { gpu: [0, 1] }), hardware: hw(24, 2) }).slots, 4);
  assert.doesNotMatch(small.note + tight.note, /--|batch|VRAM|GiB|tok/i, 'no engine words');
});

test('batch-mtp is added where it applies: one card with a draft layer, not on a split, not when already there or one slot', () => {
  const mtp = c => chooseParallel({ config: c, hardware: hw(24) });
  assert.equal(mtp(cfg(32768)).batchMtp, true);
  assert.equal(mtp(cfg(32768, { gpu: [0, 1] })).batchMtp, false, 'measured slower on a split with all experts resident');
  assert.equal(mtp({ args: ['--max-context', '32768'], gpu: 0 }).batchMtp, false, 'no draft layer, nothing to batch');
  assert.equal(mtp(cfg(32768, { args: ['--max-context', '32768', '--mtp', '/m', '--batch-mtp'] })).batchMtp, false);
  const applied = applyParallel(cfg(32768), mtp(cfg(32768)));
  assert.equal(applied.cfg.parallel, 4); assert.ok(applied.cfg.args.includes('--batch-mtp'));
});

test("a number the person chose is kept exactly: Expert mode's value, the saved settings, an explicit --batch", () => {
  assert.deepEqual(applyParallel(cfg(32768, { parallel: 1 }), chooseParallel({ config: cfg(32768, { parallel: 1 }), hardware: hw(24) })).cfg.parallel, 1);
  const six = chooseParallel({ config: cfg(32768), values: { parallel: 6 }, hardware: hw(8) });
  assert.deepEqual([six.slots, six.explicit, six.batchMtp], [6, true, false], 'recommend, never force: 6 on an 8 GB card is allowed');
  const one = chooseParallel({ config: cfg(32768), values: { parallel: 1 }, hardware: hw(24) });
  assert.deepEqual([one.slots, one.explicit], [1, true]);
  assert.equal(chooseParallel({ config: cfg(32768, { args: ['--max-context', '32768', '--batch', '3'] }), hardware: hw(24) }).explicit, true);
  const saved = cfg(32768, { parallel: 3 });
  assert.equal(applyParallel(saved, chooseParallel({ config: saved })).cfg, saved, 'untouched');
});

test('the retry configuration has no slots and no batch-mtp', () => {
  const next = withoutParallel({ ...cfg(32768), parallel: 4, args: [...cfg(32768).args, '--batch-mtp'] });
  assert.equal(next.parallel, undefined); assert.ok(!next.args.includes('--batch-mtp'));
});

test('helper agents never outnumber the engine slots', () => {
  const p = buildProfile({ local: [{ id: 'a', baseUrl: 'http://127.0.0.1:1/v1', modelId: 'm', slots: 4 }] });
  assert.equal(JSON.parse(p.patch.split('\n').slice(1).join('\n')).find(e => e.id === 'subagent').config.maxActiveSubagents, 4);
  assert.notEqual(p.signature, buildProfile({ local: [{ id: 'a', baseUrl: 'http://127.0.0.1:1/v1', modelId: 'm', slots: 2 }] }).signature, 'a changed slot count restarts the agent when idle');
});

// ---------------------------------------------------------------------------------------- the installed app's load

function fixture({ failStarts = 0, config = {} } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-par-test-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  const path = join(install, 'strata-iq3_s.json');
  writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--max-context', '32768', '--mtp', '/m', '--spec', '4'], model_name: 'Qwen iq3_s', port: 8080, ...config }));
  const saved = []; const calls = [];
  let starts = 0;
  const control = {
    control: { dir: join(root, 'control') }, version: '0.1.45',
    async installedModels() { return { install_dir: install, models: [{ id: 'iq3_s', name: 'Qwen iq3_s', source_config: path, component_status: 'present', context_tokens: 32768 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.1.45', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); calls.push(['saveProfile', name, basename(file)]); return {}; },
    async start(name) { calls.push(['start', name]); starts++; if (starts <= failStarts) throw new Error('server_exited: out of memory'); return { ok: true, state: 'ready' }; },
    async stop(name) { calls.push(['stop', name]); return { ok: true }; },
  };
  const fetcher = async () => Response.json({ status: 'ok' });
  const native = createNativeControl({ control, directory: data, spawnProcess: () => { throw new Error('the host starts no process'); }, fetcher, requests: () => [] });
  return { native, saved, calls, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('the installed app loads with 4 slots and batch-mtp by default', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq3_s' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.saved.length, 1);
    assert.equal(f.saved[0].parallel, 4);
    assert.ok(f.saved[0].args.includes('--batch-mtp'));
    assert.equal(f.native.state().instances[0].parallel, 4, 'the agent reads the slot count from here');
  } finally { await f.native.close(); f.cleanup(); }
});

test("the person's own number wins: saved settings and Expert mode's value", async () => {
  for (const [config, values, expected] of [[{ parallel: 1 }, {}, 1], [{}, { parallel: 3 }, 3], [{ parallel: 2 }, { parallel: 6 }, 6]]) {
    const f = fixture({ config });
    try {
      await f.native.refresh(true);
      const job = await finish(f.native, f.native.operation('load', { model: 'iq3_s', values }).id);
      assert.equal(job.state, 'done', job.error);
      assert.equal(f.saved[0].parallel, expected);
      assert.ok(!f.saved[0].args.includes('--batch-mtp'));
    } finally { await f.native.close(); f.cleanup(); }
  }
});

test('a load that fails with several slots is tried again with one, and the person is told in plain words', async () => {
  const f = fixture({ failStarts: 1 });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq3_s' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.saved.length, 2);
    assert.equal(f.saved[0].parallel, 4); assert.equal(f.saved[1].parallel, undefined);
    assert.ok(!f.saved[1].args.includes('--batch-mtp'));
    assert.match(job.result.summary, /one request at a time/);
    assert.equal(job.result.summary.includes(FALLBACK_NOTE), true);
    assert.equal(f.native.state().instances[0].parallel, 1);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a load that fails even with one slot reports the failure, and a chosen number is never silently changed', async () => {
  const both = fixture({ failStarts: 2 });
  try {
    await both.native.refresh(true);
    const job = await finish(both.native, both.native.operation('load', { model: 'iq3_s' }).id);
    assert.equal(job.state, 'error');
    assert.equal(both.saved.length, 2);
  } finally { await both.native.close(); both.cleanup(); }
  const chosen = fixture({ failStarts: 1, config: { parallel: 3 } });
  try {
    await chosen.native.refresh(true);
    const job = await finish(chosen.native, chosen.native.operation('load', { model: 'iq3_s' }).id);
    assert.equal(job.state, 'error', 'the person asked for 3: no quiet fallback');
    assert.equal(chosen.saved.length, 1);
  } finally { await chosen.native.close(); chosen.cleanup(); }
});
