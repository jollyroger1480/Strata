import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { bundledPython, componentsDirectory, controlEnvironment, createControl, findControl, runCli } from '../server/control-runtime.mjs';
import { resolvePython } from '../server/python-resolver.mjs';
import { createNativeControl } from '../server/native-control.mjs';

const isWin = process.platform === 'win32';

function fakeControlDir() {
  const dir = mkdtempSync(join(tmpdir(), 'strata-control-dir-'));
  mkdirSync(join(dir, 'python', ...(isWin ? [] : ['bin'])), { recursive: true });
  mkdirSync(join(dir, 'strata_control'), { recursive: true });
  mkdirSync(join(dir, 'serve'), { recursive: true });
  writeFileSync(bundledPython(dir), 'x');
  writeFileSync(join(dir, 'strata.py'), '# cli');
  writeFileSync(join(dir, 'serve', 'server.py'), 'port\nparallel\n/v1/responses');
  writeFileSync(join(dir, 'BUNDLE.json'), JSON.stringify({ version: '0.1.45', python: '3.12.15' }));
  return dir;
}

test('the Control component is found by STRATA_CONTROL_DIR or beside the host, and only if it is whole', () => {
  const dir = fakeControlDir();
  try {
    const found = findControl({ env: { STRATA_CONTROL_DIR: dir }, hostDir: join(dir, 'nowhere') });
    assert.equal(found.dir, dir);
    assert.equal(found.python, bundledPython(dir));
    assert.equal(found.bundle.python, '3.12.15');
    // beside the host: <install>/host + <install>/control
    const install = mkdtempSync(join(tmpdir(), 'strata-install-'));
    try {
      mkdirSync(join(install, 'host', 'server'), { recursive: true });
      assert.equal(findControl({ env: {}, hostDir: join(install, 'host') }), null, 'nothing beside the host yet');
      mkdirSync(join(install, 'control', 'python', ...(isWin ? [] : ['bin'])), { recursive: true });
      mkdirSync(join(install, 'control', 'strata_control'));
      writeFileSync(bundledPython(join(install, 'control')), 'x');
      writeFileSync(join(install, 'control', 'strata.py'), '#');
      assert.equal(findControl({ env: {}, hostDir: join(install, 'host') }).dir, join(install, 'control'));
    } finally { rmSync(install, { recursive: true, force: true }); }
    rmSync(join(dir, 'strata.py'));
    assert.equal(findControl({ env: { STRATA_CONTROL_DIR: dir }, hostDir: join(dir, 'x') }), null, 'a half component is not used');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("Control runs on its own interpreter with nothing from the user's Python and acquiring forbidden", () => {
  const control = { dir: '/c', python: '/c/python/python3', cli: '/c/strata.py' };
  const env = controlEnvironment(control, { env: { PYTHONHOME: 'x', PYTHONPATH: 'y', VIRTUAL_ENV: 'z', STRATA_PYTHON: 'q', PATH: 'p', STRATA_API_KEY: 'secret' }, components: '/components' });
  assert.equal(env.PYTHONHOME, undefined);
  assert.equal(env.PYTHONPATH, undefined);
  assert.equal(env.VIRTUAL_ENV, undefined);
  assert.equal(env.STRATA_PYTHON, undefined);
  assert.equal(env.PYTHONNOUSERSITE, '1');
  assert.equal(env.STRATA_NO_ACQUIRE, '1');
  assert.equal(env.STRATA_COMPONENTS_DIR, '/components');
  assert.equal(env.STRATA_API_KEY, undefined, 'an inherited inference key never reaches the engine (the run config carries its own)');
  assert.equal(env.PATH, 'p');
  // the resolver prefers the bundled interpreter over py -3 / PATH / a venv
  const python = resolvePython({ repo: '/repo', control, env: {}, exists: () => true, starts: () => { throw new Error('must not probe'); }, useCache: false });
  assert.deepEqual([python.command, python.source], ['/c/python/python3', 'bundled']);
  assert.equal(resolvePython({ control, env: { STRATA_PYTHON: '/mine' }, useCache: false }).command, '/mine', 'an explicit choice still wins');
  assert.equal(componentsDirectory({ env: { STRATA_COMPONENTS_DIR: '/x' } }), '/x');
  assert.match(componentsDirectory({ env: { LOCALAPPDATA: 'C:\\L' }, platform: 'win32', home: 'C:\\H' }), /Strata[\\/]components$/);
});

test('runCli passes the arguments untouched (no shell) and reads JSON', async () => {
  const seen = [];
  const spawnProcess = (command, args, options) => {
    seen.push({ command, args, options });
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.kill = () => {};
    queueMicrotask(() => { child.stdout.emit('data', '{"ok": true, "n": 1}\n'); child.emit('close', 0); });
    return child;
  };
  const control = { dir: '/c', python: '/c/python/python3', cli: '/c/strata.py' };
  const result = await runCli(control, ['models', 'list', '--installed', '--output', 'json'], { spawnProcess, env: {} });
  assert.deepEqual(result.json, { ok: true, n: 1 });
  assert.deepEqual(seen[0].args, ['/c/strata.py', 'models', 'list', '--installed', '--output', 'json']);
  assert.equal(seen[0].options.shell, undefined);
  assert.equal(seen[0].options.windowsHide, true);
  assert.equal(seen[0].options.env.STRATA_NO_ACQUIRE, '1');
});

// ---------------------------------------------------------------------------------- native-control in packaged mode

function packagedFixture({ engineUsable = true } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-packaged-test-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  const configs = ['iq2_xs', 'iq3_s'].map(id => {
    const path = join(install, `strata-${id}.json`);
    writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--max-context', '32768'], model_name: `Qwen ${id}`, port: 8080 }));
    return { id, path };
  });
  const calls = [];
  const spawned = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.1.45',
    async installedModels(options) { calls.push(['installedModels', options?.installDir ?? null]); return { install_dir: install, models: configs.map(c => ({ id: c.id, name: `Qwen ${c.id}`, source_config: c.path, component_status: 'present', context_tokens: 32768, backend: 'cuda' })) }; },
    async engine() {
      calls.push(['engine']);
      return engineUsable
        ? { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.1.45', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\engine\\strata.exe', root: 'E:\\engine' } }
        : { usable: false, notice: 'No engine for this PC was found in E:\\c: the component folder holds no engine package. The app never downloads an engine while it starts or serves.', backend: 'cuda', engine: null, components_dir: 'E:\\c' };
    },
    async saveProfile(name, path) { calls.push(['saveProfile', name, basename(path)]); return {}; },
    async start(name, options) { calls.push(['start', name, options?.waitSeconds]); return { ok: true, state: 'ready' }; },
    async stop(name) { calls.push(['stop', name]); return { ok: true, state: 'stopped' }; },
  };
  const requests = [];
  const fetcher = async url => { calls.push(['fetch', url.replace(/:\d+/, ':PORT')]); return Response.json({ status: 'ok' }); };
  const native = createNativeControl({ control, directory: data, spawnProcess: (...args) => { spawned.push(args); throw new Error('no process may be started by the host in packaged mode'); }, fetcher, requests: () => requests });
  return { native, calls, spawned, install, root, cleanup: () => { rmSync(root, { recursive: true, force: true }); } };
}
const finish = async (native, id) => { for (let i = 0; i < 400; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('packaged: models come from Control, no python, git or engine process is started by the host', async () => {
  const f = packagedFixture();
  try {
    const state = await f.native.refresh(true);
    assert.equal(state.available, true, state.error);
    assert.deepEqual(state.installed.models.map(m => m.model), ['iq2_xs', 'iq3_s']);
    assert.equal(state.installed.models[0].ready, true);
    assert.equal(state.packaged, true);
    assert.match(state.summary, /engine 0\.1\.45/);
    assert.equal(state.version, '0.1.45');
    assert.equal(f.spawned.length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: load = saved profile + runtime start under Control; unload = runtime stop; two models get two profiles', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const load = async args => { const job = await finish(f.native, f.native.operation('load', args).id); assert.equal(job.state, 'done', job.error); return job; };
    await load({ model: 'iq2_xs' });
    const first = f.native.state().instances[0];
    assert.equal(first.status, 'ready');
    const startCalls = f.calls.filter(c => ['saveProfile', 'start'].includes(c[0]));
    assert.deepEqual(startCalls.map(c => c.slice(0, 2)), [['saveProfile', 'app-1'], ['start', 'app-1']]);
    assert.equal(startCalls[1][2], 300);
    await load({ model: 'iq3_s', multiple: true });
    assert.deepEqual(f.calls.filter(c => c[0] === 'start').map(c => c[1]), ['app-1', 'app-2']);
    assert.equal(f.native.state().instances.length, 2);
    await f.native.unload(first.id);
    assert.deepEqual(f.calls.filter(c => c[0] === 'stop').at(-1), ['stop', 'app-1']);
    assert.equal(f.native.state().instances.length, 1);
    assert.equal(f.spawned.length, 0, 'the host never spawned anything itself');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: no engine for this PC is the notice, and nothing is started', async () => {
  const f = packagedFixture({ engineUsable: false });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /No engine for this PC/);
    assert.match(job.error, /never downloads an engine/);
    assert.equal(f.calls.some(c => ['saveProfile', 'start'].includes(c[0])), false);
    assert.equal(f.spawned.length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: install, update, rollback and calibration are refused with the Terminal notice and start nothing', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    for (const [kind, args] of [['install', { family: 'qwen', model: 'IQ2_XS' }], ['update-engine', {}], ['update-strata', {}], ['rollback', {}], ['calibrate', { model: 'iq3_s' }]]) {
      const job = await finish(f.native, f.native.operation(kind, args).id);
      assert.equal(job.state, 'error', kind);
      assert.match(job.error, /Terminal|calibrates NVIDIA|No saved previous engine/, `${kind}: ${job.error}`);
    }
    assert.equal(f.spawned.length, 0);
    assert.equal(f.calls.filter(c => ['saveProfile', 'start'].includes(c[0])).length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the install folder can be chosen, and it is what Control is asked about', async () => {
  const f = packagedFixture();
  try {
    const other = join(f.root, 'other');
    mkdirSync(other);
    await f.native.configure({ repository: other });
    const saved = JSON.parse(readFileSync(join(f.root, 'data', 'native-host.json'), 'utf8'));
    assert.equal(saved.installDir.toLowerCase(), other.toLowerCase());
    assert.equal(f.calls.filter(c => c[0] === 'installedModels').at(-1)[1].toLowerCase(), other.toLowerCase());
  } finally { await f.native.close(); f.cleanup(); }
});

test('createControl asks the CLI for the engine, models and runtime and reads the notice of a missing engine', async () => {
  const seen = [];
  const control = createControl({
    control: { dir: '/c', python: '/p', cli: '/c/strata.py', bundle: { version: '9' } },
    run: async args => { seen.push(args.join(' ')); return args[0] === 'components' ? { code: 5, json: { ok: false, usable: false, notice: 'No engine for this PC' }, text: '', stderr: '' } : { code: 0, json: { ok: true, profiles: [] }, text: '', stderr: '' }; },
  });
  assert.equal((await control.engine()).notice, 'No engine for this PC');
  await control.installedModels({ installDir: 'D:\\Strata' });
  await control.saveProfile('app-1', 'C:\\x.json');
  await control.start('app-1');
  await control.stop('app-1');
  assert.deepEqual(seen, [
    'components engine --output json',
    'models list --installed --install-dir D:\\Strata --output json',
    'profile list --output json',
    'profile create app-1 --from-config C:\\x.json --output json',
    'runtime start --profile app-1 --wait 300 --output json',
    'runtime stop --profile app-1 --wait 120 --output json',
  ]);
  assert.equal(createControl({ control: null }), null);
});
