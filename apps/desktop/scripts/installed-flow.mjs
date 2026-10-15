#!/usr/bin/env node
/**
 * Prove the INSTALLED app layout can start and stop an engine with nothing acquired, using a stub engine.
 *
 *   node apps/desktop/scripts/installed-flow.mjs --install <folder the installer wrote> [--work <scratch>] [--spy <acquire-spy.mjs>]
 *
 * It starts the installed host with the installed node.exe exactly as the Tauri shell does (STRATA_PACKAGED=1,
 * STRATA_NO_ACQUIRE=1; Control found beside the host), against a scratch component folder holding a fake engine
 * package and a scratch install folder holding one run config. Everything per-user (profiles, runtime state, host data,
 * components) lives under --work, so the real profile of whoever runs this is never read or written.
 *
 * Steps (each one printed, the run exits 1 if any fails):
 *   host health, Control status (engine usable, acquiring forbidden), models from Control, load through the host
 *   (profile + `runtime start` under the owner runtime), a chat answer from the stub engine, a second client attaches to
 *   the same owner/instance, unload stops it, no engine for this PC is a notice with nothing started, the Tauri-style
 *   clean shutdown stops the engines the host started, a hard-killed host leaves the engine to Control (reported), stop.
 * With --spy the host runs under tests/helpers/acquire-spy.mjs and the run fails on any attempt to reach the internet or
 * run an installer.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, chmodSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]?.startsWith('--') || all[i + 1] === undefined ? true : all[i + 1]]] : acc), []));
if (!args.install) { console.error('usage: installed-flow.mjs --install <folder> [--work <dir>] [--spy <acquire-spy.mjs>]'); process.exit(2); }
const install = resolve(args.install);
const isWin = process.platform === 'win32';
const node = join(install, 'node', isWin ? 'node.exe' : 'node');
const hostEntry = join(install, 'host', 'server', 'index.mjs');
const controlDir = join(install, 'control');
const python = isWin ? join(controlDir, 'python', 'python.exe') : join(controlDir, 'python', 'bin', 'python3');
const work = resolve(args.work ?? mkdtempSync(join(tmpdir(), 'strata-installed-flow-')));
mkdirSync(work, { recursive: true });
const dirs = { components: join(work, 'components'), models: join(work, 'models'), profiles: join(work, 'profiles'), runtime: join(work, 'runtime'), home: join(work, 'home') };
// the runtime and profile folders are created by Control itself, with the owner-only ACL it insists on
for (const [name, d] of Object.entries(dirs)) if (!['runtime', 'profiles'].includes(name)) mkdirSync(d, { recursive: true });

const steps = [];
const record = (name, ok, detail = '') => { steps.push({ name, ok: !!ok, detail: String(detail) }); console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`); };
const delay = ms => new Promise(r => setTimeout(r, ms));
const freePort = async () => { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; };
const sha = buf => createHash('sha256').update(buf).digest('hex');

// ------------------------------------------------------------------------------ fixtures
const baseEnv = {
  ...process.env,
  STRATA_COMPONENTS_DIR: dirs.components, STRATA_PROFILE_DIR: dirs.profiles, STRATA_RUNTIME_DIR: dirs.runtime,
  STRATA_INSTALL_DIR: dirs.models, STRATA_ENGINE_BACKEND: 'cuda', STRATA_HOME: dirs.home, STRATA_MODELS_DIR: dirs.models,
  APPDATA: join(work, 'appdata'), XDG_CONFIG_HOME: join(work, 'xdg'), PYTHONDONTWRITEBYTECODE: '1',
};
delete baseEnv.STRATA_API_KEY;
mkdirSync(baseEnv.APPDATA, { recursive: true });
const mark = join(work, 'engine-ran.json');

const FAKE_ENGINE = `import os, sys, json
from pathlib import Path
Path(os.environ['FAKE_ENGINE_MARK']).write_text(json.dumps({'pid': os.getpid(), 'script': os.path.abspath(__file__), 'argv': sys.argv[1:]}))
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.strip() == 'QUIT': break
    if line.startswith(('GEN ', 'BGEN ')):
        slot = int(line.split()[1]) if line.startswith('BGEN ') else None
        print('T 97', flush=True)
        print('DONE 1 1 1.0 1.0 length 0 0 0', flush=True)
        if slot is not None: print(f'BADM {slot} 0', flush=True)
`;

function installFakeEngine() {
  const platform = isWin ? 'windows-x86_64' : 'linux-x86_64';
  const folder = join(dirs.components, 'engine', `0.1.45-${platform}-cuda`);
  mkdirSync(folder, { recursive: true });
  writeFileSync(join(folder, 'engine_main.py'), FAKE_ENGINE);
  const entry = isWin ? 'engine.cmd' : 'engine';
  writeFileSync(join(folder, entry), isWin
    ? `@echo off\r\n"${python}" "%~dp0engine_main.py" %*\r\n`
    : `#!/bin/sh\nexec "${python}" "$(dirname "$0")/engine_main.py" "$@"\n`);
  if (!isWin) chmodSync(join(folder, entry), 0o755);
  const files = [entry, 'engine_main.py'].map(p => { const b = readFileSync(join(folder, p)); return { path: p, sha256: sha(b), bytes: b.length }; });
  writeFileSync(join(folder, 'engine-component.json'), JSON.stringify({ schema_version: 1, id: 'strata-engine', version: '0.1.45', platform, backend: 'cuda', entrypoint: entry, files }));
  return folder;
}

function makeTokenizerAndConfig() {
  const tok = join(dirs.models, 'tokenizer');
  mkdirSync(tok, { recursive: true });
  const script = `import json, sys\nfrom tools.strata_tokenizer import BYTE_TO_UNICODE\ntok = sys.argv[1]\ntokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ['ab', '<|im_start|>', '<|im_end|>', '<|endoftext|>']\njson.dump({t: i for i, t in enumerate(tokens)}, open(tok + '/vocab.json', 'w'))\nopen(tok + '/merges.txt', 'w').write('a b')\njson.dump([0] * 257 + [3, 3, 3], open(tok + '/token_type.json', 'w'))\n`;
  const done = spawnSync(python, ['-c', script, tok], { cwd: controlDir, env: { ...baseEnv, PYTHONNOUSERSITE: '1' }, encoding: 'utf8' });
  if (done.status !== 0) throw new Error('tokenizer fixture failed: ' + done.stderr);
  writeFileSync(join(dirs.models, 'strata-stub.json'), JSON.stringify({ exe: 'component:strata-engine', backend: 'cuda', tokenizer: tok, args: [], cwd: work, model_name: 'stub', host: '127.0.0.1', port: 0 }));
}

const control = (...argv) => spawnSync(python, [join(controlDir, 'strata.py'), ...argv, '--output', 'json'], { cwd: controlDir, env: { ...baseEnv, PYTHONNOUSERSITE: '1', STRATA_NO_ACQUIRE: '1', FAKE_ENGINE_MARK: mark }, encoding: 'utf8', timeout: 120000 });
const controlJson = (...argv) => { const r = control(...argv); let json = null; try { json = JSON.parse(r.stdout.slice(r.stdout.indexOf('{'))); } catch { /* not json */ } return { code: r.status, json, text: r.stdout + r.stderr }; };
const alive = pid => { try { process.kill(pid, 0); return true; } catch { return false; } };

// ------------------------------------------------------------------------------ the host
let host = null, port = 0, base = '', spyLog = null, shutdownToken = randomBytes(32).toString('hex');
async function startHost() {
  port = await freePort(); base = `http://127.0.0.1:${port}`;
  const hostArgs = [...(args.spy ? ['--import', pathToFileURL(resolve(args.spy)).href] : []), hostEntry];
  if (args.spy) { spyLog = join(work, 'acquire-spy.log'); writeFileSync(spyLog, ''); }
  host = spawn(node, hostArgs, {
    cwd: join(install, 'host'), stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
    env: { ...baseEnv, STRATA_PORT: String(port), STRATA_DIST: join(install, 'host', 'dist'), NODE_ENV: 'production', STRATA_PACKAGED: '1', STRATA_REAL_ENGINE: '1',
      STRATA_NO_ACQUIRE: '1', STRATA_SHUTDOWN_TOKEN: shutdownToken, FAKE_ENGINE_MARK: mark, STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()),
      ...(spyLog ? { ACQUIRE_SPY_LOG: spyLog } : {}) },
  });
  let log = '';
  host.stdout.on('data', c => { log += c; }); host.stderr.on('data', c => { log += c; });
  host.hostLog = () => log;
  for (let i = 0; i < 200; i++) { try { if ((await fetch(base + '/api/health')).ok) return true; } catch { /* not yet */ } await delay(150); }
  return false;
}
const call = async (path, method = 'GET', body) => {
  const response = await fetch(base + path, { method, headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: response.status, body: await response.json().catch(() => null) };
};
async function waitJob(id, seconds = 120) {
  for (let i = 0; i < seconds * 5; i++) {
    const state = (await call('/api/native')).body;
    const job = state.jobs.find(j => j.id === id);
    if (job && job.state !== 'running') return { job, state };
    await delay(200);
  }
  throw new Error('operation did not finish');
}
async function stopHost(hard = false) {
  if (!host || host.exitCode !== null) return;
  if (hard) { if (isWin) spawnSync('taskkill', ['/PID', String(host.pid), '/T', '/F'], { windowsHide: true }); else host.kill('SIGKILL'); }
  else host.kill();
  for (let i = 0; i < 100 && host.exitCode === null; i++) await delay(100);
}

async function main() {
  for (const need of [node, hostEntry, python, join(controlDir, 'strata.py')]) if (!existsSync(need)) { record('installed layout', false, `missing ${need}`); return; }
  record('installed layout', true, install);
  installFakeEngine();
  makeTokenizerAndConfig();
  record('fixtures', true, `stub engine package in ${dirs.components}; run config strata-stub.json`);

  if (!(await startHost())) { record('host healthy', false, host.hostLog().slice(-600)); return; }
  record('host healthy', true, base);

  const status = (await call('/api/control/status')).body;
  record('Control status: bundled, engine usable, acquiring forbidden', status.control?.python?.startsWith(controlDir) && status.engine?.usable === true && status.acquisition?.forbidden === true,
    `python=${status.control?.python}; engine=${status.engine?.engine?.version}; forbidden=${status.acquisition?.forbidden}`);

  const native = (await call('/api/native?refresh=1')).body;
  const model = native.installed?.models?.find(m => m.model === 'stub');
  record('models come from Control', native.available && model?.ready === true && native.packaged === true, `available=${native.available}; model ready=${model?.ready}; ${native.summary ?? native.error}`);

  // ---- load
  const started = (await call('/api/native/operations', 'POST', { kind: 'load', args: { model: 'stub' } })).body;
  const loaded = await waitJob(started.id);
  const instance = loaded.state.instances[0];
  record('load through the host starts the engine under Control', loaded.job.state === 'done' && instance?.status === 'ready', `job=${loaded.job.state} ${loaded.job.error ?? ''}; instance=${instance?.status}`);
  const ran = existsSync(mark) ? JSON.parse(readFileSync(mark, 'utf8')) : null;
  record('the engine that ran is the one in the component folder', ran && ran.script.toLowerCase().startsWith(dirs.components.toLowerCase()), ran?.script ?? 'engine never ran');

  const runtime = controlJson('runtime', 'status', '--profile', 'app-1');
  record('a second client sees the same owner and instance (ownership)', runtime.json?.state === 'ready' && !!runtime.json.owner_id && !!runtime.json.instance_id && runtime.json.host_pid !== host.pid,
    `state=${runtime.json?.state}; owner=${String(runtime.json?.owner_id).slice(0, 8)}; instance=${String(runtime.json?.instance_id).slice(0, 8)}; owner host pid ${runtime.json?.host_pid} is not the app host (${host.pid})`);

  // ---- answer
  const chat = await call('/v1/chat/completions', 'POST', { model: instance?.apiModelId ?? 'stub', messages: [{ role: 'user', content: 'hi' }], max_tokens: 4, stream: false });
  const message = chat.body?.choices?.[0]?.message ?? {};
  const answer = `${message.content ?? ''}${message.reasoning_content ?? ''}`;   // the stub emits token 97 ("a"); the server files it as reasoning
  record('the engine answers through the host (the stub token, not an error)', chat.status === 200 && answer === 'a', `status=${chat.status}; answer=${JSON.stringify(answer)}${chat.status === 200 ? '' : ' ' + JSON.stringify(chat.body).slice(0, 160)}`);

  // ---- unload
  const unloaded = await call(`/api/native/instances/${instance.id}/unload`, 'POST', {});
  await delay(500);
  const after = controlJson('runtime', 'status', '--profile', 'app-1');
  record('unload stops the engine', unloaded.status === 200 && after.json?.state === 'stopped' && (!ran || !alive(ran.pid)), `status=${unloaded.status}; runtime=${after.json?.state}; engine pid ${ran?.pid} alive=${ran ? alive(ran.pid) : 'n/a'}`);

  // ---- no engine for this PC
  rmSync(join(dirs.components, 'engine'), { recursive: true, force: true });
  rmSync(mark, { force: true });
  const missing = await waitJob((await call('/api/native/operations', 'POST', { kind: 'load', args: { model: 'stub' } })).body.id);
  record('no engine for this PC is a notice and nothing is started', missing.job.state === 'error' && /No engine for this PC/.test(missing.job.error) && /never downloads/.test(missing.job.error) && !existsSync(mark),
    `${missing.job.state}: ${String(missing.job.error).slice(0, 160)}`);
  const refused = await waitJob((await call('/api/native/operations', 'POST', { kind: 'install', args: { family: 'qwen', model: 'IQ2_XS' } })).body.id);
  record('installing from the app is refused (Terminal only)', refused.job.state === 'error' && /Terminal/.test(refused.job.error), refused.job.error);
  const binary = await call('/api/engine/binary', 'POST');
  record('the llama.cpp download route is refused', binary.status === 409 && /never downloads/.test(binary.body?.error ?? ''), `${binary.status} ${binary.body?.error}`);

  // ---- clean shutdown (what the Tauri shell does on exit)
  installFakeEngine();
  rmSync(mark, { force: true });
  const again = await waitJob((await call('/api/native/operations', 'POST', { kind: 'load', args: { model: 'stub' } })).body.id);
  const ran2 = existsSync(mark) ? JSON.parse(readFileSync(mark, 'utf8')) : null;
  record('load again after the engine package returns', again.job.state === 'done', again.job.error ?? '');
  const wrong = await fetch(base + '/api/control/shutdown', { method: 'POST', headers: { 'x-strata-shutdown': 'nope' } });
  record('shutdown without the shell\'s secret is refused', wrong.status === 403, String(wrong.status));
  const exited = new Promise(r => host.once('exit', code => r(code)));
  const stopping = await fetch(base + '/api/control/shutdown', { method: 'POST', headers: { 'x-strata-shutdown': shutdownToken } });
  const code = await Promise.race([exited, delay(90000).then(() => 'timeout')]);
  const final = controlJson('runtime', 'status', '--profile', 'app-1');
  record('clean shutdown stops the engine the app started and exits', stopping.status === 202 && code === 0 && final.json?.state === 'stopped' && (!ran2 || !alive(ran2.pid)), `status=${stopping.status}; host exit=${code}; runtime=${final.json?.state}; engine alive=${ran2 ? alive(ran2.pid) : 'n/a'}`);

  // ---- hard kill: report what happens to an engine whose app host dies
  if (!(await startHost())) { record('host restarts', false, host.hostLog().slice(-400)); return; }
  await call('/api/native?refresh=1');
  rmSync(mark, { force: true });
  const third = await waitJob((await call('/api/native/operations', 'POST', { kind: 'load', args: { model: 'stub' } })).body.id);
  const ran3 = existsSync(mark) ? JSON.parse(readFileSync(mark, 'utf8')) : null;
  await stopHost(true);
  await delay(1500);
  const orphan = controlJson('runtime', 'status', '--profile', 'app-1');
  record('(info) a hard-killed app host leaves the engine to Control', third.job.state === 'done', `engine after the host was killed: runtime=${orphan.json?.state}, engine alive=${ran3 ? alive(ran3.pid) : 'n/a'}`);
  const cleanup = controlJson('runtime', 'stop', '--profile', 'app-1');
  record('Control stops it on request', cleanup.json?.state === 'stopped' && (!ran3 || !alive(ran3.pid)), `runtime=${cleanup.json?.state}; engine alive=${ran3 ? alive(ran3.pid) : 'n/a'}`);

  if (spyLog) {
    const attempts = readFileSync(spyLog, 'utf8').split('\n').filter(Boolean);
    record('no acquisition attempt by the host during the whole run (spy)', attempts.length === 0, attempts.join(' | '));
  }
}

try { await main(); }
catch (error) { record('flow', false, error.stack ?? error.message); }
finally {
  await stopHost(true);
  controlJson('runtime', 'stop', '--profile', 'app-1');
  controlJson('runtime', 'stop', '--profile', 'app-2');
}
const failed = steps.filter(s => !s.ok);
writeFileSync(join(work, 'installed-flow.json'), JSON.stringify({ ok: failed.length === 0, install, work, steps }, null, 1));
console.log(`\n${steps.length - failed.length}/${steps.length} ok; report ${join(work, 'installed-flow.json')}`);
if (!args.keep && !args.work && failed.length === 0) { try { rmSync(work, { recursive: true, force: true }); } catch { /* leave it */ } }
process.exit(failed.length ? 1 : 0);
