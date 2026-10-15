/**
 * The App Host's bridge to Strata Control, the Python part of the product.
 *
 * In the installed app Control ships as a component next to the host (<install>/control): a pinned CPython with
 * hash-locked wheels, strata_control, strata_cli and the API server. The host never uses the user's Python, never runs
 * pip, git or a compiler, and never downloads anything to start a model: it asks Control, and Control starts the engine
 * under its owner runtime (loopback TCP + 256-bit token on Windows, a private socket elsewhere; a Job Object that ends
 * the server with its host; leases that stop a second owner).
 *
 * What lives here: where Control and the engine component folder are, a runner for the CLI (JSON in, JSON out, a
 * timeout, no shell), and the four operations the app needs: which engine this PC would use, which models are
 * installed, start a saved profile, stop it.
 */
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));

export function bundledPython(dir, platform = process.platform) {
  return platform === 'win32' ? join(dir, 'python', 'python.exe') : join(dir, 'python', 'bin', 'python3');
}

/** The per-user component folder, the same rule as strata_control/engine_component.py components_dir(). */
export function componentsDirectory({ env = process.env, platform = process.platform, home = homedir() } = {}) {
  if (env.STRATA_COMPONENTS_DIR) return env.STRATA_COMPONENTS_DIR;
  if (platform === 'win32') return join(env.LOCALAPPDATA || join(home, 'AppData', 'Local'), 'Strata', 'components');
  return join(env.XDG_DATA_HOME || join(home, '.local', 'share'), 'strata', 'components');
}

/**
 * Is there a Control component to use? STRATA_CONTROL_DIR names it; otherwise it is the `control` folder beside
 * `host` in an installed app. Returns null in a source checkout (the host then keeps its older paths).
 */
export function findControl({ env = process.env, platform = process.platform, exists = existsSync, hostDir = resolve(HERE, '..') } = {}) {
  const candidates = [env.STRATA_CONTROL_DIR, join(hostDir, '..', 'control')].filter(Boolean).map(p => resolve(p));
  for (const dir of candidates) {
    const python = bundledPython(dir, platform);
    if (exists(python) && exists(join(dir, 'strata.py')) && exists(join(dir, 'strata_control'))) {
      let bundle = null;
      try { bundle = JSON.parse(readFileSync(join(dir, 'BUNDLE.json'), 'utf8').replace(/^﻿/, '')); } catch { /* optional */ }
      return { dir, python, cli: join(dir, 'strata.py'), bundle };
    }
  }
  return null;
}

/** The environment Control runs in: nothing from the user's Python, and acquiring forbidden. */
export function controlEnvironment(control, { env = process.env, components = componentsDirectory({ env }) } = {}) {
  const next = { ...env };
  for (const key of ['PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'VIRTUAL_ENV', 'STRATA_PYTHON', 'STRATA_NATIVE_PYTHON']) delete next[key];
  return {
    ...next,
    PYTHONNOUSERSITE: '1', PYTHONUTF8: '1', PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8',
    STRATA_NO_ACQUIRE: '1', STRATA_COMPONENTS_DIR: components, STRATA_API_KEY: '',
    ...(control?.dir ? { STRATA_CONTROL_DIR: control.dir } : {}),
  };
}

export class ControlError extends Error {
  constructor(message, { code = 'control_failed', exitCode = null, output = null } = {}) {
    super(message);
    this.name = 'ControlError';
    this.code = code;
    this.exitCode = exitCode;
    this.output = output;
  }
}

/** Run one CLI command of Control. Resolves { code, json, text, stderr }; rejects only if it cannot start or times out. */
export function runCli(control, args, { env = process.env, timeoutMs = 60_000, spawnProcess = spawn, components } = {}) {
  return new Promise((resolveRun, reject) => {
    const child = spawnProcess(control.python, [control.cli, ...args], {
      cwd: control.dir, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
      env: controlEnvironment(control, { env, ...(components ? { components } : {}) }),
    });
    let out = '', err = '', done = false;
    const finish = (fn, value) => { if (done) return; done = true; clearTimeout(alarm); fn(value); };
    const alarm = setTimeout(() => {
      try { child.kill(); } catch { /* already gone */ }
      finish(reject, new ControlError(`Strata Control did not answer within ${Math.round(timeoutMs / 1000)} s (${args.slice(0, 2).join(' ')}).`, { code: 'control_timeout' }));
    }, timeoutMs);
    child.stdout.on('data', c => { out += c; if (out.length > 4_000_000) child.kill(); });
    child.stderr.on('data', c => { err += c; if (err.length > 200_000) err = err.slice(-100_000); });
    child.on('error', e => finish(reject, new ControlError(`Strata Control could not start: ${e.message}`, { code: 'control_missing' })));
    child.on('close', code => {
      let json = null;
      const text = out.trim();
      if (text) { try { json = JSON.parse(text.slice(text.indexOf('{'))); } catch { /* not JSON: caller reads text */ } }
      finish(resolveRun, { code, json, text, stderr: err.trim() });
    });
  });
}

const failure = (result, fallback) => new ControlError(
  result.json?.error?.message ?? result.json?.error ?? result.json?.notice ?? (result.stderr.split(/\r?\n/).filter(Boolean).pop() || fallback),
  { exitCode: result.code, output: result.json });

/** Operations the app uses. `run` is injectable so the host's tests need no Python. */
export function createControl({ control = findControl(), env = process.env, run = null } = {}) {
  if (!control) return null;
  const call = run ?? ((args, options) => runCli(control, args, { env, ...options }));
  return {
    control,
    version: control.bundle?.version ?? null,
    async engine({ backend = null } = {}) {
      const result = await call(['components', 'engine', '--output', 'json', ...(backend ? ['--backend', backend] : [])], { timeoutMs: 30_000 });
      if (result.json) return result.json;                  // exit 5 still carries the report and the notice
      throw failure(result, 'The engine check failed.');
    },
    async installedModels({ installDir = null } = {}) {
      const result = await call(['models', 'list', '--installed', ...(installDir ? ['--install-dir', installDir] : []), '--output', 'json'], { timeoutMs: 60_000 });
      if (result.code !== 0 || !result.json) throw failure(result, 'The installed models could not be read.');
      return result.json;
    },
    async saveProfile(name, configPath) {
      const listed = await call(['profile', 'list', '--output', 'json'], { timeoutMs: 20_000 });
      const known = (listed.json?.profiles ?? []).some(p => p.id === name);
      const result = await call(known
        ? ['profile', 'use-config', name, '--from-config', configPath, '--output', 'json']
        : ['profile', 'create', name, '--from-config', configPath, '--output', 'json'], { timeoutMs: 30_000 });
      if (result.code !== 0) throw failure(result, 'The model profile could not be saved.');
      return result.json;
    },
    async start(name, { waitSeconds = 300 } = {}) {
      const result = await call(['runtime', 'start', '--profile', name, '--wait', String(waitSeconds), '--output', 'json'], { timeoutMs: (waitSeconds + 60) * 1000 });
      if (result.code !== 0 || !result.json?.ok) throw failure(result, 'The engine did not start.');
      return result.json;
    },
    async status(name) {
      const result = await call(['runtime', 'status', '--profile', name, '--output', 'json'], { timeoutMs: 30_000 });
      return result.json ?? { ok: false, state: 'unknown', error: result.stderr };
    },
    async stop(name, { waitSeconds = 120 } = {}) {
      const result = await call(['runtime', 'stop', '--profile', name, '--wait', String(waitSeconds), '--output', 'json'], { timeoutMs: (waitSeconds + 60) * 1000 });
      // an instance that is not running is already stopped
      if (result.code !== 0 && !/not running|no owner|stopped|absent|no managed/i.test(JSON.stringify(result.json ?? result.stderr))) throw failure(result, 'The engine did not stop.');
      return result.json;
    },
  };
}
