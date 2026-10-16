/**
 * Time-skip, from the App Host.
 *
 * Strata Control owns the job: checkpointing, isolated candidates, the checks and the decision all
 * live in `strata_control/time_skip.py` and are driven through the `strata timeskip` CLI. This
 * module is only the bridge — it starts that command, streams its output, and reads back the durable
 * `run.json` the job writes.
 *
 * Two rules it must not break:
 *
 *   1. **It never decides anything.** Whether a candidate was kept is the job's answer, read from
 *      disk. This module does not compute a status, a score or a progress fraction.
 *   2. **It never reports progress it does not have.** The job's phases are coarse and a check can
 *      take minutes with nothing to say, so the stream carries what the command actually printed and
 *      the UI shows a phase, not a percentage.
 *
 * A run is a job, not a request: it outlives the HTTP call that started it. The endpoint returns the
 * run id immediately and the client follows it with `/api/timeskip/status`.
 */
import { spawn } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { timeSkipDir } from './platform-paths.mjs';

/** Where runs are recorded. Mirrors the CLI's own default resolution. */
function stateDir(env = process.env) {
  return timeSkipDir(env);
}

function readRun(root, id) {
  const file = join(root, id, 'run.json');
  if (!existsSync(file)) return null;
  try {
    return JSON.parse(readFileSync(file, 'utf8'));
  } catch {
    // A partially written record is not a run: the job writes atomically, so a parse failure means
    // something else touched the file. Report it as missing rather than as an empty run.
    return null;
  }
}

function listRuns(root) {
  if (!existsSync(root)) return [];
  const out = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const run = readRun(root, entry.name);
    if (run) out.push(summarise(run));
  }
  return out.sort((a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0));
}

/**
 * The projection the UI consumes.
 *
 * Everything here is copied from the job's own record. `decision` and `needsUserReview` are the
 * job's answers; `phase` is its current state. There is deliberately no derived `percent`.
 */
function summarise(run) {
  const attempts = run.attempts ?? [];
  const checkpoints = run.checkpoints ?? [];
  return {
    id: run.id,
    phase: run.state,
    workspace: run.workspace,
    prompt: run.prompt,
    startedAt: run.started_at ?? null,
    finishedAt: run.finished_at ?? null,
    attempts: attempts.map((a) => ({
      index: a.index,
      decision: a.decision ?? null,
      reason: a.reason ?? '',
      checks: (a.checks ?? []).map((c) => ({ id: c.id, outcome: c.outcome, kind: c.kind })),
    })),
    checkpoints: checkpoints.map((c) => ({
      id: c.id,
      label: c.label,
      digest: c.digest,
      checks: (c.checks ?? []).map((x) => ({ id: x.id, outcome: x.outcome, kind: x.kind })),
    })),
    selectedCheckpoint: run.selected_checkpoint ?? null,
    decision: run.decision ?? null,
    // `unknown` checks must survive to the UI as unknown; a caller that renders this as a failure
    // or as a pass is the bug this field exists to prevent.
    needsUserReview: Boolean(run.decision?.needs_user_review),
    error: run.error ?? null,
  };
}

const TERMINAL = new Set(['completed', 'cancelled', 'failed', 'interrupted']);
const SKIP_DIRS = new Set(['.git', '.strata', '__pycache__', 'node_modules']);
const MAX_TEXT_BYTES = 200_000;
const MAX_LISTED = 60;
const MAX_WITH_TEXT = 12;

/** Every file under a folder, as relative forward-slash paths. */
function walkFiles(root) {
  const out = [];
  if (!root || !existsSync(root)) return out;
  const walk = dir => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (out.length > 5000) return;
      const full = join(dir, entry.name);
      if (entry.isDirectory()) { if (!SKIP_DIRS.has(entry.name)) walk(full); }
      else if (entry.isFile()) out.push(relative(root, full).split('\\').join('/'));
    }
  };
  walk(root);
  return out.sort();
}

function readText(path) {
  try {
    if (statSync(path).size > MAX_TEXT_BYTES) return { big: true };
    const buffer = readFileSync(path);
    if (buffer.subarray(0, 4096).includes(0)) return { binary: true };
    return { text: buffer.toString('utf8') };
  } catch { return { missing: true }; }
}

/** What the assistant is told for one revision attempt. It works on a copy, so it is told to stay in it and to say what it did. */
export function timeskipRevisionMessage(request) {
  return [
    'You are improving a copy of the user\'s work. The current folder is that copy: the original is safe and you cannot see it.',
    'Rules:',
    '1. Work only inside the current folder. Never read or write outside it.',
    '2. Make the change the user asked for by creating or editing files here. If the folder is empty, create the file(s) the request needs.',
    '3. Do not delete the user\'s files unless the request says to.',
    '4. Do not run long programs or start servers.',
    '5. When you are done, answer in one or two plain sentences: what you changed.',
    '',
    'The request:',
    String(request).trim(),
  ].join('\n');
}

/** How big a folder is, stopping early once it is clearly too big to copy for a trial. */
export function measureFolder(root, { maxFiles = 3000, maxBytes = 300 * 1024 * 1024 } = {}) {
  let files = 0, bytes = 0, tooBig = false;
  const walk = dir => {
    if (tooBig) return;
    let entries;
    try { entries = readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const entry of entries) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) { if (!SKIP_DIRS.has(entry.name)) walk(full); }
      else if (entry.isFile()) {
        files += 1;
        try { bytes += statSync(full).size; } catch { /* ignore */ }
        if (files > maxFiles || bytes > maxBytes) { tooBig = true; return; }
      }
      if (tooBig) return;
    }
  };
  walk(root);
  return { files, bytes, tooBig };
}

/**
 * What a run changed, file by file: the saved original against the revision, for the Compare view.
 * Reads the folders the job itself kept; it decides nothing.
 */
/** Pictures the Compare view shows side by side (read as bytes through /api/timeskip/file, never as text). */
export const IMAGE_TYPES = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', gif: 'image/gif', webp: 'image/webp', svg: 'image/svg+xml', bmp: 'image/bmp' };
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
export const imageTypeOf = path => IMAGE_TYPES[String(path).split('.').pop().toLowerCase()] ?? null;

/** The two folders a run compares: the saved original and the revision (or the last attempt when the checks kept nothing). */
export function runRoots(record) {
  const baseline = (record.checkpoints ?? []).find(c => c.label === 'baseline');
  const selected = (record.checkpoints ?? []).find(c => c.id === record.selected_checkpoint);
  const attempts = record.attempts ?? [];
  const last = attempts[attempts.length - 1] ?? null;
  const useAttempt = !selected || selected.id === baseline?.id;
  return { baseRoot: baseline?.path ?? null, updatedRoot: (useAttempt ? last?.path : selected?.path) ?? null, useAttempt, last };
}

/** The absolute path of one picture of a run, for one side, or null: not a picture, not inside the run's folder, missing or too big. */
export function runImagePath(record, relPath, version) {
  if (!imageTypeOf(relPath)) return null;
  const { baseRoot, updatedRoot } = runRoots(record);
  const root = version === 'original' ? baseRoot : version === 'updated' ? updatedRoot : null;
  if (!root) return null;
  const full = resolve(root, ...String(relPath).split('/'));
  const inside = relative(resolve(root), full);
  if (!inside || inside.startsWith('..') || isAbsolute(inside)) return null;
  try { const st = statSync(full); return st.isFile() && st.size <= MAX_IMAGE_BYTES ? full : null; } catch { return null; }
}

export function diffRun(record) {
  // A kept checkpoint is the revision; when nothing was kept (the checks could not decide) the person looks at the last attempt.
  const { baseRoot, updatedRoot, useAttempt, last } = runRoots(record);
  if (!baseRoot || !updatedRoot) return { files: [], applyAttempt: null, more: 0 };
  const paths = [...new Set([...walkFiles(baseRoot), ...walkFiles(updatedRoot)])].sort();
  const files = [];
  let withText = 0, more = 0;
  for (const path of paths) {
    const a = readText(join(baseRoot, path)), b = readText(join(updatedRoot, path));
    const status = a.missing ? 'added' : b.missing ? 'removed' : null;
    const same = !status && a.text !== undefined && a.text === b.text;
    if (same) continue;
    if (imageTypeOf(path)) {
      // A picture is compared as bytes and shown from the file itself, never read as text.
      let differs = true;
      try { differs = a.missing || b.missing || !readFileSync(join(baseRoot, path)).equals(readFileSync(join(updatedRoot, path))); } catch { /* treat as changed */ }
      if (!differs) continue;
      if (files.length >= MAX_LISTED) { more += 1; continue; }
      files.push({ path, status: status ?? 'changed', image: true });
      continue;
    }
    if (!status && !a.text && !b.text && !a.big && !b.big) {
      // Both binary: compare sizes only.
      try { if (statSync(join(baseRoot, path)).size === statSync(join(updatedRoot, path)).size && readFileSync(join(baseRoot, path)).equals(readFileSync(join(updatedRoot, path)))) continue; } catch { /* treat as changed */ }
    }
    if (files.length >= MAX_LISTED) { more += 1; continue; }
    const entry = { path, status: status ?? 'changed' };
    const textual = (a.text !== undefined || a.missing) && (b.text !== undefined || b.missing);
    if (textual && withText < MAX_WITH_TEXT) { entry.original = a.text ?? ''; entry.updated = b.text ?? ''; withText += 1; }
    else entry.unreadable = true;
    files.push(entry);
  }
  return { files, applyAttempt: useAttempt ? (last?.index ?? null) : null, more };
}

/**
 * Run one `strata timeskip …` command and collect its result.
 *
 * The command is given a bounded time. A Time-skip run can legitimately take many minutes (each
 * attempt is an agent turn), so the default bound is generous, and exceeding it is reported as a
 * timeout rather than as success.
 */
export function createTimeSkip({ repoRoot, python, pythonArgs = [], timeoutMs = 30 * 60 * 1000, env = process.env } = {}) {
  const cli = join(repoRoot, 'strata.py');

  /** Runs this host started and is still driving. A run that is not terminal and not in here died with its host. */
  const live = new Map();
  const withLive = summary => (summary && !TERMINAL.has(summary.phase) && !live.has(summary.id) ? { ...summary, phase: 'interrupted' } : summary);

  function runCli(args, { signal } = {}) {
    return new Promise((resolve) => {
      const root = stateDir(env);
      const child = spawn(python, [...pythonArgs, cli, ...args, '--output', 'json'], {
        cwd: repoRoot,
        windowsHide: true,
        env: { ...env, STRATA_TIMESKIP_DIR: root },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      let timedOut = false;
      const timer = setTimeout(() => {
        timedOut = true;
        try {
          child.kill('SIGTERM');
        } catch {
          /* already gone */
        }
      }, timeoutMs);
      const onAbort = () => {
        try {
          child.kill('SIGTERM');
        } catch {
          /* already gone */
        }
      };
      signal?.addEventListener('abort', onAbort, { once: true });

      child.stdout.on('data', (b) => {
        stdout += b.toString();
      });
      child.stderr.on('data', (b) => {
        stderr += b.toString();
      });
      child.on('error', (e) => {
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
        resolve({ ok: false, code: null, error: `could not start the Time-skip command: ${e.message}` });
      });
      child.on('close', (code) => {
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
        if (timedOut) {
          resolve({ ok: false, code, error: `the Time-skip command did not finish within ${Math.round(timeoutMs / 1000)}s` });
          return;
        }
        // The CLI's machine envelope is one JSON object on stdout; anything else is a defect worth
        // surfacing rather than guessing at.
        let parsed = null;
        try {
          parsed = JSON.parse(stdout.trim().split('\n').filter(Boolean).pop() ?? '');
        } catch {
          parsed = null;
        }
        if (!parsed) {
          resolve({ ok: false, code, error: stderr.trim() || stdout.trim() || `the command exited ${code} with no JSON` });
          return;
        }
        resolve({ ok: Boolean(parsed.ok), code, envelope: parsed, stderr: stderr.trim() });
      });
    });
  }

  return {
    stateDir: () => stateDir(env),

    /** Every recorded run, newest first. */
    list: () => listRuns(stateDir(env)).map(withLive),

    /** One run's durable record, or null. */
    status: (id) => {
      if (!/^[0-9a-f]{8,64}$/.test(String(id ?? ''))) return null;
      const run = readRun(stateDir(env), id);
      return run ? withLive(summarise(run)) : null;
    },

    /** The changed files of a run (saved original against the revision), for the Compare view. */
    files: id => {
      if (!/^[0-9a-f]{8,64}$/.test(String(id ?? ''))) return null;
      const run = readRun(stateDir(env), id);
      return run ? diffRun(run) : null;
    },

    /** One picture of a run (original or updated), for the Compare view: { path, type } or null. */
    image: (id, relPath, version) => {
      if (!/^[0-9a-f]{8,64}$/.test(String(id ?? ''))) return null;
      const run = readRun(stateDir(env), id);
      const full = run ? runImagePath(run, relPath, version) : null;
      return full ? { path: full, type: imageTypeOf(relPath) } : null;
    },

    /** Stop a run this host is driving. The saved version stays; the run reads as interrupted. */
    cancel: id => {
      const child = live.get(id);
      if (!child) return false;
      try { child.kill('SIGTERM'); } catch { /* already gone */ }
      return true;
    },

    /**
     * Start a run. Returns as soon as the job has been *accepted*; the caller follows the run id.
     *
     * Note the deliberate absence of a `reviseCommand` argument: a revision is performed by the
     * Harness through `integrations/harness/revise_once.py`, and letting an HTTP caller supply an
     * arbitrary shell command would hand a remote caller command execution. The path is configured,
     * not requested.
     */
    async start({ workspace, prompt, attempts = 2, reviseCommand = null, signal } = {}) {
      if (!workspace || typeof workspace !== 'string') {
        return { ok: false, code: 2, error: 'workspace is required' };
      }
      if (!prompt || typeof prompt !== 'string') {
        return { ok: false, code: 2, error: 'prompt is required' };
      }
      const args = ['timeskip', 'run', '--workspace', workspace, '--prompt', prompt,
        '--attempts', String(Math.max(0, Math.min(10, Number(attempts) || 0)))];
      if (reviseCommand) args.push('--revise-command', reviseCommand);
      const result = await runCli(args, { signal });
      if (!result.ok) return { ok: false, code: result.code ?? 6, error: result.error ?? result.stderr };
      const run = result.envelope?.run;
      return run ? { ok: true, run: summarise(run) } : { ok: false, code: 6, error: 'the job reported success but wrote no run record' };
    },

    /**
     * Start a run and return as soon as its record exists; the job goes on without the caller waiting.
     * The caller follows the run id with `status`. A command that ends before any record appears is an error, in the command's words.
     */
    async startDetached({ workspace, prompt, attempts = 1, reviseCommand = null, extraEnv = {} } = {}) {
      if (!workspace || typeof workspace !== 'string') return { ok: false, code: 2, error: 'workspace is required' };
      if (!prompt || typeof prompt !== 'string') return { ok: false, code: 2, error: 'prompt is required' };
      const root = stateDir(env);
      const before = new Set(existsSync(root) ? readdirSync(root) : []);
      const args = ['timeskip', 'run', '--workspace', workspace, '--prompt', prompt,
        '--attempts', String(Math.max(0, Math.min(10, Number(attempts) || 0))), '--output', 'json'];
      if (reviseCommand) args.push('--revise-command', reviseCommand);
      const child = spawn(python, [...pythonArgs, cli, ...args], {
        cwd: repoRoot, windowsHide: true, env: { ...env, ...extraEnv, STRATA_TIMESKIP_DIR: root }, stdio: ['ignore', 'pipe', 'pipe'],
      });
      let out = '', err = '', closed = null;
      child.stdout.on('data', b => { out += b.toString(); });
      child.stderr.on('data', b => { err += b.toString(); });
      const ended = new Promise(resolve => {
        child.on('error', e => { closed = { code: null, error: `could not start the Time-skip command: ${e.message}` }; resolve(); });
        child.on('close', code => { closed = { code }; resolve(); });
      });
      const deadline = Date.now() + 20_000;
      let id = null;
      while (!id && Date.now() < deadline) {
        if (existsSync(root)) {
          for (const name of readdirSync(root)) {
            if (before.has(name)) continue;
            const run = readRun(root, name);
            if (run && run.prompt === prompt) { id = name; break; }
          }
        }
        if (id || closed) break;
        await new Promise(r => setTimeout(r, 100));
      }
      if (!id) {
        if (!closed) { try { child.kill('SIGTERM'); } catch { /* gone */ } return { ok: false, code: 6, error: 'the Time-skip job did not start in time' }; }
        let message = closed.error ?? '';
        try { const parsed = JSON.parse(out.trim().split('\n').filter(Boolean).pop() ?? ''); message = message || parsed?.error?.message || parsed?.error || ''; } catch { /* not JSON */ }
        return { ok: false, code: closed.code ?? 6, error: String(message || err.trim() || out.trim() || 'the Time-skip command ended without a run').slice(0, 600) };
      }
      live.set(id, child);
      void ended.then(() => live.delete(id));
      return { ok: true, run: withLive(summarise(readRun(root, id))) };
    },

    /** Apply the retained result, or preview it. Refuses when the destination changed. */
    async apply({ id, workspace = null, dryRun = false, restore = false, attempt = null, signal } = {}) {
      if (!/^[0-9a-f]{8,64}$/.test(String(id ?? ''))) {
        return { ok: false, code: 2, error: 'a run id is required' };
      }
      const args = ['timeskip', 'apply', id];
      if (workspace) args.push('--workspace', workspace);
      if (dryRun) args.push('--dry-run');
      if (restore) args.push('--restore');
      if (attempt !== null && attempt !== undefined) args.push('--attempt', String(Math.max(1, Math.floor(Number(attempt)) || 1)));
      const result = await runCli(args, { signal });
      if (!result.ok) {
        // Exit 6 is the documented "refused / uncertain" code, which is how a concurrent user edit
        // arrives here. It is a normal outcome, not an internal error.
        return { ok: false, code: result.code ?? 6, conflict: result.code === 6,
                 error: result.error ?? result.stderr };
      }
      return { ok: true, apply: result.envelope?.apply ?? result.envelope?.restore ?? null };
    },
  };
}
