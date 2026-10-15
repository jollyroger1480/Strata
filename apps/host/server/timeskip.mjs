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
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

/** Where runs are recorded. Mirrors the CLI's own default resolution. */
function stateDir(env = process.env) {
  if (env.STRATA_TIMESKIP_DIR) return env.STRATA_TIMESKIP_DIR;
  const base = env.XDG_STATE_HOME || join(env.HOME || '.', '.local', 'state');
  return join(base, 'strata', 'time-skip');
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

/**
 * Run one `strata timeskip …` command and collect its result.
 *
 * The command is given a bounded time. A Time-skip run can legitimately take many minutes (each
 * attempt is an agent turn), so the default bound is generous, and exceeding it is reported as a
 * timeout rather than as success.
 */
export function createTimeSkip({ repoRoot, python, timeoutMs = 30 * 60 * 1000, env = process.env } = {}) {
  const cli = join(repoRoot, 'strata.py');

  function runCli(args, { signal } = {}) {
    return new Promise((resolve) => {
      const root = stateDir(env);
      const child = spawn(python, [cli, ...args, '--output', 'json'], {
        cwd: repoRoot,
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
    list: () => listRuns(stateDir(env)),

    /** One run's durable record, or null. */
    status: (id) => {
      if (!/^[0-9a-f]{8,64}$/.test(String(id ?? ''))) return null;
      const run = readRun(stateDir(env), id);
      return run ? summarise(run) : null;
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

    /** Apply the retained result, or preview it. Refuses when the destination changed. */
    async apply({ id, workspace = null, dryRun = false, restore = false, signal } = {}) {
      if (!/^[0-9a-f]{8,64}$/.test(String(id ?? ''))) {
        return { ok: false, code: 2, error: 'a run id is required' };
      }
      const args = ['timeskip', 'apply', id];
      if (workspace) args.push('--workspace', workspace);
      if (dryRun) args.push('--dry-run');
      if (restore) args.push('--restore');
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
