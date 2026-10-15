/**
 * The agent bridge: Strata chats as Harness sessions.
 *
 * What it owns
 *   - the pinned Harness process (one, lazily started, restarted only when a route it needs changed);
 *   - the stable Strata-chat -> Harness-session map, persisted next to the host's other state;
 *   - one `session/follow` stream per live turn, projected into the NDJSON frames the chat UI renders;
 *   - approvals: the Harness `approval/request` waterfall in, the user's Approve/Deny back out;
 *   - Stop: `session/cancel` for exactly one session, resolved only once the turn has really ended.
 *
 * What it deliberately does not do
 *   - it never falls back to another model: a chat picks a route (local instance or saved API
 *     connection) and a missing route is a typed error;
 *   - it never implements Stop by killing the Harness (that would cancel every other chat);
 *   - it never holds a secret outside the environment of the agent process it starts.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { GatewayClient } from './gateway-client.mjs';
import { HarnessProcess, resolveHarness, redact, stopStaleAgent } from './harness-runtime.mjs';
import { buildProfile, routeFor, MODE_PRESETS, modeOf } from './profile.mjs';
import { TurnProjector } from './events.mjs';

export class AgentUnavailable extends Error {
  constructor(message, code = 'agent-unavailable') { super(message); this.name = 'AgentUnavailable'; this.code = code; }
}

/** A timer that never keeps the process alive by itself. */
const after = (ms, fn) => { const t = setTimeout(fn, ms); t.unref?.(); return t; };
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };

/**
 * @param {object} options
 * @param {string} options.dataDir                      the host's data directory (agent state lives in `<dataDir>/agent`)
 * @param {string|null} [options.componentsDir]         where installed components live
 * @param {() => Array} options.localInstances          ready local instances (see profile.mjs)
 * @param {() => Array} options.connections             saved API connections WITH their keys (host-internal)
 * @param {NodeJS.ProcessEnv} [options.env]
 * @param {string} [options.nodePath]                   the Node runtime that runs the Harness
 * @param {(line:string)=>void} [options.log]
 */
export function createAgentBridge({ dataDir, componentsDir = null, localInstances, connections, env = process.env, nodePath = process.execPath, log = () => {}, startTimeoutMs = 120_000 }) {
  const root = join(dataDir, 'agent');
  const home = join(root, 'home');
  const sessionsFile = join(root, 'sessions.json');
  mkdirSync(root, { recursive: true });

  /** chatId -> { sessionId, cwd, createdAt } */
  let sessionMap = {};
  if (existsSync(sessionsFile)) {
    try { sessionMap = JSON.parse(readFileSync(sessionsFile, 'utf8')) ?? {}; } catch { sessionMap = {}; }
  }
  const persist = () => {
    const temp = `${sessionsFile}.${randomUUID()}.tmp`;
    writeFileSync(temp, JSON.stringify(sessionMap, null, 2));
    renameSync(temp, sessionsFile);
  };

  /** @type {{hp:HarnessProcess, gw:GatewayClient, profile:object, clientId:string|null, generation:number, events:object}|null} */
  let rt = null;
  let generation = 0;
  let lastError = null;
  let starting = null;
  const chats = new Map();         // chatId -> { turn, route, routeGeneration }
  const pending = new Map();       // `${sessionId}:${callId}` -> { eventId, sessionId, callId, at }
  const idleWaiters = new Set();

  const chat = id => { let c = chats.get(id); if (!c) { c = { id, turn: null, route: null, routeGeneration: -1 }; chats.set(id, c); } return c; };
  const activeTurns = () => [...chats.values()].filter(c => c.turn).length;
  const notifyIdle = () => { if (activeTurns() === 0) { for (const w of idleWaiters) w(); idleWaiters.clear(); } };
  const waitIdle = (signal, ms) => new Promise((resolve, reject) => {
    if (activeTurns() === 0) return resolve();
    const timer = setTimeout(() => { idleWaiters.delete(done); reject(new AgentUnavailable('Another chat is still running a turn and the model list changed. Try again when it finishes.', 'agent-busy')); }, ms);
    const done = () => { clearTimeout(timer); resolve(); };
    idleWaiters.add(done);
    signal?.addEventListener('abort', () => { clearTimeout(timer); idleWaiters.delete(done); reject(signal.reason ?? new Error('aborted')); }, { once: true });
  });

  function harnessInfo() { return resolveHarness({ env, componentsDir }); }

  /** The capability state the app shows. Never carries a token or a key. */
  function status() {
    const h = harnessInfo();
    if (!h.ok) return { state: 'unavailable', reason: h.reason, code: 'component-missing' };
    const profile = buildProfile({ local: localInstances(), connections: connections() });
    return {
      state: rt?.hp.running ? 'ready' : starting ? 'starting' : lastError ? 'error' : 'idle',
      reason: lastError ?? null,
      harness: { version: h.version },
      runtime: rt?.hp.running ? { pid: rt.hp.pid, generation: rt.generation } : null,
      routes: profile.routes.map(({ route, kind, modelId, name, contextWindow }) => ({ route, kind, modelId, name, contextWindow })),
      activeTurns: activeTurns(),
      pendingApprovals: pending.size,
    };
  }

  async function startRuntime(profile) {
    const h = harnessInfo();
    if (!h.ok) throw new AgentUnavailable(h.reason, 'component-missing');
    mkdirSync(home, { recursive: true });
    const stale = stopStaleAgent(join(root, 'harness.pid'));
    if (stale) log(`agent · stopped a leftover Harness process (pid ${stale}) from an earlier run`);
    const patchFile = join(root, `profile-${profile.signature}.patch.yml`);
    writeFileSync(patchFile, profile.patch);
    const workdir = join(root, 'run');
    const hp = new HarnessProcess({
      harness: h, home, patchFile, cwd: workdir, env: profile.env, nodePath, logFile: join(root, 'harness.log'), pidFile: join(root, 'harness.pid'),
      onExit: exit => {
        log(`agent · the Harness process exited (code ${exit.code})`);
        if (rt?.hp === hp) { lastError = 'The agent process stopped.'; rt = null; }
        for (const c of chats.values()) c.turn?.abortFromRuntime('The agent process stopped before this turn finished.');
      },
    });
    const info = await hp.start({ timeoutMs: startTimeoutMs });
    const gw = new GatewayClient({ baseUrl: info.baseUrl, token: info.token });
    await gw.connect();
    const next = { hp, gw, profile, clientId: null, generation: ++generation, events: null };
    next.events = gw.openStream('$events', {}, {
      onItem: value => onHostEvent(next, value),
      onError: e => log(`agent · event stream: ${redact(e.message, Object.values(profile.env))}`),
    });
    rt = next;
    lastError = null;
    log(`agent · Harness ${h.version ?? ''} ready (pid ${hp.pid}, ${profile.routes.length} route(s))`);
    return next;
  }

  /** Make sure a runtime exists that knows the route this turn needs. */
  async function ensureRuntime(choice, signal) {
    for (;;) {
      if (starting) { await starting.catch(() => {}); continue; }
      const profile = buildProfile({ local: localInstances(), connections: connections() });
      const route = routeFor(profile, choice);
      if (!route) {
        throw new AgentUnavailable(choice?.connectionId
          ? 'This API model is no longer available. Choose another model in the chat picker.'
          : 'No local model is ready for the agent. Load a model in Models, or choose an API model.', 'model-unavailable');
      }
      const known = rt?.hp.running && rt.profile.routes.some(r => r.route === route.route && r.modelId === route.modelId && r.fp === route.fp);
      if (known) return { rt, route };
      starting = (async () => {
        if (rt?.hp.running) {
          await waitIdle(signal, 120_000);
          const old = rt; rt = null;
          old.events?.close(); old.gw.close(); await old.hp.stop();
        }
        try { return await startRuntime(profile); } catch (e) { lastError = redact(String(e.message ?? e), Object.values(profile.env)); throw e; }
      })();
      try { await starting; } finally { starting = null; }
    }
  }

  function onHostEvent(runtime, value) {
    if (value.type === 'ready') { runtime.clientId = value.clientId; return; }
    if (value.type === 'waterfall' && value.event === 'approval/request') {
      const r = value.request ?? {};
      if (!r.callId) return;
      pending.set(`${value.agentId}:${r.callId}`, { eventId: value.eventId, sessionId: value.agentId, callId: r.callId, tool: r.toolName, reason: r.reason ?? null, at: Date.now(), clientId: runtime.clientId });
    } else if (value.type === 'waterfall' && value.event === 'user-questions/request') {
      // The app has no surface for the agent's own questions yet. Declaring that nobody can answer makes
      // the tool return an error the model reads and carries on from, instead of a turn that waits forever.
      void runtime.gw.answerEvent({ clientId: runtime.clientId, eventId: value.eventId, outcome: 'next' }).catch(() => {});
    } else if (value.type === 'cancel') {
      for (const [k, v] of pending) if (v.eventId === value.eventId) pending.delete(k);
    }
  }

  async function sessionFor(c, cwd, runtime) {
    const known = sessionMap[c.id];
    if (known) return { sessionId: known.sessionId, fresh: false };
    const created = await runtime.gw.rpc('session/create', { request: { cwd } });
    sessionMap[c.id] = { sessionId: created.sessionId, cwd, createdAt: new Date().toISOString() };
    persist();
    return { sessionId: created.sessionId, fresh: true };
  }

  /**
   * Put the session into the preset of the chat's mode, through the Harness's own `/permission` command, and
   * check that it took. A turn never starts in a mode other than the one asked for: if the switch cannot be
   * confirmed the turn fails closed with a typed error instead of running with unknown permissions.
   */
  async function applyMode(runtime, sessionId, mode) {
    const want = MODE_PRESETS[mode].key;
    const read = async () => {
      const r = await runtime.gw.rpc('session/projections', { request: { sessionId, keys: ['permissions'] } });
      return r?.values?.permissions?.currentValue ?? null;
    };
    try {
      if (await read() === want) return;
      const done = await runtime.gw.rpc('commands/execute', { agentId: sessionId, line: `/permission ${want}`, submittedAttachments: [] });
      if (done?.result?.kind !== 'success') throw new Error(done?.result?.text ?? 'the agent did not recognise the command');
      const now = await read();
      if (now !== want) throw new Error(`the session reports "${now}" instead of "${want}"`);
    } catch (e) {
      throw new AgentUnavailable(`The agent could not switch this chat to ${MODE_PRESETS[mode].name} mode, so nothing ran: ${e.message}`, 'mode-unavailable');
    }
  }

  /**
   * Run one agent turn for a Strata chat and stream it as frames.
   * `signal` means the client went away: the turn keeps running (that is not Stop).
   */
  async function runTurn(options) {
    const c = chat(options.chatId);
    if (c.turn || c.starting) throw new AgentUnavailable('This chat is already running a turn.', 'turn-active');
    // From here until the Harness has the prompt, Stop has no session to cancel; it cancels the start.
    c.starting = true; c.cancelStart = false;
    try { return await runTurnInner(c, options); } finally { c.starting = false; }
  }

  async function runTurnInner(c, { chatId, message, workspacePath, choice = null, mode = null, history = [], onFrame, requestId = `req-${randomUUID()}`, signal }) {
    let send = onFrame;
    signal?.addEventListener('abort', () => { send = () => {}; }, { once: true });
    const emit = frame => { try { send(frame); } catch { send = () => {}; } };

    emit({ type: 'queued', requestId });
    const { rt: runtime, route } = await ensureRuntime(choice, signal);
    const { sessionId, fresh } = await sessionFor(c, workspacePath, runtime);
    const cancelled = () => { emit({ type: 'done', finishReason: 'aborted', cancelled: true, tokensIn: null, tokensOut: null, usageSource: 'unavailable', durationMs: 0, model: route.name, contextMax: route.contextWindow }); return { cancelled: true }; };
    if (c.cancelStart) return cancelled();

    if (c.route !== `${route.route}/${route.modelId}` || c.routeGeneration !== runtime.generation) {
      try {
        await runtime.gw.rpc('session/selectModel', { request: { sessionId, provider: route.route, model: route.modelId } });
      } catch (e) {
        if (e.code === 'session/not-found') { delete sessionMap[c.id]; persist(); }
        throw new AgentUnavailable(`The agent could not use ${route.name}: ${e.message}`, 'model-unavailable');
      }
      c.route = `${route.route}/${route.modelId}`; c.routeGeneration = runtime.generation;
    }

    // The chat's permission mode is read once, here, for this whole turn. A change the user makes while the
    // turn runs applies to the next one, so a running turn never sees its permissions move under it.
    const appliedMode = modeOf(mode);
    await applyMode(runtime, sessionId, appliedMode);

    const projector = new TurnProjector({ policy: MODE_PRESETS[appliedMode].approval });
    const turn = {
      sessionId, projector, route, requestId, live: false, lastSeq: -1,
      ended: deferred(), follow: null, snapshot: deferred(), finished: false, prompted: false, stopBeforePrompt: false,
      abortFromRuntime: reason => finishTurn(reason),
    };
    c.turn = turn;
    function finishTurn(failure) {
      if (turn.finished) return;
      turn.finished = true;
      turn.follow?.close();
      const frames = failure ? [{ type: 'error', message: failure, code: 'AGENT_STOPPED' }] : projector.finish({ model: route.name, contextMax: route.contextWindow });
      for (const f of frames) emit(f);
      c.turn = null;
      turn.ended.resolve({ reason: projector.ended, failure: failure ?? null });
      notifyIdle();
    }

    emit({ type: 'start', model: route.name, contextMax: route.contextWindow, demo: false, remoteModel: route.kind === 'api', agent: true, mode: appliedMode, sessionId, requestId, skills: [] });

    turn.follow = runtime.gw.openStream('session/follow', { request: { address: { kind: 'session', sessionId }, assistantStream: true } }, {
      onItem: item => {
        if (item.type === 'snapshot') {
          if (turn.lastSeq < 0) turn.lastSeq = item.cursor;
          else for (const r of item.records ?? []) if (r.type === 'event' && r.event.seq > turn.lastSeq) handle({ type: 'event', event: r.event });
          turn.snapshot.resolve();
          return;
        }
        handle(item);
      },
      onError: e => finishTurn(`The agent lost the session stream: ${e.message}`),
    });

    function handle(item) {
      if (turn.finished) return;
      if (item.type === 'event') {
        const e = item.event;
        if (e.seq <= turn.lastSeq) return;
        turn.lastSeq = e.seq;
        // The turn is ours from the user message that carries our request id. Until then an older,
        // detached turn on the same session may still be finishing; its events are not ours.
        if (!turn.live) {
          if (e.type === 'user/message' && e.data?.source?.rpcId === requestId) turn.live = true;
          return;
        }
      } else if (!turn.live) return;
      for (const f of projector.consume(item)) emit(f);
      if (item.type === 'event' && item.event.type === 'turn/end') finishTurn(null);
    }

    const snapshotTimer = {};
    await Promise.race([turn.snapshot.promise, new Promise((_, rej) => { snapshotTimer.t = after(20_000, () => rej(new Error('the session stream did not open'))); })])
      .catch(e => { finishTurn(e.message); })
      .finally(() => clearTimeout(snapshotTimer.t));
    if (turn.finished) return { error: true };
    if (turn.stopBeforePrompt || c.cancelStart) { projector.ended = { kind: 'aborted', reason: { kind: 'user' } }; finishTurn(null); return { cancelled: true }; }

    let prompt = message;
    if (fresh && history.length) prompt = withEarlierConversation(history, message);
    try {
      await runtime.gw.rpc('session/prompt', { request: { requestId, sessionId, mode: 'queue', content: [{ type: 'text', text: prompt }] } });
    } catch (e) {
      finishTurn(`The agent did not accept the message: ${e.message}`);
      return { error: true };
    }
    turn.prompted = true;
    if (turn.stopBeforePrompt) {
      // Stop landed in the instant between 'prompt sent' and 'prompt accepted'. A cancel before the turn
      // starts arms nothing, so wait for the turn to be ours, then cancel it.
      for (let i = 0; i < 100 && !turn.live && !turn.finished; i++) await new Promise(r => setTimeout(r, 50));
      if (!turn.finished) await runtime.gw.rpc('session/cancel', { request: { sessionId } }).catch(() => {});
    }

    // Wait for the turn, or for the client to leave. Leaving does not stop the turn.
    const gone = new Promise(r => signal?.addEventListener('abort', () => r('detached'), { once: true }));
    const outcome = await Promise.race([turn.ended.promise, gone]);
    if (outcome === 'detached') return { detached: true, sessionId, ended: turn.ended.promise.then(r => ({ ...r, tokensOut: projector.tokensOut })) };
    return { sessionId, reason: outcome.reason, tokensIn: projector.tokensIn, tokensOut: projector.tokensOut };
  }

  /** Answer one pending approval. `decision` is `approve` or `deny`. */
  async function approve({ chatId, callId, decision }) {
    const sessionId = sessionMap[chatId]?.sessionId;
    // The durable `approval/asked` (which makes the card say "needs approval") and the waterfall frame
    // (which makes it answerable) travel on different streams; a fast click can beat the second one.
    let entry = sessionId ? pending.get(`${sessionId}:${callId}`) : null;
    for (let i = 0; !entry && sessionId && i < 40; i++) {
      await new Promise(r => setTimeout(r, 50));
      entry = pending.get(`${sessionId}:${callId}`);
    }
    if (!entry) throw new AgentUnavailable('This approval is no longer pending.', 'approval-gone');
    if (!rt?.hp.running) throw new AgentUnavailable('The agent is not running, so this approval has expired.', 'approval-gone');
    pending.delete(`${sessionId}:${callId}`);
    try {
      await rt.gw.answerEvent({ clientId: entry.clientId ?? rt.clientId, eventId: entry.eventId, outcome: decision === 'approve' ? 'allowed-once' : 'rejected' });
    } catch (e) {
      throw new AgentUnavailable(`The answer was not accepted: ${e.message}`, 'approval-gone');
    }
    return { ok: true };
  }

  /**
   * Stop one chat's turn. Resolves only once the turn has ended (or the wait times out), because the
   * Harness acknowledges a cancel as admitted, not as done.
   */
  async function stop(chatId, { waitMs = 20_000 } = {}) {
    const c = chats.get(chatId);
    const turn = c?.turn;
    if (!turn && c?.starting) { c.cancelStart = true; return { stopped: true, ended: 'cancelled-before-start', ms: 0 }; }
    if (!turn || !rt) return { stopped: false, reason: 'no active turn' };
    const t0 = Date.now();
    if (!turn.prompted) {
      // The Harness has not been given the prompt yet, so there is nothing to cancel there.
      turn.stopBeforePrompt = true;
      const early = await Promise.race([turn.ended.promise, new Promise(r => setTimeout(() => r(null), 100))]);
      if (early) return { stopped: true, ended: early.reason?.kind ?? 'aborted', ms: Date.now() - t0 };
    }
    await rt.gw.rpc('session/cancel', { request: { sessionId: turn.sessionId } });
    let timer; const result = await Promise.race([turn.ended.promise, new Promise(r => { timer = after(waitMs, () => r(null)); })]);
    clearTimeout(timer);
    if (!result) return { stopped: false, pending: true, reason: 'the turn is still stopping' };
    return { stopped: true, ended: result.reason?.kind ?? 'unknown', ms: Date.now() - t0 };
  }

  function pendingFor(chatId) {
    const sessionId = sessionMap[chatId]?.sessionId;
    return [...pending.values()].filter(p => p.sessionId === sessionId).map(({ callId, tool, reason, at }) => ({ callId, tool, reason, at }));
  }

  async function close() {
    const r = rt; rt = null;
    if (r) { r.events?.close(); r.gw.close(); await r.hp.stop(); }
  }

  return { status, runTurn, approve, stop, pendingFor, close, sessionOf: chatId => sessionMap[chatId]?.sessionId ?? null, isRunning: chatId => !!chats.get(chatId)?.turn, _debug: () => ({ rt, sessionMap }) };
}

/** The Harness session of a chat that existed before the agent did knows nothing; carry a bounded summary. */
export function withEarlierConversation(history, message) {
  const rows = history.filter(m => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim()).slice(-12);
  if (!rows.length) return message;
  const text = rows.map(m => `${m.role}: ${m.content.length > 1500 ? `${m.content.slice(0, 1499)}…` : m.content}`).join('\n');
  return `Earlier in this conversation, before you were given tools:\n${text}\n\nCurrent message:\n${message}`;
}
