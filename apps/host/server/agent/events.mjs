/**
 * Pure projections from Harness facts to the frames the chat UI already renders.
 *
 * No I/O here, so the mapping can be tested without an agent. The inputs are the two things a
 * `session/follow` stream carries: durable session events (`tool/call`, `approval/asked`,
 * `tool/result`, `turn/end`, ...) and the process-local `assistant-stream` frames (text and
 * reasoning deltas, usage).
 */

const SHELL = new Set(['pwsh', 'bash', 'job_kill', 'job_output', 'job_list']);
const WRITE = new Set(['write', 'edit']);
const READ = new Set(['read', 'glob', 'grep', 'read_image', 'working_directory', 'todo_write', 'get_goal', 'list_agents', 'skill', 'present']);

export function toolScope(name) {
  if (SHELL.has(name)) return 'shell';
  if (WRITE.has(name)) return 'write';
  if (READ.has(name)) return 'read';
  return 'connected';
}

const clip = (value, max) => {
  const text = String(value ?? '');
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

export function parseArguments(raw) {
  if (raw && typeof raw === 'object') return raw;
  try { const v = JSON.parse(String(raw ?? '{}')); return v && typeof v === 'object' ? v : {}; } catch { return {}; }
}

/** The ToolCall fields the card needs, from one Harness tool call. */
export function presentTool({ callId, name, args, reason = null }) {
  const scope = toolScope(name);
  const preview = scope === 'shell' ? args.command ?? ''
    : scope === 'write' ? args.file_path ?? args.path ?? args.filePath ?? ''
      : args.file_path ?? args.path ?? args.pattern ?? args.url ?? args.query ?? JSON.stringify(args);
  const effect = reason ?? (scope === 'shell' ? (args.description ?? 'Runs a command in the workspace.')
    : scope === 'write' ? 'Changes a file in the workspace.'
      : scope === 'read' ? 'Reads from the workspace.' : 'Uses a connected tool.');
  return { id: callId, name, args, preview: clip(preview, 600), effect: clip(effect, 400), scope };
}

/** Text of a tool-result message, bounded. */
export function resultText(message) {
  const parts = Array.isArray(message?.content) ? message.content : [];
  const text = parts.map(p => (p?.type === 'text' ? p.text : p?.type ? `[${p.type}]` : '')).join('\n').trim();
  return clip(text || '(no output)', 4000);
}

/**
 * Accumulates one turn's frames. Feed it follow items; it returns the frames to send.
 * It owns the per-turn state (tool cards, usage, text emitted) so the I/O layer stays thin.
 */
export class TurnProjector {
  constructor({ startedAt = Date.now() } = {}) {
    this.startedAt = startedAt;
    this.tools = new Map();          // callId -> { card, calledAt }
    this.tokensIn = 0;
    this.tokensOut = 0;
    this.usageSeen = false;
    this.firstTokenAt = null;
    this.emittedText = new Map();    // attemptId -> chars sent live
    this.ended = null;               // the turn/end reason, once seen
    this.denied = new Set();         // callIds a human or the policy rejected
    this.asked = new Map();          // approval id -> callId
  }

  #card(callId, patch) {
    const entry = this.tools.get(callId);
    if (!entry) return null;
    entry.card = { ...entry.card, ...patch };
    return { type: 'agent-tool', tool: entry.card };
  }

  /** Process one `session/follow` item (event or assistant-stream). Returns frames. */
  consume(item) {
    const out = [];
    if (item.type === 'assistant-stream') {
      const f = item.frame;
      if (f.type === 'chunk') {
        const c = f.chunk;
        if (c.type === 'text-delta' && c.text) {
          this.firstTokenAt ??= Date.now();
          this.emittedText.set(f.attemptId, (this.emittedText.get(f.attemptId) ?? 0) + c.text.length);
          out.push({ type: 'delta', text: c.text });
        } else if (c.type === 'reasoning-delta' && c.text) {
          this.firstTokenAt ??= Date.now();
          out.push({ type: 'reasoning', text: c.text });
        } else if (c.type === 'usage' && c.usage) {
          this.usageSeen = true;
          this.tokensIn += Number(c.usage.inputTokens) || 0;
          this.tokensOut += Number(c.usage.outputTokens) || 0;
        }
      }
      return out;
    }
    if (item.type !== 'event') return out;
    const e = item.event;
    const d = e.data ?? {};
    switch (e.type) {
      case 'assistant/message': {
        // Reconcile a text the live frames did not fully carry (a socket gap), never duplicating it.
        const live = [...this.emittedText.values()].pop() ?? 0;
        const text = (d.message?.content ?? []).filter(p => p.type === 'text').map(p => p.text).join('');
        if (text.length > live && live > 0) out.push({ type: 'delta', text: text.slice(live) });
        this.emittedText.clear();
        break;
      }
      case 'tool/call': {
        const args = parseArguments(d.arguments);
        const card = { ...presentTool({ callId: d.callId, name: d.name, args }), state: 'running' };
        this.tools.set(d.callId, { card, calledAt: e.time });
        out.push({ type: 'agent-tool', tool: card });
        break;
      }
      case 'approval/asked': {
        if (!d.callId) break;
        this.asked.set(d.id, d.callId);
        const entry = this.tools.get(d.callId);
        const patch = { state: 'proposed', effect: clip(d.reason ?? entry?.card.effect ?? 'Needs your approval.', 400) };
        const frame = this.#card(d.callId, patch);
        if (frame) out.push(frame);
        break;
      }
      case 'approval/decided': {
        const callId = this.asked.get(d.id);
        const frame = callId ? this.decide(callId, d.outcome) : null;
        if (frame) out.push(frame);
        break;
      }
      case 'tool/result': {
        const callId = d.message?.toolCallId;
        const entry = this.tools.get(callId);
        if (!entry) break;
        const isError = d.message?.isError === true;
        const denied = this.denied.has(callId);
        const frame = this.#card(callId, {
          state: denied ? 'denied' : isError ? 'error' : 'done',
          result: resultText(d.message),
          durationMs: Math.max(0, (e.time ?? 0) - entry.calledAt),
        });
        if (frame) out.push(frame);
        break;
      }
      case 'turn/end':
        this.ended = d.reason ?? { kind: 'completed' };
        break;
      default:
        break;
    }
    return out;
  }

  /** An approval was decided. `outcome` is the Harness vocabulary. */
  decide(callId, outcome) {
    if (!this.tools.has(callId)) return null;
    if (outcome === 'allowed-once') return this.#card(callId, { state: 'running' });
    this.denied.add(callId);
    return this.#card(callId, { state: 'denied', result: outcome === 'rejected' ? 'Denied. Nothing ran.' : outcome === 'cancelled' ? 'Cancelled. Nothing ran.' : 'No one could answer this request, so nothing ran.' });
  }

  /** The terminal frames for a finished turn. */
  finish({ model, contextMax } = {}) {
    const reason = this.ended ?? { kind: 'interrupted' };
    const durationMs = Date.now() - this.startedAt;
    const usage = this.usageSeen
      ? { tokensIn: this.tokensIn, tokensOut: this.tokensOut, usageSource: 'reported' }
      : { tokensIn: null, tokensOut: null, usageSource: 'unavailable' };
    const base = { durationMs, ttftMs: this.firstTokenAt ? this.firstTokenAt - this.startedAt : null, model, contextMax };
    switch (reason.kind) {
      case 'completed':
        return [{ type: 'done', ...base, ...usage, finishReason: 'stop' }];
      case 'max-tokens':
        return [{ type: 'done', ...base, ...usage, finishReason: 'length' }];
      case 'aborted':
        return [{ type: 'done', ...base, ...usage, finishReason: 'aborted', cancelled: true }];
      case 'error':
        return [{ type: 'error', message: friendlyFailure(reason.error), code: reason.error?.code ?? 'UNKNOWN' }];
      case 'blocked':
        return [{ type: 'error', message: 'The agent stopped because this turn was blocked by policy.', code: 'BLOCKED' }];
      default:
        return [{ type: 'error', message: 'The agent turn was interrupted before it finished.', code: 'INTERRUPTED' }];
    }
  }
}

/** A model-route failure in words a user can act on. */
export function friendlyFailure(failure) {
  const message = String(failure?.message ?? 'The model request failed.');
  if (/context window|too long|exceeds/i.test(message)) return `The conversation no longer fits the model's context window. ${clip(message, 200)}`;
  if (/ECONNREFUSED|fetch failed|connect/i.test(message)) return `The model is not reachable. ${clip(message, 200)}`;
  if (/401|403|credential|api key/i.test(message)) return 'The model rejected the credentials. Check the connection in Settings.';
  return clip(message, 400);
}
