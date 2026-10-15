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

/** Delegation tools have no card: the helper agents show as rows, and the tool's raw reply ("started subagent <id>") is not for the person. */
export const NO_CARD = new Set(['subagent', 'subagent_fork', 'send_message', 'interrupt_agent', 'list_agents']);

export function toolScope(name) {
  if (SHELL.has(name)) return 'shell';
  if (WRITE.has(name)) return 'write';
  if (READ.has(name)) return 'read';
  return 'connected';
}

/** Shown on a card the mode refused (Plan, Bypass): the model's own error text would say "the user rejected". */
export const NOT_IN_MODE = 'Not allowed in this mode. Nothing ran.';

const clip = (value, max) => {
  const text = String(value ?? '');
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

export function parseArguments(raw) {
  if (raw && typeof raw === 'object') return raw;
  try { const v = JSON.parse(String(raw ?? '{}')); return v && typeof v === 'object' ? v : {}; } catch { return {}; }
}

/**
 * One plain sentence and one short detail per tool, so a card never shows raw JSON or an internal tool name as its
 * explanation. Unknown tools (a connected MCP server's) fall back to a generic sentence with their arguments.
 */
const WORDS = {
  read: a => ['Reads a file.', a.file_path ?? a.path],
  glob: a => ['Looks for files by name.', a.pattern],
  grep: a => ['Searches inside files.', a.pattern ?? a.query],
  read_image: a => ['Looks at a picture.', a.file_path ?? a.path],
  working_directory: () => ['Checks which folder it is working in.', ''],
  todo_write: a => ['Updates its to-do list.', Array.isArray(a.todos) ? a.todos.map(t => t?.content ?? t?.title ?? '').filter(Boolean).join('; ') : ''],
  skill: a => ['Uses a saved skill.', a.name ?? a.skill],
  present: a => ['Shows you a result.', a.file_path ?? a.path ?? a.title],
  create_goal: a => ['Sets a goal to work towards.', a.objective],
  update_goal: a => ['Updates the goal.', a.objective ?? a.status ?? ''],
  get_goal: () => ['Checks the current goal.', ''],
  web_fetch: a => ['Opens a web page.', a.url],
  web_search: a => ['Searches the web.', Array.isArray(a.queries) ? a.queries.join('; ') : a.query],
  workflow: a => ['Runs a multi-step plan.', a.description ?? a.name ?? ''],
  job_list: () => ['Checks the background tasks.', ''],
  job_output: a => ['Reads what a background task printed.', a.job_id ?? a.id],
  job_kill: a => ['Stops a background task.', a.job_id ?? a.id],
  lsp: a => ['Looks up how the code fits together.', a.file_path],
};

/** The ToolCall fields the card needs, from one Harness tool call. */
export function presentTool({ callId, name, args, reason = null }) {
  const scope = toolScope(name);
  const words = WORDS[name]?.(args);
  const preview = scope === 'shell' && !words ? args.command ?? ''
    : scope === 'write' ? args.file_path ?? args.path ?? args.filePath ?? ''
      : words ? words[1] ?? '' : args.file_path ?? args.path ?? args.pattern ?? args.url ?? args.query ?? JSON.stringify(args);
  const effect = reason ?? (words ? words[0] : scope === 'shell' ? (args.description ?? 'Runs a command in the workspace.')
    : scope === 'write' ? 'Changes a file in the workspace.'
      : scope === 'read' ? 'Reads from the workspace.' : 'Uses a connected tool.');
  return { id: callId, name, args, preview: clip(preview, 600), effect: clip(effect, 400), scope };
}

/** A tool's failure in plain words, or null when there is nothing better than the original text. */
export function plainToolError(text) {
  const t = String(text ?? '');
  if (/non-public ip|enotfound|getaddrinfo|web fetch failed|econnrefused|econnreset|etimedout|fetch failed/i.test(t)) return 'That web page could not be opened. Check the address and your internet connection.';
  if (/invalid arguments|missing required property|not a declared property/i.test(t)) return 'The assistant asked this tool in the wrong way. It will usually try again.';
  if (/unknown or no longer available/i.test(t)) return 'That skill is not available.';
  if (/cannot modify|must (?:first )?be read|has not been read|changed since/i.test(t)) return 'The file could not be changed. It may need to be read again first, or it changed in the meantime.';
  if (/sandbox|access denied|permission denied|eacces|eperm/i.test(t)) return 'The computer did not allow this.';
  if (/timed out|timeout/i.test(t)) return 'This took too long and was stopped.';
  if (/extension|not a supported image|cannot read/i.test(t)) return 'This file could not be read.';
  return null;
}

/** A failed tool's result: the plain sentence first, the original text after it (the card's disclosure is the Details). */
export function withPlain(raw) {
  const plain = plainToolError(raw);
  return plain ? `${plain}\n\nDetails: ${raw.replace(/^Error:\s*/, '')}` : raw;
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
  /**
   * `policy` is the session's approval policy for this turn. Under `never` the Harness rejects an ask by itself, so
   * no question ever reaches the person: the card goes straight from running to denied, without a moment where it
   * offers Approve and Deny buttons that would do nothing.
   */
  constructor({ startedAt = Date.now(), policy = 'ask' } = {}) {
    this.startedAt = startedAt;
    this.policy = policy;
    this.byPolicy = new Set();       // callIds the approval policy rejected without asking
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
          this.emittedAny = true;
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
        if (NO_CARD.has(d.name)) break;
        const args = parseArguments(d.arguments);
        const card = { ...presentTool({ callId: d.callId, name: d.name, args }), state: 'running' };
        this.tools.set(d.callId, { card, calledAt: e.time });
        out.push({ type: 'agent-tool', tool: card });
        break;
      }
      case 'approval/asked': {
        if (!d.callId) break;
        this.asked.set(d.id, d.callId);
        if (this.policy === 'never') break;
        const entry = this.tools.get(d.callId);
        const patch = { state: 'proposed', effect: clip(d.reason ?? entry?.card.effect ?? 'Needs your approval.', 400) };
        const frame = this.#card(d.callId, patch);
        if (frame) out.push(frame);
        break;
      }
      case 'approval/decided': {
        const callId = this.asked.get(d.id);
        if (callId && this.policy === 'never' && d.outcome !== 'allowed-once') {
          this.byPolicy.add(callId); this.denied.add(callId);
          const frame = this.#card(callId, { state: 'denied', result: NOT_IN_MODE });
          if (frame) out.push(frame);
          break;
        }
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
          // A card the person (or the mode) refused keeps its plain sentence; the model's own error text ("the user rejected ...") is not for them.
          result: this.byPolicy.has(callId) ? NOT_IN_MODE : denied ? (entry.card.result ?? 'Denied. Nothing ran.') : isError ? withPlain(resultText(d.message)) : resultText(d.message),
          durationMs: Math.max(0, (e.time ?? 0) - entry.calledAt),
        });
        if (frame) out.push(frame);
        break;
      }
      case 'turn/start':
        // The parent woke up again (a helper finished and reported back): the message goes on, as a new paragraph.
        this.ended = null;
        if (this.emittedAny) out.push({ type: 'delta', text: '\n\n' });
        break;
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
    // `approved` is what lets the card say "Approved" (the user answered) rather than "Ran" (it never asked).
    if (outcome === 'allowed-once') return this.#card(callId, { state: 'running', approved: true });
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
