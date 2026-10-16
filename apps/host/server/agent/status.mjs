/**
 * Live status for a running turn: one small `status` frame that says what the assistant is doing right now.
 *
 * The page draws only the LATEST status, so a status must be replaced the moment it stops being true. This reporter therefore sends a
 * new frame on every change of phase or target (at most one per `gapMs`, the newest change wins), and between changes only repeats the
 * frame to refresh a counter (reasoning tokens, bytes written) every `beatMs`. It never sends the same words twice in a row.
 *
 *   { type:'status', seq, phase, code, text, args?, tokens?, elapsedMs }
 *
 * phases: starting waiting thinking answering preparing tool approval helpers goal idle   (see notes-split-agent3.md)
 */

/** The sentence for a tool, in the person's words: what it is doing to what ("Writing pagoda.html…"). */
export function toolStatusText(name, args = {}, { bytes = 0 } = {}) {
  const base = value => String(value ?? '').split(/[\\/]/).filter(Boolean).pop() ?? '';
  const file = base(args.file_path ?? args.path ?? args.filePath);
  const size = bytes >= 1500 ? ` (${(bytes / 1024).toFixed(bytes >= 10240 ? 0 : 1)} KB)` : '';
  switch (name) {
    case 'write': return file ? `Writing ${file}…${size}` : `Writing a file…${size}`;
    case 'edit': return file ? `Changing ${file}…${size}` : `Changing a file…${size}`;
    case 'read': return file ? `Reading ${file}…` : 'Reading a file…';
    case 'read_image': return 'Looking at a picture…';
    case 'pwsh': case 'bash': return 'Running a command…';
    case 'glob': case 'grep': return 'Searching your files…';
    case 'web_search': return 'Searching the web…';
    case 'web_fetch': { let host = ''; try { host = new URL(args.url).hostname; } catch { /* partial address */ } return host ? `Opening ${host}…` : 'Opening a web page…'; }
    case 'todo_write': return 'Updating its to-do list…';
    case 'skill': return 'Using a saved skill…';
    case 'present': return 'Showing you a result…';
    case 'workflow': return 'Running a multi-step plan…';
    case 'job_list': case 'job_output': case 'job_kill': return 'Checking background tasks…';
    default:
      if (name.startsWith('mcp__browser__')) return name.endsWith('navigate') ? (args.url ? `Opening ${args.url}…` : 'Opening a page…') : 'Using the browser…';
      if (name.startsWith('mcp__strata-instructions__')) return 'Reading your instructions…';
      if (name.startsWith('mcp__strata-commands__')) return 'Running your tool…';
      return 'Using a tool…';
  }
}

const COUNTERS = new Set(['tokens', 'bytes']);
const keyOf = u => `${u.phase}|${u.code}|${JSON.stringify(Object.fromEntries(Object.entries(u.args ?? {}).filter(([k]) => !COUNTERS.has(k))))}`;

export class StatusReporter {
  /**
   * @param {{emit: (frame: object) => void, now?: () => number, gapMs?: number, beatMs?: number, timers?: {set: Function, clear: Function}}} options
   */
  constructor({ emit, now = Date.now, gapMs = 300, beatMs = 2000, timers = { set: setTimeout, clear: clearTimeout } }) {
    Object.assign(this, { emitFrame: emit, now, gapMs, beatMs, timers });
    this.seq = 0;
    this.current = null;      // what the page was last told
    this.wanted = null;       // the newest thing that is true now
    this.phaseStartedAt = 0;
    this.lastEmitAt = -Infinity;
    this.timer = null;
    this.closed = false;
  }

  /** Say what is true now. `{phase, code, text, args?, tokens?}`. */
  set(update) {
    if (this.closed) return;
    const next = { ...update, args: update.args ? { ...update.args } : undefined };
    const changed = !this.wanted || keyOf(next) !== keyOf(this.wanted);
    if (changed) this.phaseStartedAt = this.now();
    this.wanted = next;
    const sinceEmit = this.now() - this.lastEmitAt;
    const due = changed ? this.gapMs : this.beatMs;
    if (!this.current || sinceEmit >= due) this.#flush();
    else if (this.#differs()) this.#arm(due - sinceEmit);
  }

  /** Nothing is happening any more: tell the page to drop the line. */
  clear() {
    if (this.closed) return;
    if (this.current && this.current.phase !== 'idle') { this.wanted = { phase: 'idle', code: 'idle', text: '' }; this.#flush(); }
    this.dispose();
  }

  dispose() { this.closed = true; if (this.timer) this.timers.clear(this.timer); this.timer = null; }

  #differs() {
    const a = this.wanted, b = this.current;
    return !b || a.text !== b.text || a.tokens !== b.tokens;
  }

  #arm(ms) {
    if (this.timer || this.closed) return;
    this.timer = this.timers.set(() => { this.timer = null; if (!this.closed && this.#differs()) this.#flush(); }, Math.max(0, ms));
    this.timer.unref?.();
  }

  #flush() {
    if (this.timer) { this.timers.clear(this.timer); this.timer = null; }
    const w = this.wanted;
    if (!w || (this.current && this.current.text === w.text && this.current.phase === w.phase && this.current.tokens === w.tokens)) return;
    this.current = w;
    this.lastEmitAt = this.now();
    this.emitFrame({
      type: 'status', seq: ++this.seq, phase: w.phase, code: w.code, text: w.text,
      ...(w.args ? { args: w.args } : {}), ...(w.tokens !== undefined ? { tokens: w.tokens } : {}),
      elapsedMs: Math.max(0, this.now() - this.phaseStartedAt),
    });
  }
}

/** Counts streamed tokens: one per chunk, or characters / 4 when a server batches many characters into a chunk. */
export class TokenCount {
  constructor() { this.chunks = 0; this.chars = 0; }
  add(text) { this.chunks += 1; this.chars += String(text).length; }
  get tokens() { return this.chars / Math.max(1, this.chunks) < 8 ? this.chunks : Math.ceil(this.chars / 4); }
}

export const fmtTokens = n => Number(n).toLocaleString('en-US');

/**
 * The same status line for a plain chat turn (no tools): the request is being read, the model is thinking (tokens), the model is writing.
 * Wraps the function that writes frames to the client; the returned function has `dispose()` for a client that left early.
 */
export function withPlainStatus(write, options = {}) {
  const reporter = new StatusReporter({ emit: write, ...options });
  const reasoning = new TokenCount();
  reporter.set({ phase: 'starting', code: 'starting', text: 'Starting…' });
  const frame = f => {
    switch (f.type) {
      case 'start': reporter.set({ phase: 'waiting', code: 'reading-prompt', text: 'Reading your message…' }); break;
      case 'reasoning': reasoning.add(f.text ?? ''); reporter.set({ phase: 'thinking', code: 'thinking', text: `Thinking… ${fmtTokens(reasoning.tokens)} tokens`, tokens: reasoning.tokens, args: { tokens: reasoning.tokens } }); break;
      case 'delta': reporter.set({ phase: 'answering', code: 'answering', text: 'Writing the answer…' }); break;
      case 'done': case 'error': reporter.clear(); break;
      default: break;
    }
    write(f);
  };
  frame.dispose = () => reporter.dispose();
  return frame;
}
