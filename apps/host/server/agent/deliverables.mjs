/**
 * What a turn made: files (deliverables) and compact activity lines.
 *
 * DeliverableTracker. Takes a quick look at the chat's folder when a turn starts, then reports every file the turn created or changed as a
 * `deliverable` frame: { path (inside the folder), abs, name, type, status, size, lines, added, removed }. A file the agent wrote with its
 * file tools is reported as soon as the tool is done; files made some other way (a script, a build) are found by a sweep when the turn ends.
 *
 * ActivityGroups. Consecutive tool cards between two pieces of assistant text are one group, shown by the page as one compact line the way
 * Claude does ("Ran 2 commands, read a file +225 -0"). The group frame carries the counts per kind, the files and the diff stat, and is
 * sent again as it grows (same id) and closed when text follows.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, extname, join, relative, resolve, sep } from 'node:path';

const SKIP_DIR = /^(?:\.|node_modules$|__pycache__$|\$)/;
const MAX_FILES = 4000, MAX_TEXT = 200_000, MAX_TOTAL = 24_000_000, BUDGET_MS = 1500;

const TYPES = {
  html: ['html', 'htm'], markdown: ['md', 'markdown'], pdf: ['pdf'], csv: ['csv', 'tsv'],
  image: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'ico'],
  code: ['js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx', 'py', 'rs', 'go', 'java', 'c', 'h', 'cpp', 'cs', 'sh', 'ps1', 'bat', 'css', 'json', 'yaml', 'yml', 'toml', 'xml', 'sql', 'rb', 'php', 'swift', 'kt'],
  text: ['txt', 'log', 'ini', 'cfg'],
};
export function fileType(path) {
  const ext = extname(String(path)).slice(1).toLowerCase();
  for (const [type, list] of Object.entries(TYPES)) if (list.includes(ext)) return type;
  return 'other';
}
const isTextual = type => ['html', 'markdown', 'csv', 'code', 'text'].includes(type);

const linesOf = text => (text === '' ? [] : text.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n'));
/** Lines added and removed between two versions (a count of lines that are in one and not the other; order is not compared). */
export function lineDelta(before, after) {
  const bag = new Map();
  for (const l of before) bag.set(l, (bag.get(l) ?? 0) + 1);
  let added = 0;
  for (const l of after) { const n = bag.get(l) ?? 0; if (n > 0) bag.set(l, n - 1); else added += 1; }
  let removed = 0; for (const n of bag.values()) removed += n;
  return { added, removed };
}

export class DeliverableTracker {
  constructor(root, { now = Date.now } = {}) {
    this.root = root ? resolve(root) : null;
    this.now = now;
    this.before = new Map();   // rel path -> { size, mtimeMs, lines|null }
    this.sent = new Map();     // rel path -> signature of what was last reported
    this.truncated = false;
  }

  #walk(fn) {
    if (!this.root) return;
    const t0 = this.now(); let count = 0;
    const walk = dir => {
      let entries; try { entries = readdirSync(dir, { withFileTypes: true }); } catch { return; }
      for (const e of entries) {
        if (count >= MAX_FILES || this.now() - t0 > BUDGET_MS) { this.truncated = true; return; }
        const full = join(dir, e.name);
        if (e.isDirectory()) { if (!SKIP_DIR.test(e.name)) walk(full); }
        else if (e.isFile()) { count += 1; fn(full); }
      }
    };
    walk(this.root);
  }

  #read(full, size, type) {
    if (!isTextual(type) || size > MAX_TEXT || this.bytes > MAX_TOTAL) return null;
    try { const buf = readFileSync(full); if (buf.subarray(0, 2048).includes(0)) return null; this.bytes += buf.length; return linesOf(buf.toString('utf8')); } catch { return null; }
  }

  /** Remember what the folder holds now. */
  snapshot() {
    this.before.clear(); this.bytes = 0;
    this.#walk(full => {
      let st; try { st = statSync(full); } catch { return; }
      const rel = relative(this.root, full).split(sep).join('/');
      this.before.set(rel, { size: st.size, mtimeMs: st.mtimeMs, lines: this.#read(full, st.size, fileType(rel)) });
    });
  }

  /** The deliverable for one path now, compared with the snapshot (and the last report), or null when nothing changed. */
  check(rawPath) {
    if (!this.root) return null;
    const full = resolve(this.root, String(rawPath ?? ''));
    const rel = relative(this.root, full).split(sep).join('/');
    if (!rel || rel.startsWith('..') || rel.split('/').some(seg => SKIP_DIR.test(seg))) return null;
    let st; try { st = statSync(full); } catch { return null; }
    if (!st.isFile()) return null;
    const old = this.before.get(rel);
    if (old && old.size === st.size && old.mtimeMs === st.mtimeMs) return null;
    const type = fileType(rel);
    this.bytes = this.bytes ?? 0;
    const lines = this.#read(full, st.size, type);
    let added = null, removed = null;
    if (lines) { const d = lineDelta(old?.lines ?? [], lines); added = d.added; removed = d.removed; if (old && !old.lines) { added = null; removed = null; } }
    const sig = `${st.size}:${st.mtimeMs}`;
    if (this.sent.get(rel) === sig) return null;
    this.sent.set(rel, sig);
    // the next change to this file is measured against what was just reported
    this.before.set(rel, { size: st.size, mtimeMs: st.mtimeMs, lines });
    return { type: 'deliverable', path: rel, abs: full, name: basename(rel), kind: type, status: old ? 'changed' : 'created', size: st.size, lines: lines ? lines.length : null, added, removed };
  }

  /** Files the tools did not announce (a script wrote them): everything new or changed since the snapshot and not yet reported. */
  sweep() {
    const out = [];
    this.#walk(full => { const d = this.check(relative(this.root, full)); if (d) out.push(d); });
    return out;
  }
}

const KIND_OF = name => {
  if (name === 'pwsh' || name === 'bash') return 'command';
  if (name === 'read' || name === 'read_image') return 'read';
  if (name === 'write' || name === 'edit') return 'edit';
  if (name === 'glob' || name === 'grep') return 'search';
  if (name === 'web_fetch' || name === 'web_search') return 'web';
  if (String(name).startsWith('mcp__browser__')) return 'browser';
  if (name === 'subagent') return 'helper';
  return 'other';
};
const NOUN = { command: ['command', 'commands'], read: ['file', 'files'], edit: ['file', 'files'], search: ['search', 'searches'], web: ['web page', 'web pages'], browser: ['browser step', 'browser steps'], helper: ['helper', 'helpers'], other: ['tool', 'tools'] };
const VERB = { command: ['Ran', 'ran'], read: ['Read', 'read'], edit: ['Edited', 'edited'], search: ['Searched', 'searched'], web: ['Looked up', 'looked up'], browser: ['Used the browser for', 'used the browser for'], helper: ['Started', 'started'], other: ['Used', 'used'] };
const ORDER = ['command', 'read', 'edit', 'search', 'web', 'browser', 'helper', 'other'];
const count = (n, [one, many]) => `${n === 1 ? 'a' : n} ${n === 1 ? one : many}`;

/** "Ran 2 commands, read a file +225 -0" */
export function groupLine(counts, added, removed) {
  const parts = ORDER.filter(k => counts[k] > 0);
  if (!parts.length) return '';
  const text = parts.map((k, i) => `${VERB[k][i === 0 ? 0 : 1]} ${count(counts[k], NOUN[k])}`).join(', ');
  return (added || removed) ? `${text} +${added ?? 0} −${removed ?? 0}` : text;
}

export class ActivityGroups {
  constructor({ emit, tracker = null }) { this.emit = emit; this.tracker = tracker; this.seq = 0; this.current = null; this.seen = new Set(); }
  #open() { if (!this.current) this.current = { id: `g${++this.seq}`, counts: {}, files: new Map(), added: 0, removed: 0, tools: 0 }; return this.current; }
  #send(closed) {
    const g = this.current; if (!g) return;
    const files = [...g.files.values()];
    this.emit({ type: 'activity-group', id: g.id, closed, counts: { ...g.counts }, files, added: g.added, removed: g.removed, tools: g.tools, line: groupLine(g.counts, g.added, g.removed) });
  }
  /** A tool card frame (every state change). Counted once, when it is done or has failed. */
  card(tool, deliverable = null) {
    if (!tool || !['done', 'error', 'denied'].includes(tool.state)) return;
    if (this.seen.has(tool.id)) { if (deliverable) this.#file(deliverable, true); return; }
    this.seen.add(tool.id);
    if (tool.state === 'denied') return;
    const g = this.#open(); const kind = KIND_OF(tool.name);
    g.counts[kind] = (g.counts[kind] ?? 0) + 1; g.tools += 1;
    if (deliverable) this.#file(deliverable, false);
    this.#send(false);
  }
  #file(d, resend) {
    const g = this.#open(); const prev = g.files.get(d.path);
    if (prev) { g.added += (d.added ?? 0) - (prev.added ?? 0); g.removed += (d.removed ?? 0) - (prev.removed ?? 0); } else { g.added += d.added ?? 0; g.removed += d.removed ?? 0; }
    g.files.set(d.path, { path: d.path, name: d.name, kind: d.kind, status: d.status, added: d.added, removed: d.removed });
    if (resend) this.#send(false);
  }
  /** A file a script made: part of the group it was made in, without counting a tool. */
  addFile(d) { this.#file(d, true); }
  /** Assistant text ends the group. */
  text() { if (this.current) { this.#send(true); this.current = null; } }
  end() { this.text(); }
}
