/** Live status of a running turn: the sentences, the throttling, and what the projector says as the model moves from thinking to writing a file. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { StatusReporter, toolStatusText, TokenCount } from '../server/agent/status.mjs';
import { TurnProjector } from '../server/agent/events.mjs';

test('a tool is described by what it does to what, in the person\'s words', () => {
  assert.equal(toolStatusText('write', { file_path: 'C:\\work\\pagoda.html' }), 'Writing pagoda.html…');
  assert.equal(toolStatusText('write', { file_path: '/a/b/pagoda.html' }, { bytes: 12000 }), 'Writing pagoda.html… (12 KB)');
  assert.equal(toolStatusText('write', {}), 'Writing a file…');
  assert.equal(toolStatusText('edit', { path: 'notes.md' }), 'Changing notes.md…');
  assert.equal(toolStatusText('pwsh', { command: 'Get-ChildItem' }), 'Running a command…');
  assert.equal(toolStatusText('web_search', {}), 'Searching the web…');
  assert.equal(toolStatusText('web_fetch', { url: 'https://example.com/a' }), 'Opening example.com…');
  assert.equal(toolStatusText('mcp__browser__browser_navigate', { url: 'https://example.com' }), 'Opening https://example.com…');
  assert.equal(toolStatusText('mcp__strata-commands__x', {}), 'Running your tool…');
  assert.equal(toolStatusText('mcp__other__thing', {}), 'Using a tool…');
});

function rig() {
  let clock = 1000;
  const timers = [];
  const frames = [];
  const reporter = new StatusReporter({
    emit: f => frames.push(f), now: () => clock,
    timers: { set: (fn, ms) => { const t = { fn, at: clock + ms, unref() {} }; timers.push(t); return t; }, clear: t => { const i = timers.indexOf(t); if (i >= 0) timers.splice(i, 1); } },
  });
  const advance = ms => { clock += ms; for (const t of [...timers].sort((a, b) => a.at - b.at)) if (t.at <= clock) { timers.splice(timers.indexOf(t), 1); t.fn(); } };
  return { reporter, frames, advance, now: () => clock };
}

test('status: a change is sent at once, the same words are never sent twice, and a quick burst keeps only the newest', () => {
  const { reporter, frames, advance } = rig();
  reporter.set({ phase: 'starting', code: 'starting', text: 'Starting…' });
  reporter.set({ phase: 'starting', code: 'starting', text: 'Starting…' });
  assert.equal(frames.length, 1); assert.deepEqual([frames[0].type, frames[0].seq, frames[0].phase, frames[0].text], ['status', 1, 'starting', 'Starting…']);
  advance(50);
  reporter.set({ phase: 'waiting', code: 'reading-prompt', text: 'Reading your message…' });   // 50 ms after the last: held
  reporter.set({ phase: 'preparing', code: 'preparing-tool', text: 'Writing a.html…', args: { tool: 'write', target: 'a.html', bytes: 10 } });
  assert.equal(frames.length, 1, 'held back, not dropped');
  advance(300);
  assert.equal(frames.length, 2); assert.equal(frames[1].text, 'Writing a.html…', 'the newest change wins; the one in between was never true for long enough to show');
});

test('status: a counter (reasoning tokens, bytes) is refreshed every 2 s, not on every chunk', () => {
  const { reporter, frames, advance } = rig();
  reporter.set({ phase: 'thinking', code: 'thinking', text: 'Thinking… 10 tokens', tokens: 10, args: { tokens: 10 } });
  for (let i = 1; i <= 15; i++) { advance(100); reporter.set({ phase: 'thinking', code: 'thinking', text: `Thinking… ${10 + i * 10} tokens`, tokens: 10 + i * 10, args: { tokens: 10 + i * 10 } }); }
  assert.ok(frames.length <= 2, `1.5 s of chunks made ${frames.length} frames`);
  advance(600);
  assert.equal(frames.at(-1).tokens, 160, 'the refresh carries the latest count');
  assert.ok(frames.length <= 3);
  assert.ok(frames.every(f => f.phase === 'thinking'));
});

test('status: clear says idle once and then nothing more is sent', () => {
  const { reporter, frames, advance } = rig();
  reporter.set({ phase: 'tool', code: 'tool', text: 'Running a command…' });
  advance(400);
  reporter.clear();
  assert.deepEqual(frames.map(f => f.phase), ['tool', 'idle']);
  reporter.set({ phase: 'tool', code: 'tool', text: 'x' }); advance(5000);
  assert.equal(frames.length, 2);
});

test('projector: thinking counts tokens; composing a file call is "Writing pagoda.html…" while no text arrives; running, result and next step follow', () => {
  const p = new TurnProjector();
  const seen = [];
  p.onStatus = u => seen.push(u);
  const chunk = (c, attemptId = 'a1') => p.consume({ type: 'assistant-stream', frame: { type: 'chunk', attemptId, chunk: c } });
  for (const t of ['Let ', 'me ', 'think ', 'about ', 'this']) chunk({ type: 'reasoning-delta', text: t });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).tokens, seen.at(-1).text], ['thinking', 5, 'Thinking… 5 tokens']);
  chunk({ type: 'text-delta', text: 'I will build the pagoda HTML file now.' });
  assert.equal(seen.at(-1).phase, 'answering');
  chunk({ type: 'tool-call-delta', index: 0, id: 'call-1', name: 'write', argumentsDelta: '{"file_path":"C:\\\\work\\\\pagoda.html","content":"<html>' });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).code, seen.at(-1).text], ['preparing', 'preparing-tool', 'Writing pagoda.html…'], 'the stale sentence is replaced the moment the call starts');
  assert.equal(seen.at(-1).args.target, 'C:\\work\\pagoda.html');
  chunk({ type: 'tool-call-delta', index: 0, id: 'call-1', argumentsDelta: 'x'.repeat(9000) });
  assert.match(seen.at(-1).text, /^Writing pagoda\.html… \(\d+(\.\d)? KB\)$/);
  p.consume({ type: 'event', event: { type: 'tool/call', time: 1, data: { callId: 'call-1', name: 'write', arguments: JSON.stringify({ file_path: 'C:/work/pagoda.html', content: 'x' }) } } });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).text], ['tool', 'Writing pagoda.html…']);
  p.consume({ type: 'event', event: { type: 'tool/result', time: 2, data: { message: { toolCallId: 'call-1', content: [{ type: 'text', text: 'ok' }] } } } });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).code], ['waiting', 'model-working']);
});

test('projector: a tool that asks shows "waiting for your answer", and a helper call is not announced as a tool', () => {
  const p = new TurnProjector();
  const seen = [];
  p.onStatus = u => seen.push(u);
  p.consume({ type: 'event', event: { type: 'tool/call', time: 1, data: { callId: 'c1', name: 'pwsh', arguments: JSON.stringify({ command: 'rm x' }) } } });
  p.consume({ type: 'event', event: { type: 'approval/asked', time: 2, data: { id: 'ap1', callId: 'c1', reason: 'Run a command' } } });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).text], ['approval', 'Waiting for your answer…']);
  p.consume({ type: 'event', event: { type: 'approval/decided', time: 3, data: { id: 'ap1', outcome: 'allowed-once' } } });
  assert.equal(seen.at(-1).phase, 'tool', 'back to the running tool after the answer');
  const before = seen.length;
  p.consume({ type: 'assistant-stream', frame: { type: 'chunk', attemptId: 'a', chunk: { type: 'tool-call-delta', index: 0, id: 'h1', name: 'subagent', argumentsDelta: '{"prompt":"x"}' } } });
  assert.equal(seen.length, before, 'helpers have their own status');
});

test('token count: one per chunk, or characters / 4 when the server batches', () => {
  const a = new TokenCount(); for (const t of ['a', 'b', 'c']) a.add(t); assert.equal(a.tokens, 3);
  const b = new TokenCount(); b.add('x'.repeat(400)); assert.equal(b.tokens, 100);
});
