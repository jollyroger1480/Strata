import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parsePlan, parsePoints, stepObjective, stepLine, firstSentence, editedSteps, planAsPrompt, PlanStore, planPrompt } from '../server/agent/plan.mjs';

test('a plan is the last numbered list of the reply, with reasons after a dash and indented detail kept', () => {
  const p = parsePlan([
    'I looked at the folder. It is empty.',
    '',
    '## Pagoda page',
    '1. Look at what is already in the folder',
    '2. **Write pagoda.html** with the night sky and the glow — one file, inline styles',
    '   - keep it under 10 KB',
    '3. Open it and check it renders.',
    '',
    'Tell me if you want changes.',
  ].join('\n'), 'build a pagoda');
  assert.equal(p.title, 'Pagoda page');
  assert.deepEqual(p.steps.map(s => s.text), ['Look at what is already in the folder', 'Write pagoda.html with the night sky and the glow', 'Open it and check it renders']);
  assert.equal(p.steps[1].detail, 'one file, inline styles');
  assert.equal(p.steps[2].state, 'pending');
  assert.deepEqual(p.steps.map(s => s.id), ['s1', 's2', 's3']);
});

test('a reply with no list gives no plan, so the page shows the words as they are', () => {
  assert.equal(parsePlan('I would just write the file.', 'x'), null);
  assert.equal(parsePlan('1. only one item', 'x'), null);
  assert.equal(parsePlan('3. a\n4. b', 'x'), null, 'a list must start at 1');
});

test('the last list wins, and a title falls back to the request', () => {
  const p = parsePlan('1. old a\n2. old b\n\nBetter:\n1. new a\n2. new b\n3. new c', 'Make a thing');
  assert.deepEqual(p.steps.map(s => s.text), ['new a', 'new b', 'new c']);
  assert.equal(p.title, 'Make a thing');
});

test('goal points are found in lines or in one sentence, and one point is not a list', () => {
  const a = parsePoints('Build the page: 1. draw the base 2. add the roof 3. add the glow');
  assert.equal(a.title, 'Build the page');
  assert.deepEqual(a.steps.map(s => s.text), ['draw the base', 'add the roof', 'add the glow']);
  const b = parsePoints('Make it work\n1) first\n2) second');
  assert.equal(b.steps.length, 2);
  assert.equal(parsePoints('Fix the 3. test and see'), null);
  assert.equal(parsePoints('1. only'), null);
  assert.equal(parsePoints('make every test pass'), null);
});

test('a step goal names its step, what is done and what is not its job', () => {
  const plan = { title: 'Pagoda', steps: [{ text: 'Look', detail: '' }, { text: 'Write it', detail: 'one file' }, { text: 'Check', detail: '' }] };
  const o = stepObjective(plan, 1);
  assert.match(o, /step 2 of 3: Write it \(one file\)/); assert.match(o, /Already done: 1\. Look/); assert.match(o, /do not start them/);
  assert.ok(!/do not start/.test(stepObjective(plan, 2)));
  assert.equal(stepLine(1, 4, 'Build the roof', 'done'), 'Step 2 of 4: Build the roof — done');
  assert.equal(stepLine(0, 4, 'x', 'failed'), "Step 1 of 4: x — didn't finish");
  assert.equal(firstSentence('## Done\n\nThe roof is built and tiled. Then I checked it.'), 'The roof is built and tiled.');
  assert.equal(firstSentence('   '), null);
});

test('an edit keeps order, drops empty lines and refuses an empty plan', () => {
  const steps = editedSteps(['  one ', '', { text: 'two', detail: 'why' }, { text: '   ' }]);
  assert.deepEqual(steps.map(s => [s.id, s.text, s.detail]), [['s1', 'one', ''], ['s2', 'two', 'why']]);
  assert.throws(() => editedSteps([]), /at least one step/); assert.throws(() => editedSteps('x'), /list of steps/);
  assert.match(planAsPrompt({ title: 'T', steps }), /1\. one\n2\. two \(why\)/);
  assert.match(planPrompt('do x'), /Request: do x/);
});

test('plans survive a restart', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-plans-'));
  try {
    const file = join(dir, 'plans.json');
    const a = new PlanStore(file);
    const plan = a.create('chat-1', { title: 'T', steps: editedSteps(['a', 'b']), request: 'r' });
    a.update('chat-1', p => { p.state = 'approved'; });
    const b = new PlanStore(file);
    assert.equal(b.get('chat-1').id, plan.id); assert.equal(b.get('chat-1').state, 'approved');
    b.clear('chat-1'); assert.equal(new PlanStore(file).get('chat-1'), null);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
