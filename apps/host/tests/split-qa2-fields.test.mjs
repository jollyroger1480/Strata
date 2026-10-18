// QA report F05: a number typed into Response settings is checked against the field's own range.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';
async function loadTs(path) {
  const { code } = await transform(await readFile(new URL(path, import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}
const { checkWholeNumber, numberProblem } = await loadTs('../src/lib/number-field.ts');

test('Max output tokens: -1 and other numbers outside 0 to 1048576 are refused, whole numbers inside are taken', () => {
  assert.deepEqual(checkWholeNumber('-1', 0, 1048576), { ok: false, problem: 'range' });
  assert.deepEqual(checkWholeNumber('1048577', 0, 1048576), { ok: false, problem: 'range' });
  assert.deepEqual(checkWholeNumber('0', 0, 1048576), { ok: true, value: 0 });
  assert.deepEqual(checkWholeNumber(' 4096 ', 0, 1048576), { ok: true, value: 4096 });
  assert.deepEqual(checkWholeNumber('1048576', 0, 1048576), { ok: true, value: 1048576 });
});

test('Seed accepts -1 (random) and refuses what is not a whole number', () => {
  assert.deepEqual(checkWholeNumber('-1', -1, 2147483647), { ok: true, value: -1 });
  assert.equal(checkWholeNumber('-2', -1, 2147483647).ok, false);
  for (const bad of ['', '  ', '-', '1.5', '1e3', 'abc', '12abc', '99999999999999999999']) assert.equal(checkWholeNumber(bad, -1, 2147483647).ok, false, JSON.stringify(bad));
  assert.deepEqual(checkWholeNumber('', 0, 5), { ok: false, problem: 'empty' });
});

test('the sentence says what is wrong in plain words, with the range written the reader\'s way', () => {
  const fmt = n => n.toLocaleString('en');
  assert.equal(numberProblem(checkWholeNumber('7', 0, 10), 0, 10, fmt), null);
  assert.deepEqual(numberProblem(checkWholeNumber('', 0, 10), 0, 10, fmt), { text: 'Enter a number.', args: [] });
  assert.deepEqual(numberProblem(checkWholeNumber('-1', 0, 1048576), 0, 1048576, fmt), { text: 'Enter a whole number from {0} to {1}.', args: ['0', '1,048,576'] });
});

test('the Response settings dialog uses it: wrong numbers show the sentence and the save stays off', async () => {
  const source = (await readFile(new URL('../src/app/SamplingDrawer.tsx', import.meta.url), 'utf8')).replace(/\r\n/g, '\n');
  assert.match(source, /checkWholeNumber\(text, field\.min, field\.max\)/);
  assert.match(source, /disabled=\{pending \|\| !dirty \|\| invalid\}/);
  assert.match(source, /if \(!dirty \|\| pending \|\| invalid\)/);
  assert.match(source, /aria-invalid=\{problem \? true : undefined\}/);
});
