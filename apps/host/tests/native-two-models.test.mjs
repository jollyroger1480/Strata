import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

test('a second model on a card another model uses still loads (recommend, never force); two cards, two models', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const load = async args => { const done = await finishJob(f.control, f.control.operation('load', args).id); assert.equal(done.state, 'done', done.error); return done; };
    await load({ model: 'iq2_xs', gpus: [0], multiple: true });
    await load({ model: 'iq3_s', gpus: [1], multiple: true });
    assert.deepEqual(f.control.state().instances.map(i => [i.model, i.gpus, i.status]), [['iq2_xs', [0], 'ready'], ['iq3_s', [1], 'ready']]);
    await load({ model: 'iq3_s', gpus: [0], multiple: true });   // the same card as the first model: allowed, and said on the card's row
    assert.equal(f.control.state().instances.length, 3);
  } finally { await f.cleanup(); }
});

test('the screens say which card a loaded model uses, prefer a free card for a second model, and show an unload refusal beside its button', async () => {
  const models = await readFile(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  assert.match(models, /tr\('Already used by \{0\}', \[usedBy\(o\.cards\)\.join\(', '\)\]\)/);
  assert.match(models, /multiple \? cardOptions\.options\.find\(o => o\.recommended && !usedBy\(o\.cards\)\.length\)/);
  assert.match(models, /loadedAction\(`instances\/\$\{i\.id\}\/unload`\)/);
  assert.match(models, /loadedError \? <ErrorNotice message=\{loadedError\}/);
  const control = await readFile(new URL('../server/native-control.mjs', import.meta.url), 'utf8');
  assert.match(control, /is already used by \$\{sharedWith\.label\}, so this model did not fit there/);
});

test('usedBy is declared before the card choice that calls it (a const used earlier broke the Models page when Run multiple models was on)', async () => {
  const models = await readFile(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  assert.ok(models.indexOf('const usedBy = ') > 0 && models.indexOf('const usedBy = ') < models.indexOf('const activeCards = '));
});

test('a model that failed to start is not offered in the chat model picker', async () => {
  const picker = await readFile(new URL('../src/components/ChatModelPicker.tsx', import.meta.url), 'utf8');
  assert.equal((picker.match(/native\?\.instances\.filter\(i => i\.status !== 'error'\)\.map/g) ?? []).length, 2);
});
