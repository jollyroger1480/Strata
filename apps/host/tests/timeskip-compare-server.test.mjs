/** Item 4: the Compare view can show pictures side by side, so the host lists them as pictures and serves them safely. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { diffRun, imageTypeOf, runImagePath } from '../server/timeskip.mjs';

function fixture() {
  const dir = mkdtempSync(join(tmpdir(), 'strata-cmp-'));
  const base = join(dir, 'base'), upd = join(dir, 'upd'), outside = join(dir, 'secret.png');
  mkdirSync(join(base, 'img'), { recursive: true }); mkdirSync(join(upd, 'img'), { recursive: true });
  writeFileSync(outside, 'not for you');
  writeFileSync(join(base, 'index.html'), '<h1>Old</h1>'); writeFileSync(join(upd, 'index.html'), '<h1>New</h1>');
  writeFileSync(join(base, 'img', 'logo.png'), Buffer.from([137, 80, 78, 71, 0, 1, 2])); writeFileSync(join(upd, 'img', 'logo.png'), Buffer.from([137, 80, 78, 71, 0, 9, 9, 9]));
  writeFileSync(join(base, 'same.png'), Buffer.from([137, 80, 78, 71, 0])); writeFileSync(join(upd, 'same.png'), Buffer.from([137, 80, 78, 71, 0]));
  writeFileSync(join(upd, 'new.svg'), '<svg xmlns="http://www.w3.org/2000/svg"/>');
  const record = { checkpoints: [{ id: 'cp1', label: 'baseline', path: base }], selected_checkpoint: null, attempts: [{ index: 1, path: upd }] };
  return { dir, record };
}

test('pictures are listed as pictures (changed ones only), text files keep their text', () => {
  const { dir, record } = fixture();
  try {
    const found = diffRun(record);
    const by = Object.fromEntries(found.files.map(f => [f.path, f]));
    assert.equal(by['img/logo.png'].image, true); assert.equal(by['img/logo.png'].status, 'changed'); assert.equal(by['img/logo.png'].original, undefined);
    assert.equal(by['new.svg'].image, true); assert.equal(by['new.svg'].status, 'added');
    assert.equal(by['same.png'], undefined, 'an unchanged picture is not listed');
    assert.equal(by['index.html'].original, '<h1>Old</h1>'); assert.equal(by['index.html'].updated, '<h1>New</h1>');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a picture is served only from inside the run, only as a picture', () => {
  const { dir, record } = fixture();
  try {
    assert.ok(runImagePath(record, 'img/logo.png', 'original')?.endsWith('logo.png'));
    assert.ok(runImagePath(record, 'img/logo.png', 'updated'));
    assert.equal(runImagePath(record, 'img/logo.png', 'other'), null);
    assert.equal(runImagePath(record, '../secret.png', 'original'), null, 'no way out of the folder');
    assert.equal(runImagePath(record, 'img/../../secret.png', 'updated'), null);
    assert.equal(runImagePath(record, 'index.html', 'updated'), null, 'a page is not a picture');
    assert.equal(runImagePath(record, 'missing.png', 'updated'), null);
    assert.equal(imageTypeOf('a/b/C.JPG'), 'image/jpeg'); assert.equal(imageTypeOf('a.txt'), null);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
