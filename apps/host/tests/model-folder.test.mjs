import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, parse } from 'node:path';
import { alignModels, dataFolderFromPick, isDriveRoot, libraryFolderFromPick, makeFolder, rememberDataFolder } from '../server/model-folder.mjs';
import { nativeFixture } from './helpers/native-fixture.mjs';

const scratch = t => { const dir = realpathSync(mkdtempSync(join(tmpdir(), 'strata-folder-'))); t.after(() => rmSync(dir, { recursive: true, force: true })); return dir; };

test('a whole drive is not filled with models: it becomes Strata (data) and Strata/Models (library)', () => {
  const root = parse(tmpdir()).root;
  assert.equal(isDriveRoot(root), true);
  assert.equal(isDriveRoot(tmpdir()), false);
  assert.deepEqual(dataFolderFromPick(root), { path: join(root, 'Strata'), note: `${root} is a whole drive, so the models go in ${join(root, 'Strata')}.` });
  assert.equal(libraryFolderFromPick(root).path, join(root, 'Strata', 'Models'));
  assert.deepEqual(dataFolderFromPick(tmpdir()), { path: tmpdir(), note: null });
  assert.deepEqual(libraryFolderFromPick(tmpdir()), { path: tmpdir(), note: null });
});

test('a folder that cannot be made is said in words', t => {
  const dir = scratch(t);
  writeFileSync(join(dir, 'a-file'), 'x');
  assert.throws(() => makeFolder(join(dir, 'a-file', 'Strata')), /Strata could not create .*Choose another folder\./);
  assert.equal(makeFolder(join(dir, 'new', 'Strata')), join(dir, 'new', 'Strata'));
  assert.equal(existsSync(join(dir, 'new', 'Strata')), true);
});

test('models and Models are one folder; a disk that cannot hold the link gets a plain models folder', t => {
  const dir = scratch(t), library = join(dir, 'Models');
  mkdirSync(library);
  const refuses = () => { throw Object.assign(new Error('EPERM: operation not permitted, symlink'), { code: 'EPERM' }); };
  const result = alignModels(dir, library, { link: refuses });
  assert.equal(result.linked, false);
  assert.equal(existsSync(join(dir, 'models')), true);
  // where the names are one folder already (Windows, macOS) nothing is made; where a link works it is used
  const other = scratch(t);
  const calls = [];
  assert.equal(alignModels(other, join(other, 'Models'), { link: (...args) => { calls.push(args); } }).linked, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][1], join(other, 'models'));
});

test('the data folder is written into the Control settings file, merged and only when it changes', t => {
  const dir = scratch(t), file = join(dir, 'config', 'app-settings.json');
  assert.equal(rememberDataFolder(file, 'D:\\Strata'), true);
  assert.deepEqual(JSON.parse(readFileSync(file, 'utf8')), { data_dir: 'D:\\Strata' });
  writeFileSync(file, JSON.stringify({ installs: ['C:\\x'], data_dir: 'C:\\old' }));
  assert.equal(rememberDataFolder(file, 'D:\\Strata'), true);
  assert.deepEqual(JSON.parse(readFileSync(file, 'utf8')), { installs: ['C:\\x'], data_dir: 'D:\\Strata' });
  assert.equal(rememberDataFolder(file, 'D:\\Strata'), false);
  writeFileSync(file, '\uFEFF{"data_dir":"E:\\\\Strata","installs":[]}');
  assert.equal(rememberDataFolder(file, 'E:\\Strata'), false);
  writeFileSync(file, 'not json');
  assert.equal(rememberDataFolder(file, 'F:\\Strata'), true);
  assert.equal(rememberDataFolder('', 'F:\\Strata'), false);
});

test('choosing a folder that is gone or cannot be written says so in words', async t => {
  const f = nativeFixture();
  t.after(() => f.cleanup());
  await f.control.refresh();
  assert.throws(() => f.control.setDownloadDirectory(join(f.repo, 'not-there')), /That folder does not exist\. Check the path, or that the drive is connected\./);
  assert.throws(() => f.control.setDownloadDirectory(join(f.repo, 'setup.py')), /Choose a folder, not a file\./);
});

test('Windows: a drive picked as D:\\ becomes D:\\Strata, and the person is not asked for a different folder', { skip: process.platform !== 'win32' }, async t => {
  const dir = scratch(t);
  const letter = 'QRSTUVWXYZ'.split('').find(l => !existsSync(`${l}:\\`));
  assert.ok(letter, 'a free drive letter');
  execFileSync('subst', [`${letter}:`, dir]);
  t.after(() => { try { execFileSync('subst', [`${letter}:`, '/D']); } catch { /* gone */ } });
  const f = nativeFixture();
  t.after(() => f.cleanup());
  await f.control.refresh();
  f.control.setDownloadDirectory(`${letter}:\\`);
  const shown = f.control.state().downloadDirectory;
  assert.equal(shown.toLowerCase(), `${letter}:\\Strata`.toLowerCase());
  assert.equal(existsSync(join(dir, 'Strata')), true);
  assert.equal(readdirNames(dir).includes('Strata'), true);
});

function readdirNames(dir) { return execFileSync('cmd', ['/c', 'dir', '/b', dir]).toString().split(/\r?\n/).filter(Boolean); }

test('Windows: the model library on a whole drive goes to X:\\Strata\\Models', { skip: process.platform !== 'win32' }, t => {
  const dir = scratch(t), home = scratch(t);
  const letter = 'QRSTUVWXYZ'.split('').find(l => !existsSync(`${l}:\\`));
  execFileSync('subst', [`${letter}:`, dir]);
  t.after(() => { try { execFileSync('subst', [`${letter}:`, '/D']); } catch { /* gone */ } });
  const cwd = fileURLToPath(new URL('..', import.meta.url));
  const drive = `${letter}:\\`;
  const run = spawnSync(process.execPath, ['--input-type=module', '-e', `const store=await import('./server/store.mjs');console.log(store.validateModelsDirectory(${JSON.stringify(drive)}));`],
    { cwd, env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models') }, encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  assert.equal(run.stdout.trim().toLowerCase(), `${drive}Strata\\Models`.toLowerCase());
  assert.equal(existsSync(join(dir, 'Strata', 'Models')), true);
});
