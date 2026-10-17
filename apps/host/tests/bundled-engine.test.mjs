/** The engine inside the installer: installed when missing or older, never over a newer or equal one, never without a matching manifest. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, readdirSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { installBundledEngine, compareVersions } from '../server/bundled-engine.mjs';

const sha = buffer => createHash('sha256').update(buffer).digest('hex');
function makePackage(root, version, { body = 'engine-' + version, backend = 'cuda', platform = 'windows-x86_64', variant = 'cuda13', minDriver = null } = {}) {
  const name = `${version}-${platform}-${backend}-${variant}`;
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  const files = { 'strata.exe': Buffer.from(body), 'strata_kq.dll': Buffer.from('kq-' + body), 'BUILD.json': Buffer.from('{}') };
  for (const [file, data] of Object.entries(files)) writeFileSync(join(dir, file), data);
  writeFileSync(join(dir, 'engine-component.json'), JSON.stringify({
    schema_version: 2, id: 'strata-engine', name: 'strata-engine', version, platform, backend, archs: ['sm_86'], ...(minDriver ? { min_driver: { nvidia: minDriver } } : {}), build_id: 'x', entrypoint: 'strata.exe',
    files: Object.entries(files).map(([path, data]) => ({ path, sha256: sha(data), bytes: data.length })), signature: null }));
  return name;
}
const setup = () => {
  const base = mkdtempSync(join(tmpdir(), 'st-bundled-engine-'));
  const bundle = join(base, 'engine-bundle'), components = join(base, 'components');
  mkdirSync(bundle); mkdirSync(components);
  return { base, bundle, components, engine: join(components, 'engine'), cleanup: () => rmSync(base, { recursive: true, force: true }) };
};

test('versions compare by number, not text', () => {
  assert.equal(compareVersions('0.1.45', '0.1.9'), 1);
  assert.equal(compareVersions('0.1.45', '0.1.45.1'), -1);
  assert.equal(compareVersions('0.1.45', '0.1.45'), 0);
});

test('no engine installed: the bundled one is installed, hash-checked, with nothing left over', async () => {
  const t = setup();
  try {
    const name = makePackage(t.bundle, '0.1.45');
    const result = await installBundledEngine({ bundleDir: t.bundle, components: t.components });
    assert.equal(result.action, 'installed');
    assert.equal(readFileSync(join(t.engine, name, 'strata.exe'), 'utf8'), 'engine-0.1.45');
    assert.ok(existsSync(join(t.engine, name, 'engine-component.json')));
    assert.deepEqual(readdirSync(t.engine), [name], 'no temporary folder remains');
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'kept', 'a second start does nothing');
  } finally { t.cleanup(); }
});

test('an older engine is installed: the bundled one is added beside it', async () => {
  const t = setup();
  try {
    const old = makePackage(t.engine, '0.1.44');
    const name = makePackage(t.bundle, '0.1.45');
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'installed');
    assert.deepEqual(readdirSync(t.engine).sort(), [old, name].sort());
  } finally { t.cleanup(); }
});

test('a newer or equal engine is never overwritten or touched', async () => {
  const t = setup();
  try {
    const newer = makePackage(t.engine, '0.1.46', { body: 'mine' });
    makePackage(t.bundle, '0.1.45');
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'kept');
    assert.deepEqual(readdirSync(t.engine), [newer]);
    assert.equal(readFileSync(join(t.engine, newer, 'strata.exe'), 'utf8'), 'mine');
    // same version, other build: the installed one stays
    const same = makePackage(t.engine, '0.1.45', { body: 'installed-build' });
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'kept');
    assert.equal(readFileSync(join(t.engine, same, 'strata.exe'), 'utf8'), 'installed-build');
  } finally { t.cleanup(); }
});

test('a damaged copy of the same package is put right; an engine for another backend does not count', async () => {
  const t = setup();
  try {
    const name = makePackage(t.bundle, '0.1.45');
    makePackage(t.engine, '0.1.45');
    writeFileSync(join(t.engine, name, 'strata.exe'), 'cut');          // wrong size
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'repaired');
    assert.equal(readFileSync(join(t.engine, name, 'strata.exe'), 'utf8'), 'engine-0.1.45');
    const t2 = setup();
    try {
      makePackage(t2.engine, '0.1.50', { backend: 'cpu' });
      makePackage(t2.bundle, '0.1.45');
      assert.equal((await installBundledEngine({ bundleDir: t2.bundle, components: t2.components })).action, 'installed');
    } finally { t2.cleanup(); }
  } finally { t.cleanup(); }
});

test('a bundle that does not match its manifest is refused and installs nothing', async () => {
  const t = setup();
  try {
    const name = makePackage(t.bundle, '0.1.45');
    writeFileSync(join(t.bundle, name, 'strata_kq.dll'), 'kq-engine-0.1.4X');   // same size, other bytes
    const result = await installBundledEngine({ bundleDir: t.bundle, components: t.components });
    assert.equal(result.action, 'refused');
    assert.equal(existsSync(join(t.engine, name)), false);
    assert.equal((await installBundledEngine({ bundleDir: join(t.base, 'nope'), components: t.components })).action, 'none');
  } finally { t.cleanup(); }
});

test('CUDA 13 and CUDA 12 builds of one version are different engines: both are installed when the driver runs both', async () => {
  const t = setup();
  try {
    const n13 = makePackage(t.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda13', minDriver: '580.65', body: 'thirteen' });
    const n12 = makePackage(t.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda12', minDriver: '575.51', body: 'twelve' });
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components, driver: '580.178.04' })).action, 'installed');
    assert.deepEqual(readdirSync(t.engine).sort(), [n12, n13].sort());
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components, driver: '580.178.04' })).action, 'kept', 'a second start does nothing');
  } finally { t.cleanup(); }
});

test('an older driver gets only the build it can run; a driver below both gets the lowest build so the app can name the driver', async () => {
  const t = setup(), u = setup(), v = setup();
  try {
    const n13 = makePackage(t.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda13', minDriver: '580.65' });
    const n12 = makePackage(t.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda12', minDriver: '575.51' });
    await installBundledEngine({ bundleDir: t.bundle, components: t.components, driver: '575.57.08' });
    assert.deepEqual(readdirSync(t.engine), [n12], 'driver 575 runs CUDA 12 only');
    for (const x of [u]) { makePackage(x.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda13', minDriver: '580.65' }); makePackage(x.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda12', minDriver: '575.51' }); }
    await installBundledEngine({ bundleDir: u.bundle, components: u.components, driver: '550.120' });
    assert.deepEqual(readdirSync(u.engine), [n12], 'driver 550 runs neither: the CUDA 12 build is installed so Control can say 575 is needed');
    // no driver at all (a PC without NVIDIA, or nvidia-smi cannot be read): nothing is held back
    makePackage(v.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda13', minDriver: '580.65' }); makePackage(v.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda12', minDriver: '575.51' });
    await installBundledEngine({ bundleDir: v.bundle, components: v.components, driver: null });
    assert.deepEqual(readdirSync(v.engine).sort(), [n12, n13].sort());
  } finally { t.cleanup(); u.cleanup(); v.cleanup(); }
});
