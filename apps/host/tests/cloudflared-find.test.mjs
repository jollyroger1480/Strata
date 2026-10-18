import test from 'node:test';
import assert from 'node:assert/strict';
import { chmod, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { connectorCandidates, describeConnector, findConnector, findRunningConnector, MIN_VERSION, parseVersion, runningMessage, skippedMessage, versionAtLeast } from '../server/cloudflared-find.mjs';
import { createConnectorInstaller } from '../server/cloudflared-install.mjs';
import { createCloudflare } from '../server/cloudflare.mjs';

test('version text is read and compared against the minimum', () => {
  assert.deepEqual(parseVersion('cloudflared version 2026.5.0 (built 2026-05-12 )'), [2026, 5, 0]);
  assert.equal(parseVersion('no digits here'), null);
  assert.equal(versionAtLeast('2026.5.0'), true);
  assert.equal(versionAtLeast('2022.1.0'), true);
  assert.equal(versionAtLeast('2021.12.2'), false);
  assert.equal(versionAtLeast(null), true, 'a build with no readable version is not refused');
  assert.equal(MIN_VERSION, '2022.1.0');
});

test('Linux search order: own folder, component, PATH, then the usual places', () => {
  const list = connectorCandidates({ directory: '/h/engine/cloudflare', componentDirectory: '/h/components/cloudflared', platform: 'linux', env: { PATH: '/usr/bin:/opt/x/bin:/usr/bin' }, home: '/home/tester' });
  assert.deepEqual(list.slice(0, 4).map(c => [c.path, c.source]), [
    ['/h/engine/cloudflare/cloudflared', 'strata'], ['/h/components/cloudflared/cloudflared', 'component'],
    ['/usr/bin/cloudflared', 'path'], ['/opt/x/bin/cloudflared', 'path']]);
  const places = list.filter(c => c.source === 'system').map(c => c.path);
  for (const want of ['/usr/local/bin/cloudflared', '/opt/homebrew/bin/cloudflared', '/home/tester/.local/bin/cloudflared']) assert.ok(places.includes(want), want);
  assert.equal(list.filter(c => c.path === '/usr/bin/cloudflared').length, 1, '/usr/bin is listed once (PATH wins)');
});

test('Windows and Mac usual places', () => {
  const win = connectorCandidates({ directory: 'C:\\h\\cf', platform: 'win32', env: { Path: 'C:\\tools', ProgramFiles: 'C:\\Program Files', 'ProgramFiles(x86)': 'C:\\Program Files (x86)', LOCALAPPDATA: 'C:\\Users\\a\\AppData\\Local' }, home: 'C:\\Users\\a' }).map(c => c.path);
  assert.ok(win.includes('C:\\tools\\cloudflared.exe'));
  assert.ok(win.includes('C:\\Program Files\\cloudflared\\cloudflared.exe'));
  assert.ok(win.includes('C:\\Program Files (x86)\\cloudflared\\cloudflared.exe'));
  assert.ok(win.includes('C:\\Users\\a\\AppData\\Local\\Microsoft\\WinGet\\Links\\cloudflared.exe'));
  const mac = connectorCandidates({ directory: '/h/cf', platform: 'darwin', env: { PATH: '' }, home: '/Users/a' }).map(c => c.path);
  assert.ok(mac.includes('/opt/homebrew/bin/cloudflared') && mac.includes('/usr/local/bin/cloudflared'));
});

test('a system cloudflared 2026.5.0 in /usr/bin is used, with its version and where it came from', async () => {
  const there = new Set(['/usr/bin/cloudflared']);
  const result = await findConnector({ directory: '/h/engine/cloudflare', platform: 'linux', env: { PATH: '/usr/bin:/bin' }, home: '/home/t', usable: async p => there.has(p), run: async () => 'cloudflared version 2026.5.0 (built 2026-05-12-1234 UTC)' });
  assert.deepEqual(result.found, { path: '/usr/bin/cloudflared', version: '2026.5.0', source: 'path', label: 'found on your PATH' });
  assert.match(describeConnector(result.found), /cloudflared 2026\.5\.0 · \/usr\/bin\/cloudflared \(found on your PATH\)/);
});

test('Strata’s own copy wins over a system one; an old or broken copy is skipped, with a plain reason', async () => {
  const there = new Set(['/h/cf/cloudflared', '/usr/local/bin/cloudflared']);
  const versions = { '/h/cf/cloudflared': 'cloudflared version 2021.3.0', '/usr/local/bin/cloudflared': 'cloudflared version 2025.8.1' };
  const both = await findConnector({ directory: '/h/cf', platform: 'linux', env: { PATH: '' }, home: '/h', usable: async p => there.has(p), run: async p => versions[p] });
  assert.equal(both.found.path, '/usr/local/bin/cloudflared', 'the too-old own copy is passed over');
  assert.equal(both.skipped[0].reason, 'too-old');
  const onlyOld = await findConnector({ directory: '/h/cf', platform: 'linux', env: { PATH: '' }, home: '/h', usable: async p => p === '/h/cf/cloudflared', run: async () => 'cloudflared version 2021.3.0' });
  assert.equal(onlyOld.found, null);
  assert.match(skippedMessage(onlyOld.skipped), /version 2021\.3\.0, which is too old\. Strata needs 2022\.1\.0 or newer/);
  const broken = await findConnector({ directory: '/h/cf', platform: 'linux', env: { PATH: '' }, home: '/h', usable: async p => p === '/h/cf/cloudflared', run: async () => { throw new Error('exec format error'); } });
  assert.equal(broken.found, null);
  assert.match(skippedMessage(broken.skipped), /did not run as cloudflared/);
  const none = await findConnector({ directory: '/h/cf', platform: 'linux', env: { PATH: '' }, home: '/h', usable: async () => false, run: async () => '' });
  assert.deepEqual(none, { found: null, skipped: [] });
  assert.equal(skippedMessage(none.skipped), null);
});

test('STRATA_CLOUDFLARED picks exactly one file; STRATA_CLOUDFLARED_SEARCH=own keeps to Strata’s folders', async () => {
  const one = connectorCandidates({ directory: '/h/cf', platform: 'linux', env: { PATH: '/usr/bin', STRATA_CLOUDFLARED: '/opt/mine/cloudflared' }, home: '/h' });
  assert.deepEqual(one, [{ path: '/opt/mine/cloudflared', source: 'setting' }]);
  const own = connectorCandidates({ directory: '/h/cf', componentDirectory: '/h/c', platform: 'linux', env: { PATH: '/usr/bin', STRATA_CLOUDFLARED_SEARCH: 'own' }, home: '/h' });
  assert.deepEqual(own.map(c => c.source), ['strata', 'component']);
});

test('a real executable on PATH is found and run (a user-local copy, never a system install)', { skip: process.platform === 'win32' }, async () => {
  const dir = await mkdtemp(join(tmpdir(), 'strata-cf-path-'));
  try {
    const bin = join(dir, 'bin'); await mkdir(bin);
    await writeFile(join(bin, 'cloudflared'), '#!/bin/sh\necho "cloudflared version 2026.5.0 (built 2026-05-12)"\n');
    await chmod(join(bin, 'cloudflared'), 0o755);
    const installer = createConnectorInstaller({ directory: join(dir, 'own'), platform: process.platform, env: { PATH: bin }, home: dir, search: undefined });
    assert.equal(await installer.locate(), join(bin, 'cloudflared'));
    assert.equal(installer.connector().version, '2026.5.0');
    assert.equal(await installer.ensure(), join(bin, 'cloudflared'), 'nothing is fetched when one is found');
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('running detection: our own pid is ignored; a system service is called a service; failing to look is not a block', async () => {
  const run = async (cmd, args) => cmd === 'pgrep' ? '4242\n77\n' : cmd === 'systemctl' ? 'active\n' : '';
  assert.deepEqual(await findRunningConnector({ platform: 'linux', run }), { running: true, kind: 'service', pids: [4242, 77] });
  assert.deepEqual(await findRunningConnector({ platform: 'linux', run, exclude: [4242, 77] }), { running: false, kind: null, pids: [] });
  const user = async cmd => cmd === 'pgrep' ? '9\n' : 'inactive\n';
  assert.equal((await findRunningConnector({ platform: 'linux', run: user })).kind, 'process');
  assert.equal((await findRunningConnector({ platform: 'darwin', run: user })).kind, 'process');
  assert.equal((await findRunningConnector({ platform: 'linux', run: async () => '' })).running, false);
  const win = async cmd => cmd === 'tasklist' ? '"cloudflared.exe","8864","Console","1","28,756 K"\r\n' : 'STATE : 4 RUNNING';
  assert.deepEqual(await findRunningConnector({ platform: 'win32', run: win }), { running: true, kind: 'service', pids: [8864] });
  assert.match(runningMessage({ running: true, kind: 'service' }), /already running on this computer as a system service, so Strata did not start a second one/);
  assert.equal(runningMessage({ running: false }), null);
});

function fakeOrigin() { return { started: 0, async start() { this.started++; }, async stop() {} }; }

test('the tunnel does not start a second connector next to a running one, and says so plainly', async () => {
  const origin = fakeOrigin(); let launched = 0;
  const cf = createCloudflare({ origin, originUrl: 'http://127.0.0.1:1', hasKey: () => true, locate: async () => '/usr/bin/cloudflared',
    connectorInfo: () => ({ path: '/usr/bin/cloudflared', version: '2026.5.0', source: 'path', label: 'found on your PATH' }),
    findRunning: async () => ({ running: true, kind: 'service', pids: [12] }), launch: () => { launched++; throw new Error('must not launch'); } });
  const seen = await cf.inspect();
  assert.equal(seen.installed, true);
  assert.equal(seen.connector.version, '2026.5.0');
  assert.match(seen.connector.summary, /cloudflared 2026\.5\.0 · \/usr\/bin\/cloudflared/);
  assert.equal(seen.external.kind, 'service');
  await assert.rejects(cf.start({ mode: 'quick' }), /already running on this computer as a system service/);
  assert.equal(launched, 0);
  assert.equal(origin.started, 0, 'the private origin is not opened either');
  assert.equal(cf.state().status, 'error');
});

test('with nothing else running the tunnel starts as before and the state names the connector in use', async () => {
  const origin = fakeOrigin(); const spawned = [];
  const fake = { stdout: { on() {} }, stderr: { on() {} }, once() {}, kill() {}, pid: 55 };
  const cf = createCloudflare({ origin, originUrl: 'http://127.0.0.1:1', hasKey: () => true, locate: async () => '/usr/bin/cloudflared',
    connectorInfo: () => ({ path: '/usr/bin/cloudflared', version: '2026.5.0', source: 'path', label: 'found on your PATH' }),
    findRunning: async ({ exclude }) => { assert.ok(Array.isArray(exclude)); return { running: false, kind: null, pids: [] }; },
    launch: (bin, args) => { spawned.push([bin, args]); return fake; }, timeoutMs: 50 });
  await cf.start({ mode: 'quick' });
  assert.equal(spawned[0][0], '/usr/bin/cloudflared');
  assert.equal(cf.state().external, null);
  await cf.stop();
});
