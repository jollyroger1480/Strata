#!/usr/bin/env node
/**
 * Headless UI flows of the installed app against a packaged host with a FAKE installer and a FAKE engine (nothing is
 * downloaded, no real engine runs; the real Control starts the fake engine through the owner runtime):
 *
 *   node apps/desktop/scripts/ui-flows.mjs [--control <staged control>] [--node <node.exe>] [--shots <dir>] [--only download,models,tune,fail,theme]
 *
 *   download  the onboarding download: ONE combined bar over both files, an ETA in minutes left of Stop, Finish later closes the
 *             pop-up with the download still running, the Models screen shows the same progress, Stop stops it
 *   models    an installed model is the one Models selects and loads, without a download pop-up
 *   tune      the tuning step offers no picture reading, tunes, writes a RAM budget for "leave 20%", opens chat; reading size
 *   fail      an engine that stops while loading: the error is in words and the pop-up does not scroll
 *   theme     the app follows the system light/dark setting
 * Per-user folders are all under a scratch folder. Exit 1 if a check fails.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, rmdirSync, symlinkSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]]] : acc), []));
const here = dirname(fileURLToPath(import.meta.url));
const repoHost = resolve(here, '../../host');
const staged = resolve(here, '../src-tauri/resources');
const stagedControl = resolve(args.control ?? join(staged, 'control'));
const nodeExe = resolve(args.node ?? join(staged, 'node', process.platform === 'win32' ? 'node.exe' : 'node'));
const work = mkdtempSync(join(tmpdir(), 'strata-ui-flows-'));
const shots = resolve(args.shots ?? join(work, 'shots'));
mkdirSync(shots, { recursive: true });
const only = new Set(String(args.only ?? 'download,models,tune,fail,theme').split(','));
const isWin = process.platform === 'win32';
const { chromium } = await import(pathToFileURL(join(repoHost, 'node_modules/playwright-core/index.mjs')).href).catch(() => import('playwright-core'));
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);
const delay = ms => new Promise(r => setTimeout(r, ms));
const sha = buf => createHash('sha256').update(buf).digest('hex');
const freePort = async () => { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; };
let failed = false;
const check = (ok, label, detail = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? '  ' + detail : ''}`); if (!ok) failed = true; };

// ------------------------------------------------------------------ the app folder: the real Control with a fake installer
const control = join(work, 'control');
mkdirSync(control, { recursive: true });
for (const name of ['python', 'strata_control', 'strata_cli', 'serve']) symlinkSync(join(stagedControl, name), join(control, name), 'junction');
for (const name of ['strata.py', 'VERSION', 'BUNDLE.json', 'requirements.txt']) copyFileSync(join(stagedControl, name), join(control, name));
cpSync(join(stagedControl, 'tools'), join(control, 'tools'), { recursive: true });
const python = isWin ? join(control, 'python', 'python.exe') : join(control, 'python', 'bin', 'python3');

const FAKE_INSTALLER = String.raw`
import json, os, sys, time
from pathlib import Path
STATE = Path(os.environ['FAKE_STATE']); STATE.mkdir(parents=True, exist_ok=True)
RATE = float(os.environ.get('FAKE_RATE_GB_S', '0.5'))
def size(model, gb, ram, mid):
    return {'model': model, 'about': model, 'download_gb': gb, 'ram_needed_gb': ram, 'experts_gb': ram - 8, 'on_this_pc': 'fits', 'installed': False, 'images': True, 'experimental': False, 'id': mid}
CATALOG = {'summary': 'fake', 'families': [{'family': 'qwen', 'title': 'Qwen3.8-Flash-Next', 'about': 'x', 'experimental': False, 'images': True, 'sizes': [
    size('Q2_0', 66.4, 48, 'q2_0'), size('IQ3_XXS', 75.8, 60, 'iq3_xxs'), size('IQ3_S', 83.6, 62, 'iq3_s')]}],
    'contexts': [8192, 32768, 65536], 'recommendation': {'family': 'qwen', 'model': 'IQ3_XXS', 'title': 'Qwen3.8-Flash-Next IQ3_XXS', 'context': 32768}, 'context_note': ''}
HARDWARE = {'ram_gb': 63.1, 'cpu': {'name': 'cpu', 'cores': 12}, 'gpus': [{'index': 0, 'name': 'NVIDIA GeForce RTX 5070', 'vram_gb': 11.9, 'usable': True}]}
TOTAL = 83.6
def job():
    f = STATE / 'job.json'
    if not f.exists(): return None
    j = json.loads(f.read_text())
    if j.get('cancelled'):
        return {'running': False, 'exit_code': 1, 'error': 'stopped', 'result': 'failed'}
    done = min(TOTAL, RATE * (time.time() - j['start']))
    return {'running': True, 'step': '5/7: downloading Qwen3.8-Flash-Next IQ3_S', 'model': 'IQ3_S',
            'download': {'file': 'model files', 'done_gb': round(done, 3), 'total_gb': TOTAL, 'remaining_gb': round(TOTAL - done, 3), 'percent': int(100 * done / TOTAL)}}
def call(name, a):
    if name == 'strata_models': return CATALOG
    if name == 'strata_status': return {'summary': 'fake', 'hardware': HARDWARE, 'recommendation': CATALOG['recommendation'], 'install': job(), 'running': []}
    if name == 'strata_install':
        if a.get('cancel'):
            f = STATE / 'job.json'
            if f.exists(): j = json.loads(f.read_text()); j['cancelled'] = True; f.write_text(json.dumps(j))
            return {'summary': 'stopped', 'install': job()}
        if a.get('confirm'):
            (STATE / 'job.json').write_text(json.dumps({'start': time.time()}))
            return {'summary': 'started', 'plan': {'download_gb': TOTAL}, 'install': job()}
        if not a: return {'summary': 'progress', 'install': job()}
        return {'summary': 'Plan', 'plan': {'title': 'Qwen3.8-Flash-Next IQ3_S', 'download_gb': TOTAL, 'disk_short': None}}
    raise Exception('unknown tool ' + name)
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    m = json.loads(line)
    if 'id' not in m: continue
    method = m.get('method')
    if method == 'initialize': res = {'protocolVersion': '2024-11-05'}
    elif method == 'tools/list': res = {'tools': [{'name': n} for n in ('strata_models', 'strata_status', 'strata_install')]}
    else:
        p = m['params']
        res = {'structuredContent': call(p['name'], p.get('arguments') or {})}
    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': m['id'], 'result': res}) + '\n'); sys.stdout.flush()
`;
writeFileSync(join(control, 'tools', 'strata_mcp.py'), FAKE_INSTALLER);

const FAKE_ENGINE = `import os, sys, json
from pathlib import Path
if os.environ.get('FAKE_ENGINE_FAIL'):
    print('memory guard: 5.1 GiB of RAM free is under the 6 GiB limit (limit 63.1 GiB)', flush=True)
    sys.exit(3)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.strip() == 'QUIT': break
    if line.startswith(('GEN ', 'BGEN ')):
        slot = int(line.split()[1]) if line.startswith('BGEN ') else None
        for ch in os.environ.get('FAKE_REPLY', 'a'):
            print('T %d' % ord(ch), flush=True)
        print('DONE 1 1 1.0 1.0 length 0 0 0', flush=True)
        if slot is not None: print(f'BADM {slot} 0', flush=True)
`;

// one scenario = one fresh host with its own per-user folders
async function scenario(name, { installed = false, env = {} } = {}, body) {
  if (!only.has(name)) return;
  console.log(`--- ${name}`);
  const dir = join(work, name);
  const dirs = { home: join(dir, 'home'), appdata: join(dir, 'appdata'), local: join(dir, 'local'), components: join(dir, 'local', 'Strata', 'components'), install: join(dir, 'local', 'Strata', 'install'), state: join(dir, 'fake-state'), profiles: join(dir, 'profiles'), runtime: join(dir, 'runtime') };
  for (const d of [dirs.home, dirs.appdata, dirs.local, dirs.components, dirs.install, dirs.state, join(dir, 'models')]) mkdirSync(d, { recursive: true });
  const baseEnv = { ...process.env, STRATA_COMPONENTS_DIR: dirs.components, STRATA_PROFILE_DIR: dirs.profiles, STRATA_RUNTIME_DIR: dirs.runtime, STRATA_HOME: dirs.home, STRATA_MODELS_DIR: join(dir, 'models'), APPDATA: dirs.appdata, LOCALAPPDATA: dirs.local, XDG_CONFIG_HOME: join(dir, 'xdg'), XDG_DATA_HOME: join(dir, 'xdgdata'), PYTHONDONTWRITEBYTECODE: '1', STRATA_ENGINE_BACKEND: 'cuda', FAKE_STATE: dirs.state, ...env };
  delete baseEnv.STRATA_API_KEY; delete baseEnv.STRATA_INSTALL_DIR;
  if (installed) {
    const platform = isWin ? 'windows-x86_64' : 'linux-x86_64';
    const folder = join(dirs.components, 'engine', `0.1.45-${platform}-cuda`);
    mkdirSync(folder, { recursive: true });
    writeFileSync(join(folder, 'engine_main.py'), FAKE_ENGINE);
    const entry = isWin ? 'engine.cmd' : 'engine';
    writeFileSync(join(folder, entry), isWin ? `@echo off\r\n"${python}" "%~dp0engine_main.py" %*\r\n` : `#!/bin/sh\nexec "${python}" "$(dirname "$0")/engine_main.py" "$@"\n`);
    const files = [entry, 'engine_main.py'].map(p => { const b = readFileSync(join(folder, p)); return { path: p, sha256: sha(b), bytes: b.length }; });
    writeFileSync(join(folder, 'engine-component.json'), JSON.stringify({ schema_version: 1, id: 'strata-engine', version: '0.1.45', platform, backend: 'cuda', entrypoint: entry, files }));
    const tok = join(dirs.install, 'tokenizer');
    mkdirSync(tok, { recursive: true });
    const script = `import json, sys\nfrom tools.strata_tokenizer import BYTE_TO_UNICODE\ntok = sys.argv[1]\ntokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ['ab', '<|im_start|>', '<|im_end|>', '<|endoftext|>']\njson.dump({t: i for i, t in enumerate(tokens)}, open(tok + '/vocab.json', 'w'))\nopen(tok + '/merges.txt', 'w').write('a b')\njson.dump([0] * 257 + [3, 3, 3], open(tok + '/token_type.json', 'w'))\n`;
    const made = spawnSync(python, ['-c', script, tok], { cwd: control, env: { ...baseEnv, PYTHONNOUSERSITE: '1' }, encoding: 'utf8' });
    if (made.status !== 0) throw new Error('tokenizer fixture: ' + made.stderr);
    writeFileSync(join(dirs.install, 'strata-iq3_s.json'), JSON.stringify({ exe: 'component:strata-engine', backend: 'cuda', tokenizer: tok, args: ['--max-context', '32768', '--pool-workers', '12'], cwd: dir, model_name: 'stub', host: '127.0.0.1', port: 0, kv: 'int8' }));
  }
  const port = await freePort(), base = `http://127.0.0.1:${port}`;
  const host = spawn(nodeExe, [join(repoHost, 'server', 'index.mjs')], { cwd: repoHost, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
    env: { ...baseEnv, STRATA_PORT: String(port), STRATA_DIST: join(repoHost, 'dist'), STRATA_PACKAGED: '1', STRATA_REAL_ENGINE: '1', STRATA_NO_ACQUIRE: '1', STRATA_CONTROL_DIR: control, STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()) } });
  let log = ''; host.stdout.on('data', c => { log += c; }); host.stderr.on('data', c => { log += c; });
  const api = async (path, init) => { const r = await fetch(base + path, init); return { status: r.status, body: await r.json().catch(() => null) }; };
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  try {
    let up = false;
    for (let i = 0; i < 200 && !up; i++) { try { up = (await api('/api/health')).status === 200; } catch { await delay(150); } }
    check(up, `${name}: the packaged host starts`);
    await api('/api/native?refresh=1');
    await body({ base, api, browser, dirs, shot: (page, n) => page.screenshot({ path: join(shots, `${name}-${n}.png`) }), log: () => log });
  } catch (error) {
    check(false, `${name}: the flow`, String(error?.stack ?? error).slice(0, 700) + '\n' + log.slice(-600));
  } finally {
    await browser.close().catch(() => {});
    host.kill(); await new Promise(r => host.once('exit', r));
  }
}
const newPage = async (browser, base, options = {}) => {
  const context = await browser.newContext({ viewport: { width: 1400, height: 950 }, ...options });
  const page = await context.newPage();
  const problems = [];
  page.on('pageerror', e => problems.push(e.message));
  page.problems = problems;
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  return page;
};
const clickText = async (page, re, timeout = 30000) => { const b = page.getByRole('button', { name: re }).first(); await b.waitFor({ timeout }); await b.click(); };
const noScroll = async page => page.evaluate(() => {
  const root = document.querySelector('.guided-onboarding');
  const bad = [];
  for (let el = document.querySelector('.onboard-tune-stage, .onboard-prepare-stage, .onboard-model-stage') ?? root; el && el !== document.body; el = el.parentElement) {
    const style = getComputedStyle(el);
    if (/(auto|scroll)/.test(style.overflowY) && el.scrollHeight > el.clientHeight + 1) bad.push(`${el.className || el.tagName}: ${el.scrollHeight} > ${el.clientHeight}`);
  }
  return bad;
});

// ------------------------------------------------------------------------------------------------ download
await scenario('download', {}, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /get started/i);
  await page.getByText(/Choose a version/i).waitFor({ timeout: 30000 });
  await page.locator('[role=radio]').filter({ hasText: 'IQ3_S' }).first().click();
  await clickText(page, /^Continue/);
  await clickText(page, /Download model/i, 60000);
  await page.locator('.onboard-progress progress').waitFor({ timeout: 30000 });
  // ONE bar: sample the percent while it runs, it only goes up
  const samples = [];
  let eta = '';
  for (let i = 0; i < 40 && !eta; i++) {
    const state = (await api('/api/native')).body;
    const d = state.jobs.at(-1)?.result?.install?.download;
    if (d) samples.push(d.done_gb);
    eta = (await page.locator('.onboard-progress-actions span').first().textContent({ timeout: 1000 }).catch(() => '')) ?? '';
    await delay(1000);
  }
  await shot(page, '1-wizard-progress');
  check(/About \d+ (min left|h)/.test(eta), 'the wizard shows about how many minutes are left', eta.trim());
  const order = await page.evaluate(() => { const a = document.querySelector('.onboard-progress-actions'); const kids = a ? [...a.children].map(k => k.tagName + ':' + Math.round(k.getBoundingClientRect().left)) : []; return kids; });
  check(order.length === 2 && order[0].startsWith('SPAN') && order[1].startsWith('BUTTON') && Number(order[0].split(':')[1]) < Number(order[1].split(':')[1]), 'the time left sits left of Stop', order.join(' '));
  check(samples.length >= 3 && samples.every((v, i) => i === 0 || v >= samples[i - 1]), 'one combined count that only goes up', samples.map(v => v.toFixed(1)).join(', '));
  const meta = await page.locator('.onboard-progress-meta span').first().textContent();
  check(/GB/.test(meta ?? '') && /83\.6/.test(meta ?? ''), 'the bar says done / total over the whole model', meta);
  // close the pop-up while it downloads
  await clickText(page, /Finish later/i);
  await page.waitForURL(/#\/models/, { timeout: 15000 }).catch(() => {});
  await page.getByText(/Preparing your model/i).waitFor({ timeout: 15000 }).catch(() => {});
  await delay(1500);
  await shot(page, '2-models-progress');
  await page.locator('.native-progress').first().screenshot({ path: join(shots, 'download-3-models-progress-card.png') }).catch(() => {});
  check(await page.locator('.guided-onboarding').count() === 0, 'the setup pop-up closed');
  check(await page.getByText(/Preparing your model/i).count() > 0, 'the Models screen shows the running download');
  const still = (await api('/api/native')).body.jobs.at(-1);
  check(still?.state === 'running', 'the download keeps running in the background', still?.state);
  const modelsEta = await page.locator('.native-progress .onboard-progress-actions span').first().textContent({ timeout: 15000 }).catch(() => '');
  check(/About \d+ (min left|h)/.test(modelsEta ?? ''), 'the Models screen shows the time left too', (modelsEta ?? '').trim());
  await clickText(page, /^Stop$/);
  await delay(2500);
  check((await api('/api/native')).body.jobs.at(-1)?.state !== 'running', 'Stop stops it');
  check(page.problems.length === 0, 'no page error', page.problems.join('; '));
});

// ------------------------------------------------------------------------------------------------ models
await scenario('models', { installed: true }, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /Go straight to the app/i);
  await page.waitForURL(/#\/models/, { timeout: 15000 }).catch(() => {});
  await page.getByText(/Conversation context/i).first().waitFor({ timeout: 30000 });
  await delay(1500);
  await shot(page, '1-models');
  await page.locator('.native-toolbar-summary').first().waitFor({ timeout: 20000 }).catch(async () => { await shot(page, '0-models-missing-summary'); console.log('info  page text:', JSON.stringify((await page.locator('body').innerText()).slice(0, 500))); });
  const summary = await page.locator('.native-toolbar-summary').first().textContent({ timeout: 5000 }).catch(() => '');
  check(/IQ3_S|High quality/.test(summary ?? ''), 'Models selects the model that is on this PC, not the recommendation', (summary ?? '').replace(/\s+/g, ' ').trim());
  check(!/IQ3_XXS|Balanced/.test(summary ?? ''), 'the recommendation (IQ3_XXS) is not what is selected');
  const load = page.getByRole('button', { name: /^Load model$/ }).first();
  await load.click();
  await delay(1200);
  check(await page.getByRole('dialog').filter({ hasText: /Download and prepare/i }).count() === 0, 'loading an installed model opens no download pop-up');
  let ready = false;
  for (let i = 0; i < 90 && !ready; i++) { const s = (await api('/api/native')).body; ready = s.instances.some(x => x.status === 'ready'); const err = s.instances.find(x => x.status === 'error'); if (err) throw new Error('load failed: ' + err.error); await delay(1000); }
  if (!ready) { const st = (await api('/api/native')).body; console.log('info  jobs', JSON.stringify(st.jobs.map(j => [j.kind, j.state, j.error ?? j.summary])), 'instances', JSON.stringify(st.instances.map(i => [i.status, i.error]))); await shot(page, '2-models-after-click'); }
  check(ready, 'the model loads (a fake engine under the real Control)');
  check(page.problems.length === 0, 'no page error', page.problems.join('; '));
});

// ------------------------------------------------------------------------------------------------ tune + reading size
await scenario('tune', { installed: true, env: { FAKE_REPLY: ['I am thinking about it.', '</think>', '', 'Here is a list:', '', '- one', '- two', '', '## Heading', '', 'Some `code` and **bold** text.', '', '| a | b |', '|---|---|', '| 1 | 2 |', '', '```', 'code line', '```', ''].join(String.fromCharCode(10)) } }, async ({ base, api, browser, dirs, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /get started/i);
  await page.getByText(/Choose a version/i).waitFor({ timeout: 30000 });
  await page.locator('[role=radio]').filter({ hasText: 'IQ3_S' }).first().click();
  await clickText(page, /^Continue/);
  await page.getByText(/Leave 20% for my work/i).waitFor({ timeout: 30000 });
  await delay(800);
  await shot(page, '1-tune-step');
  check(await page.getByText(/Read pictures when this model loads/i).count() === 0, 'the tuning step offers no picture reading for an app install');
  await clickText(page, /Tune & start chatting/i);
  let done = false, job = null;
  for (let i = 0; i < 120 && !done; i++) { job = (await api('/api/native')).body.jobs.findLast?.(j => j.kind === 'tune') ?? null; done = !!job && job.state !== 'running'; await delay(1000); }
  check(job?.state === 'done', 'tuning finishes without a vision error', job?.error ?? job?.state);
  const cfg = JSON.parse(readFileSync(join(dirs.install, 'strata-iq3_s.json'), 'utf8'));
  check(cfg.args.includes('--resident-budget-gib'), '"leave 20%" wrote a RAM budget for the experts', cfg.args.join(' '));
  const gib = Number(cfg.args[cfg.args.indexOf('--resident-budget-gib') + 1]);
  check(gib + 10 <= 63.1 * 0.8, 'the budget keeps the model within 80% of the PC RAM', `${gib} GiB`);
  // chat + reading size
  await page.getByRole('textbox', { name: /Message Strata/i }).waitFor({ timeout: 60000 });
  await page.getByRole('textbox', { name: /Message Strata/i }).fill('hello');
  await page.keyboard.press('Enter');
  await page.locator('.st-msg--assistant .st-bubble').first().waitFor({ timeout: 60000 });
  await page.locator('.st-msg--assistant .st-bubble p').first().waitFor({ timeout: 60000 }).catch(async () => console.log('info  reply so far:', JSON.stringify(await page.locator('.st-msg--assistant').first().innerText()).slice(0, 300)));
  await page.waitForFunction(() => !document.querySelector('.st-msg--assistant .caret'), null, { timeout: 90000 }).catch(() => {});
  const sizes = [];
  for (const px of [12, 16, 22]) {
    await page.evaluate(v => { const k = 'strata.prefs.v2'; const cur = JSON.parse(localStorage.getItem(k) || '{}'); localStorage.setItem(k, JSON.stringify({ ...cur, contentFontSize: v })); }, px);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.locator('.st-msg--assistant .st-bubble').first().waitFor({ timeout: 30000 });
    sizes.push(await page.evaluate(() => { const b = document.querySelector('.st-msg--assistant .st-bubble'); const t = b.querySelector('p, .md-li, span') ?? b; const u = document.querySelector('.st-msg--user .st-bubble'); const all = {}; for (const el of b.querySelectorAll('*')) { if (!el.childNodes.length || ![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) continue; all[el.tagName.toLowerCase() + (el.className ? '.' + String(el.className).split(' ')[0] : '')] = parseFloat(getComputedStyle(el).fontSize); } const extra = {}; for (const [k, q] of Object.entries({ reasoning: '.st-msg--assistant .st-collapse__body', codeblock: '.st-msg--assistant .st-code pre', codehead: '.st-msg--assistant .st-code__head' })) { const e = document.querySelector(q); if (e) extra[k] = parseFloat(getComputedStyle(e).fontSize); } Object.assign(all, extra); const chain = []; for (let e = b.querySelector('p'); e && e !== b; e = e.parentElement) chain.push(e.tagName + '.' + String(e.className).split(' ')[0] + '=' + getComputedStyle(e).fontSize); all.chain = chain; globalThis.__all = all; return [parseFloat(getComputedStyle(b).fontSize), parseFloat(getComputedStyle(t).fontSize), parseFloat(getComputedStyle(u).fontSize), all]; }));
  }
  console.log('info  chain', JSON.stringify(await page.evaluate(() => { const el = document.querySelector('.st-msg--assistant .st-bubble p'); const out = []; for (let e = el; e && !e.classList.contains('st-msg'); e = e.parentElement) out.push(e.tagName + '.' + String(e.className).split(' ')[0] + '=' + getComputedStyle(e).fontSize); return out; })));
  await shot(page, '2-chat-22px');
  console.log('info  sizes', JSON.stringify(sizes), JSON.stringify(await page.locator('.st-msg--assistant .st-bubble').first().innerHTML()).slice(0, 400));
  check(sizes[0][1] === 12 && sizes[1][1] === 16 && sizes[2][1] === 22 && sizes[0][2] === 12 && sizes[2][2] === 22, 'the reading size changes the text size of the conversation', JSON.stringify(sizes));
  // the slider itself
  await page.evaluate(() => { location.hash = '#/chat?settings=1'; });
  const slider = page.getByLabel('Reading text size');
  if (await slider.count()) {
    await slider.first().focus();
    for (let i = 0; i < 4; i++) await page.keyboard.press('ArrowLeft');
    await delay(500);
    const now = await page.evaluate(() => getComputedStyle(document.querySelector('.st-msg--assistant .st-bubble') ?? document.body).fontSize);
    check(parseFloat(now) === 18, 'moving the slider changes the conversation text at once', now);
    const box = await slider.first().boundingBox();
    await page.mouse.click(box.x + box.width * 0.9, box.y + box.height / 2);
    await delay(400);
    const dragged = await page.evaluate(() => [getComputedStyle(document.querySelector('.st-msg--assistant .st-bubble') ?? document.body).fontSize, getComputedStyle(document.documentElement).getPropertyValue('--st-content-fs'), JSON.parse(localStorage.getItem('strata.prefs.v2') || '{}').contentFontSize]);
    check(parseFloat(dragged[0]) >= 20 && parseFloat(dragged[2]) >= 20, 'clicking the slider with the mouse changes it too, and it is saved', dragged.join(' '));
  } else console.log('info  the settings slider was not reachable by hash here; the stored size was checked above');
  check(page.problems.length === 0, 'no page error', page.problems.join('; '));
});

// ------------------------------------------------------------------------------------------------ fail
await scenario('fail', { installed: true, env: { FAKE_ENGINE_FAIL: '1' } }, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /get started/i);
  await page.getByText(/Choose a version/i).waitFor({ timeout: 30000 });
  await page.locator('[role=radio]').filter({ hasText: 'IQ3_S' }).first().click();
  await clickText(page, /^Continue/);
  await clickText(page, /Tune & start chatting/i, 30000);
  await page.getByText(/ran low on memory/i).first().waitFor({ timeout: 120000 });
  await delay(1000);
  await shot(page, '1-tune-error');
  const text = await page.locator('.guided-onboarding').innerText();
  check(/ran low on memory/.test(text), 'the error says in words that the PC ran low on memory');
  check(/Leave 20% for my work/.test(text) && /low-RAM/.test(text), 'and what to try');
  check(!/server_exited|memory guard/.test(text), 'no code or log line in the main text');
  check(!/vision files/i.test(text), 'no picture-reading error');
  const bad = await noScroll(page);
  check(bad.length === 0, 'the tuning pop-up does not scroll when an error shows', bad.join('; '));
  await page.locator('.error-notice summary').first().click();
  await delay(500);
  await shot(page, '2-tune-error-details');
  const details = await page.locator('.error-notice pre').first().innerText();
  check(/server_exited/.test(details) && /exited before/.test(details), 'Details holds the code and the reason the server logged', details.replace(/\s+/g, ' ').slice(0, 120));
  const badOpen = await noScroll(page);
  check(badOpen.length === 0, 'and the pop-up still does not scroll with Details open', badOpen.join('; '));
  await page.setViewportSize({ width: 1280, height: 700 });
  await delay(600);
  const small = await noScroll(page);
  check(small.length === 0, 'nor in a small window (1280x700), Details open', small.join('; '));
  await shot(page, '3-tune-error-small-window');
  void api;
});

// ------------------------------------------------------------------------------------------------ theme
await scenario('theme', {}, async ({ base, browser, shot }) => {
  for (const scheme of ['dark', 'light']) {
    const page = await newPage(browser, base, { colorScheme: scheme });
    await delay(1500);
    const theme = await page.evaluate(() => document.documentElement.dataset.theme);
    check(theme === scheme, `a fresh app follows the system (${scheme})`, theme);
    await shot(page, `1-${scheme}`);
    await page.context().close();
  }
});

// the junctions into the staged Control go first (rmdir on a junction removes only the link)
for (const name of ['python', 'strata_control', 'strata_cli', 'serve']) { try { rmdirSync(join(control, name)); } catch { /* already gone */ } }
try { rmSync(work, { recursive: true, force: true, maxRetries: 3 }); } catch (e) { console.log('info  scratch folder left behind:', work); }
console.log(`shots in ${shots}`);
process.exit(failed ? 1 : 0);
