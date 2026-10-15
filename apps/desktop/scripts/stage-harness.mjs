#!/usr/bin/env node
/**
 * Stage the DeepSeek Harness (pinned, MIT) as the installer's `harness/` folder: a pruned PRODUCTION install, built at
 * build time, so nothing is fetched or compiled on the person's PC.
 *
 *   node apps/desktop/scripts/stage-harness.mjs --src <built Harness checkout> [--out apps/desktop/src-tauri/resources/harness]
 *
 * <built Harness checkout> is a clone of the pinned commit after `pnpm install --frozen-lockfile` and `pnpm run build:artifacts`
 * (see phaseA-tests/notes-split-s4a.md, step 1). The script never touches it: it runs `pnpm deploy --legacy --prod` into
 * the output folder, removes what a running Harness does not read (sources, tests, maps, docs, other operating systems'
 * prebuilt binaries, the LibreOffice document converter and the speech engine, which the app does not use), keeps the
 * MIT licence and third-party notices, writes BUNDLE.json, and finally boots the staged copy once as proof.
 *
 * The host finds the result at `<install>/harness` (server/agent/harness-runtime.mjs), or in the component folder.
 */
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';
import { createHash } from 'node:crypto';

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : fallback; };
const src = resolve(opt('src', process.env.STRATA_HARNESS_SRC ?? ''));
const out = resolve(opt('out', join(import.meta.dirname, '..', 'src-tauri', 'resources', 'harness')));
if (!opt('src', process.env.STRATA_HARNESS_SRC) || !existsSync(join(src, 'apps', 'cli', 'lib', 'bin.js'))) {
  console.error('Give --src <a built Harness checkout> (apps/cli/lib/bin.js must exist).');
  process.exit(2);
}
const platform = process.platform;
const run = (cmd, argv, cwd, extra = {}) => {
  const r = spawnSync(cmd, argv, { cwd, stdio: 'inherit', shell: false, ...extra });
  if (r.status !== 0) { console.error(`${cmd} ${argv.join(' ')} failed (${r.status ?? r.error})`); process.exit(1); }
};

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
const pnpm = platform === 'win32' ? ['corepack.cmd', ['pnpm']] : ['corepack', ['pnpm']];
run(pnpm[0], [...pnpm[1], 'deploy', '--legacy', '--prod', '--filter', '@deepseek-ai/dsh', out], src, { shell: platform === 'win32' });

// --- prune ------------------------------------------------------------------------------------------------------------
const OTHER_OS = platform === 'win32' ? /(^|[\\/._-])(darwin|linux|android|freebsd|musl)([\\/._-]|$)/i
  : platform === 'darwin' ? /(^|[\\/._-])(win32|linux|android|freebsd|musl)([\\/._-]|$)/i
    : /(^|[\\/._-])(win32|darwin|android|freebsd)([\\/._-]|$)/i;
/** Whole packages the app never loads: office-document conversion (hundreds of MB) and speech input. */
const DROP_PACKAGES = [/libreoffice-kit/i, /sherpa-onnx/i];
const DROP_DIRS = new Set(['test', 'tests', '__tests__', 'docs', 'doc', 'website', 'benchmarks', 'examples', 'example', '.github']);
const DROP_FILES = /(\.map|\.md|\.markdown|\.pdb|\.ts\.map|\.tsbuildinfo)$/i;
let removed = 0;
function prune(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isSymbolicLink()) continue;
    if (e.isDirectory()) {
      const inLib = p.split(sep).includes('lib');
      if ((DROP_DIRS.has(e.name) && !inLib) || DROP_PACKAGES.some(r => r.test(e.name)) || (OTHER_OS.test(e.name) && /prebuild|bin|native|platform|-(win32|darwin|linux)/i.test(p))) {
        rmSync(p, { recursive: true, force: true }); removed++; continue;
      }
      prune(p);
    } else if (DROP_FILES.test(e.name) && !KEEP_FILES.test(e.name)) {
      rmSync(p, { force: true }); removed++;
    }
  }
}
prune(out);

// --- licence: the MIT text and the third-party notices travel with the files ---------------------------------------------
for (const name of ['LICENSE', 'THIRD_PARTY_NOTICES.md']) {
  if (existsSync(join(src, name))) cpSync(join(src, name), join(out, name));
}
const cli = JSON.parse(readFileSync(join(out, 'package.json'), 'utf8'));
let commit = process.env.STRATA_HARNESS_COMMIT ?? null;
if (!commit) { const r = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: src, encoding: 'utf8' }); if (r.status === 0) commit = r.stdout.trim(); }
function size(dir) { let t = 0, n = 0; for (const e of readdirSync(dir, { withFileTypes: true })) { const p = join(dir, e.name); if (e.isDirectory()) { const s = size(p); t += s.bytes; n += s.files; } else if (e.isFile()) { t += statSync(p).size; n++; } } return { bytes: t, files: n }; }
const total = size(out);
writeFileSync(join(out, 'BUNDLE.json'), JSON.stringify({
  component: 'deepseek-harness', version: cli.version, commit, license: 'MIT', platform, node: process.version,
  bytes: total.bytes, files: total.files, entry: 'apps/cli/lib/bin.js', builtAt: new Date().toISOString(),
  note: 'Pruned production install (pnpm deploy --prod). No package manager runs on the user\'s PC.',
}, null, 2));

// --- proof: the staged copy answers --version and boots the web profile once ----------------------------------------
const bin = join(out, 'apps', 'cli', 'lib', 'bin.js');
const entry = existsSync(bin) ? bin : join(out, 'lib', 'bin.js');
const v = spawnSync(process.execPath, [entry, '--version'], { encoding: 'utf8' });
if (v.status !== 0) { console.error('staged Harness does not start:', v.stderr); process.exit(1); }
console.log(`staged ${cli.version} (${commit?.slice(0, 8) ?? 'no commit'}) in ${out}: ${(total.bytes / 1048576).toFixed(0)} MiB, ${total.files} files, ${removed} pruned items; --version -> ${v.stdout.trim()}`);
console.log(`sha256(BUNDLE.json)=${createHash('sha256').update(readFileSync(join(out, 'BUNDLE.json'))).digest('hex')}`);
