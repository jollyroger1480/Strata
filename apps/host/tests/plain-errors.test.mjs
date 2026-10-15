import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko'];
const catalogs = Object.fromEntries(await Promise.all(languages.map(async l => [l, JSON.parse(await readFile(`src/locales/${l}.json`, 'utf8'))])));
const built = await build({ stdin: { contents: "export * from './src/lib/plain-errors'; export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { plainFailure, plainFailureText, setTranslationLanguage } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

const cases = [
  ['server_exited', 'memory', /ran low on memory/],
  ['server_exited: memory guard: 5.1 GiB of RAM free is under the 6 GiB limit', 'memory', /ran low on memory/],
  ['Engine stopped (exit 3). Open diagnostics or reload.', 'memory', /ran low on memory/],
  ['Loading took more than five minutes. Check the engine log and free memory.', 'slow-load', /taking too long/],
  ['No engine for this PC was found in C:/x/components: the component folder holds no engine package. The app never downloads an engine while it starts or serves.', 'engine-missing', /engine for this computer isn’t installed yet/],
  ['cannot reach huggingface.co (<urlopen error [Errno 11001] getaddrinfo failed>)', 'network', /couldn’t reach the internet/],
  ['sha256 mismatch for Qwen3.8-Flash-Next-00001-of-00002.gguf', 'integrity', /safety check/],
  ['The agent component (DeepSeek Harness) is not installed.', 'assistant-missing', /coding assistant isn’t installed/],
  ['component-missing', 'assistant-missing', /coding assistant/],
  ['Strata Control did not answer within 60 s (models list).', 'control', /couldn’t read its own components/],
  ['Prepare this model’s vision files before enabling picture reading.', 'picture-reading', /Picture reading isn’t available/],
  ['Traceback (most recent call last):\n  File "x.py", line 3', 'technical', /Something went wrong/],
];

for (const [raw, kind, message] of cases) {
  test(`plain words for: ${raw.slice(0, 60)}`, () => {
    const f = plainFailure(raw);
    assert.equal(f.kind, kind);
    assert.match(f.message, message);
    assert.ok(f.next, 'a next step');
    assert.equal(f.detail, raw.trim(), 'the original text stays for Details');
    assert.doesNotMatch(f.message, /server_exited|exit \d|component-missing|Traceback/);
  });
}

test('an ordinary sentence is left as it is and gets no Details', () => {
  const f = plainFailure('Some model files are missing. Use Download / repair to restore them.');
  assert.equal(f.kind, 'plain');
  assert.equal(f.detail, undefined);
});

test('every plain message and next step is translated in all eight languages', () => {
  const en = catalogs.en;
  const ids = Object.fromEntries(Object.entries(en).map(([id, text]) => [text, id]));
  for (const [raw] of cases) {
    const f = plainFailure(raw);
    for (const text of [f.message, f.next]) {
      const id = ids[text];
      assert.ok(id, `"${text}" is in the catalog`);
      for (const l of languages) if (l !== 'en') assert.notEqual(catalogs[l][id], text, `${l}: "${text}" is translated`);
    }
  }
  setTranslationLanguage('de');
  assert.notEqual(plainFailureText('server_exited').message, plainFailure('server_exited').message);
  setTranslationLanguage('en');
});
