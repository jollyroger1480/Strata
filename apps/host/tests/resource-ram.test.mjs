import test from 'node:test';
import assert from 'node:assert/strict';
import { resourceSettings } from '../server/native-resources.mjs';

const pc64 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5070', vram_gb: 11.9 }] };
const iq3s = { arenaGb: 50.3 };
const config = { gpu: 0, args: ['--native', 'x.gguf', '--max-context', '32768', '--pool-workers', '12'] };
const flag = (args, name) => args[args.indexOf(name) + 1];

test('"leave 20% for my work" keeps the model within 80% of the RAM: a RAM budget for the experts', () => {
  const work = resourceSettings(config, pc64, 'work', iq3s);
  assert.ok(work.args.includes('--resident-budget-gib'), 'a model whose experts do not fit 80% of 63 GB gets a RAM budget');
  const gib = Number(flag(work.args, '--resident-budget-gib'));
  assert.ok(gib + 10 <= 63 * 0.8, `${gib} GiB plus the 10 GB the rest needs stays inside 80% of the RAM`);
  assert.equal(work.ui_resource_baseline.ramBudget, true);
});

test('full performance takes that budget out again and keeps what the installation had', () => {
  const work = resourceSettings(config, pc64, 'work', iq3s);
  const back = resourceSettings({ ...config, args: work.args, ui_resource_profile: 'work', ui_resource_baseline: work.ui_resource_baseline }, pc64, 'performance', iq3s);
  assert.equal(back.args.includes('--resident-budget-gib'), false);
  assert.equal(flag(back.args, '--pool-workers'), '12');
});

test('a model that fits, or one that already has a RAM mode, is left alone', () => {
  assert.equal(resourceSettings(config, { ...pc64, ram_gb: 128 }, 'work', iq3s).args.includes('--resident-budget-gib'), false);
  assert.equal(resourceSettings(config, pc64, 'work', { arenaGb: 34 }).args.includes('--resident-budget-gib'), false, 'Q2_0 fits 80% of 63 GB with 10 to spare');
  const own = { ...config, args: [...config.args, '--mmap-experts'] };
  assert.equal(resourceSettings(own, pc64, 'work', iq3s).args.includes('--resident-budget-gib'), false);
  assert.equal(resourceSettings(config, pc64, 'work').args.includes('--resident-budget-gib'), false, 'no model size, no change');
});

test('neither profile reserves less VRAM than the engine\'s safe floor (the driver pages GPU memory below it and decoding collapses)', () => {
  const card12 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, vram_gb: 11.9 }] };
  const card24 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, vram_gb: 24 }] };
  for (const profile of ['work', 'performance']) {
    const lowConfig = { gpu: 0, args: ['--max-context', '32768', '--vram-reserve-mib', '700'] };
    assert.ok(resourceSettings(lowConfig, card12, profile).vram_reserve_mib >= 1536, `${profile} on 12 GB`);
    assert.ok(resourceSettings({ gpu: 0, args: [] }, card12, profile).vram_reserve_mib >= 1536, `${profile}, nothing configured`);
    assert.ok(resourceSettings(lowConfig, card24, profile).vram_reserve_mib >= 1024, `${profile} on 24 GB`);
  }
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card12, 'performance').vram_reserve_mib, 1536);
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card24, 'performance').vram_reserve_mib, 1024);
  assert.equal(resourceSettings({ gpu: 0, args: [], vram_reserve_mib: 2437 }, card12, 'performance').vram_reserve_mib, 2437, 'a larger reserve of the installation is kept');
});
