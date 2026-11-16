'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const BIN = path.join(__dirname, '..', 'bin', 'vram-fit.js');
const run = (...args) => {
  const r = spawnSync(process.execPath, [BIN, ...args], { encoding: 'utf8' });
  return { code: r.status, out: r.stdout, err: r.stderr };
};
const json = (...args) => {
  const r = run(...args, '--json');
  assert.equal(r.code, 0, r.err);
  return JSON.parse(r.out);
};

test('--model llama31-70b --quant int4 --context 8192 prints the breakdown and rates date', () => {
  const r = run('--model', 'llama31-70b', '--quant', 'int4', '--context', '8192');
  assert.equal(r.code, 0);
  assert.match(r.out, /Weights\s+39\.38 GB/);
  assert.match(r.out, /KV cache\s+2\.50 GB/);
  assert.match(r.out, /Activations\s+3\.94 GB/);
  assert.match(r.out, /Overhead\s+1\.50 GB/);
  assert.match(r.out, /Total\s+47\.31 GB/);
  assert.match(r.out, /within the last 30 days/);
});

test('--json output has the breakdown and GPU rows', () => {
  const j = json('--model', 'llama31-70b', '--quant', 'int4', '--context', '8192');
  assert.deepEqual(j.estimateGB, { weights: 39.38, kvCache: 2.5, activations: 3.94, overhead: 1.5, total: 47.31 });
  assert.equal(j.kvBytesPerToken, 327680);
  assert.equal(j.ratesChecked, 'within the last 30 days');
  assert.equal(j.cloud, 'secure');
  assert.equal(j.gpus[0].id, 'a40');
  assert.equal(j.gpus[0].status, 'tight');
  j.gpus.forEach((g) => assert.ok([1, 2, 4, 8].includes(g.count)));
});

test('--params with --finetune and the equals form of options', () => {
  const j = json('--params=8', '--quant=fp16', '--context=4096', '--finetune');
  assert.equal(j.input.finetune, true);
  assert.equal(j.input.model, null);
  assert.ok(j.notes.some((n) => /extrapolated/.test(n)));
});

test('community cloud skips GPUs without a community rate', () => {
  const secure = json('--model', 'llama31-8b', '--quant', 'int4', '--cloud', 'secure');
  const community = json('--model', 'llama31-8b', '--quant', 'int4', '--cloud', 'community');
  assert.ok(secure.gpus.some((g) => g.id === 'mi300x'));
  assert.ok(!community.gpus.some((g) => g.id === 'mi300x'));
});

test('--list-models lists models; --json gives an array', () => {
  const r = run('--list-models');
  assert.equal(r.code, 0);
  assert.match(r.out, /llama31-70b/);
  assert.match(r.out, /deepseek-v3/);
  const j = JSON.parse(run('--list-models', '--json').out);
  assert.ok(Array.isArray(j) && j.some((m) => m.id === 'qwen25-14b'));
});

test('--help and --version exit 0', () => {
  const h = run('--help');
  assert.equal(h.code, 0);
  assert.match(h.out, /Usage:/);
  assert.match(h.out, /Exit codes: 0 ok, 2 bad input/);
  const v = run('--version');
  assert.equal(v.code, 0);
  assert.match(v.out, /^\d+\.\d+\.\d+/);
});

test('bad input exits 2 with a message on stderr and nothing on stdout', () => {
  const cases = [
    [],
    ['--params', '0'],
    ['--params', '-5'],
    ['--params', 'NaN'],
    ['--params', 'Infinity'],
    ['--params', 'abc'],
    ['--params', ''],
    ['--params', '1e9'],
    ['--params', '70', '--context', '0'],
    ['--params', '70', '--context', '-4096'],
    ['--params', '70', '--context', '8.5'],
    ['--params', '70', '--context', '999999999999'],
    ['--params', '70', '--batch', '0'],
    ['--params', '70', '--batch', '5000'],
    ['--params', '70', '--quant', 'int3'],
    ['--params', '70', '--cloud', 'spot'],
    ['--model', 'nope'],
    ['--model', 'llama31-8b', '--params', '8'],
    ['--params'],
    ['--bogus'],
    ['--json=yes', '--params', '8'],
    ['--params', '8', '--params', '9']
  ];
  cases.forEach((args) => {
    const r = run(...args);
    assert.equal(r.code, 2, JSON.stringify(args) + ' -> ' + r.code);
    assert.ok(r.err.length > 0, JSON.stringify(args));
    assert.equal(r.out, '', JSON.stringify(args));
  });
});

test('a workload that fits nowhere exits 0 with an explanation', () => {
  const args = ['--model', 'deepseek-v3', '--quant', 'fp32', '--context', '131072', '--batch', '64'];
  const r = run(...args);
  assert.equal(r.code, 0);
  assert.match(r.out, /No GPU configuration/);
  assert.equal(json(...args).gpus.length, 0);
});
