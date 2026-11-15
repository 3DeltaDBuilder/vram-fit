'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const lib = require('../src/estimate.js');
const gpuData = require('../data/gpus.json');
const MODELS = require('../data/models.json');
const QUANTS = require('../data/quants.json');

const model = (id) => lib.byId(MODELS, id);
const quant = (id) => lib.byId(QUANTS, id);
const est = (m, q, context, batch, finetune) =>
  lib.estimate({ model: model(m), quant: quant(q), context, batch: batch || 1, finetune: !!finetune });
const r2 = (n) => Number(n.toFixed(2));
const GIB = 1024 ** 3;

test('Llama 3.1 70B int4 at 8K context totals 47.31 GB', () => {
  const e = est('llama31-70b', 'int4', 8192);
  assert.equal(r2(e.weights), 39.38); // 70 x 0.5625
  assert.equal(r2(e.kv), 2.5); // 327,680 B/token x 8192 / 1024^3
  assert.equal(r2(e.activation), 3.94);
  assert.equal(e.overhead, 1.5);
  assert.equal(r2(e.total), 47.31);
});

test('Qwen2.5-32B AWQ at 16K context totals 25.61 GB', () => {
  const e = est('qwen25-32b', 'awq', 16384);
  assert.equal(r2(e.weights), 18.28);
  assert.equal(r2(e.kv), 4.0); // 262,144 B/token
  assert.equal(r2(e.total), 25.61);
});

test('Mixtral 8x7B int4 at 4K context totals 30.90 GB', () => {
  assert.equal(r2(est('mixtral-8x7b', 'int4', 4096).total), 30.9);
});

test('KV cache bytes per token', () => {
  assert.equal(lib.kvBytesPerToken(model('llama31-8b')), 131072);
  assert.equal(lib.kvBytesPerToken(model('llama31-70b')), 327680);
  assert.equal(lib.kvBytesPerToken(model('qwen25-32b')), 262144);
  assert.equal(lib.kvBytesPerToken(model('qwen25-14b')), 196608);
});

test('Llama 3.1 8B KV cache at 131,072 tokens is 16.00 GiB', () => {
  assert.equal((131072 * 131072) / GIB, 16);
  assert.equal(est('llama31-8b', 'fp16', 131072).kv, 16);
});

test('DeepSeek V3 uses 70,272 B/token (multi-head latent attention)', () => {
  assert.equal(lib.kvBytesPerToken(model('deepseek-v3')), 70272);
  assert.equal(61 * (512 + 64) * 2, 70272);
});

test('Qwen2.5-14B KV cache at 128K context is 24.0 GB (48 layers, 8 KV heads)', () => {
  const m = model('qwen25-14b');
  assert.equal(m.layers, 48);
  assert.equal(m.kvHeads, 8);
  assert.equal(est('qwen25-14b', 'fp16', 131072).kv, 24);
});

test('custom model KV is extrapolated from Llama 3.1 8B by params^0.45', () => {
  assert.equal(lib.kvBytesPerToken(null, 8), 131072);
  const e = lib.estimate({ customParams: 70, quant: quant('int4'), context: 8192, batch: 1 });
  assert.ok(e.kvExtrapolated);
  assert.ok(Math.abs(e.kvBytesPerToken - 131072 * Math.pow(70 / 8, 0.45)) < 1e-6);
  assert.equal(est('llama31-70b', 'int4', 8192).kvExtrapolated, false);
});

test('fine-tune allowance doubles weights, KV and activations but not overhead', () => {
  const a = est('llama31-8b', 'fp16', 4096);
  const b = est('llama31-8b', 'fp16', 4096, 1, true);
  assert.equal(b.weights, a.weights * 2);
  assert.equal(b.kv, a.kv * 2);
  assert.equal(b.activation, a.activation * 2);
  assert.equal(b.overhead, a.overhead);
  assert.equal(b.finetune, true);
});

test('image model has no KV cache and ignores context and batch', () => {
  const e = lib.estimate({ model: model('sdxl'), quant: quant('fp16'), context: 999999999, batch: 0 });
  assert.equal(e.kv, 0);
  assert.equal(r2(e.total), 9.2); // 7.0 + 0.7 + 1.5
});

test('total grows with context, batch and bits per parameter', () => {
  const base = est('llama31-8b', 'int8', 4096).total;
  assert.ok(est('llama31-8b', 'int8', 8192).total > base);
  assert.ok(est('llama31-8b', 'int8', 4096, 2).total > base);
  assert.ok(est('llama31-8b', 'fp16', 4096).total > base);
  assert.ok(est('llama31-8b', 'int4', 4096).total < base);
});

test('estimate rejects zero, negative, NaN, infinite and absurd values', () => {
  const q = quant('fp16');
  const bad = [
    { customParams: 0, quant: q, context: 4096, batch: 1 },
    { customParams: -7, quant: q, context: 4096, batch: 1 },
    { customParams: NaN, quant: q, context: 4096, batch: 1 },
    { customParams: Infinity, quant: q, context: 4096, batch: 1 },
    { customParams: 1e9, quant: q, context: 4096, batch: 1 },
    { customParams: '70', quant: q, context: 4096, batch: 1 },
    { customParams: 70, quant: q, context: 0, batch: 1 },
    { customParams: 70, quant: q, context: -1, batch: 1 },
    { customParams: 70, quant: q, context: NaN, batch: 1 },
    { customParams: 70, quant: q, context: 4096.5, batch: 1 },
    { customParams: 70, quant: q, context: 1e12, batch: 1 },
    { customParams: 70, quant: q, context: 4096, batch: 0 },
    { customParams: 70, quant: q, context: 4096, batch: -2 },
    { customParams: 70, quant: q, context: 4096, batch: 100000 },
    { customParams: 70, quant: null, context: 4096, batch: 1 }
  ];
  bad.forEach((o) => assert.throws(() => lib.estimate(o), lib.InputError, JSON.stringify(o)));
});

test('evaluate rejects an unknown cloud', () => {
  const e = est('llama31-8b', 'fp16', 4096);
  assert.throws(() => lib.evaluate(e, 'spot', gpuData.gpus), lib.InputError);
});

// Boundary: an 80 GB GPU has 72 GB usable (90%). shared + 1.5 overhead == 72 exactly.
const A100 = { id: 'x', name: 'X', vram: 80, secure: 1, community: 1, max: 1 };
const onlyOne = (shared, gpus) =>
  lib.evaluate({ weights: shared, kv: 0, activation: 0 }, 'secure', gpus || [A100]);

test('fits at exactly 90% of VRAM, tight just above, nothing above 100%', () => {
  assert.equal(onlyOne(70.5)[0].status, 'fits'); // needs 72.0 = 90%
  assert.equal(onlyOne(70.51)[0].status, 'tight'); // needs 72.01
  assert.equal(onlyOne(78.5)[0].status, 'tight'); // needs 80.0 = 100%
  assert.equal(onlyOne(78.51).length, 0); // needs 80.01
});

test('GPU counts are limited to 1, 2, 4, 8 (never 3, 5, 6, 7)', () => {
  assert.deepEqual(lib.GPU_COUNTS, [1, 2, 4, 8]);
  const gpu = { id: 'y', name: 'Y', vram: 24, secure: 1, community: 1, max: 8 };
  const seen = new Set();
  for (let shared = 1; shared < 160; shared += 0.5) {
    onlyOne(shared, [gpu]).forEach((r) => seen.add(r.count));
  }
  assert.deepEqual([...seen].sort((a, b) => a - b), [1, 2, 4, 8]);
  // 3 x 24 GB would hold 60 GB shared; the smallest allowed count above 1 GPU is 4.
  assert.equal(onlyOne(60, [gpu])[0].count, 4);
});

test('per-GPU overhead is repeated for each GPU in a multi-GPU setup', () => {
  const gpu = { id: 'y', name: 'Y', vram: 24, secure: 1, community: 1, max: 8 };
  const r = onlyOne(60, [gpu])[0];
  assert.equal(r.count, 4);
  assert.equal(r.needed, 60 + 1.5 * 4);
  assert.equal(r.totalVram, 96);
  assert.equal(r.status, 'multi');
});

test('smallest count wins and max per pod is respected', () => {
  const single = { id: 's', name: 'S', vram: 32, secure: 1, community: 1, max: 1 };
  assert.equal(onlyOne(100, [single]).length, 0); // would need 4 GPUs but max is 1
  const dual = { id: 'd', name: 'D', vram: 32, secure: 1, community: 1, max: 2 };
  assert.equal(onlyOne(40, [dual])[0].count, 2);
  assert.equal(onlyOne(100, [dual]).length, 0);
});

test('multi-tight when a multi-GPU set is between 90% and 100%', () => {
  const gpu = { id: 'g', name: 'G', vram: 24, secure: 1, community: 1, max: 2 };
  // 2 x 24 = 48 GB, 90% = 43.2 GB; needed = 43 + 3.0 = 46.0, so tight on 2 GPUs.
  const r = onlyOne(43, [gpu])[0];
  assert.equal(r.count, 2);
  assert.equal(r.status, 'multi-tight');
});

test('GPUs without a price in the chosen cloud are skipped', () => {
  const e = est('llama31-8b', 'int4', 4096);
  const secure = lib.evaluate(e, 'secure', gpuData.gpus).map((r) => r.gpu.id);
  const community = lib.evaluate(e, 'community', gpuData.gpus).map((r) => r.gpu.id);
  assert.ok(secure.includes('mi300x'));
  assert.ok(!community.includes('mi300x'));
  const nullGpu = { id: 'n', name: 'N', vram: 80, secure: 1, community: null, max: 1 };
  assert.equal(lib.evaluate(e, 'community', [nullGpu]).length, 0);
});

test('rows are sorted by hourly cost, ties by larger VRAM', () => {
  const e = est('llama31-8b', 'int4', 4096);
  const rows = lib.evaluate(e, 'secure', gpuData.gpus);
  for (let i = 1; i < rows.length; i++) {
    assert.ok(rows[i - 1].hourly <= rows[i].hourly);
  }
  const a = { id: 'a', name: 'A', vram: 24, secure: 1, community: 1, max: 1 };
  const b = { id: 'b', name: 'B', vram: 48, secure: 1, community: 1, max: 1 };
  assert.deepEqual(lib.evaluate(e, 'secure', [a, b]).map((r) => r.gpu.id), ['b', 'a']);
});

test('hourly cost is the per-GPU rate times the GPU count', () => {
  const gpu = { id: 'y', name: 'Y', vram: 24, secure: 0.5, community: 0.2, max: 8 };
  const r = onlyOne(60, [gpu])[0];
  assert.equal(r.hourly, 0.5 * 4);
  const rc = lib.evaluate({ weights: 60, kv: 0, activation: 0 }, 'community', [gpu])[0];
  assert.equal(rc.hourly, 0.2 * 4);
});

test('70B int4 at 8K on the shipped catalog: cheapest entry is a 48 GB card, marked tight', () => {
  const rows = lib.evaluate(est('llama31-70b', 'int4', 8192), 'secure', gpuData.gpus);
  assert.equal(rows[0].gpu.id, 'a40');
  assert.equal(rows[0].status, 'tight');
  assert.equal(rows[0].hourly, 0.59);
});
