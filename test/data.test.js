'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const gpuData = require('../data/gpus.json');
const MODELS = require('../data/models.json');
const QUANTS = require('../data/quants.json');

test('gpus.json has a rates note and a source note', () => {
  assert.equal(gpuData.ratesChecked, 'within the last 30 days');
  assert.ok(typeof gpuData.source === 'string' && gpuData.source.length > 20);
  assert.ok(Array.isArray(gpuData.gpus) && gpuData.gpus.length > 0);
});

test('every GPU entry is well formed', () => {
  const ids = new Set();
  gpuData.gpus.forEach((g) => {
    assert.ok(!ids.has(g.id), 'duplicate id ' + g.id);
    ids.add(g.id);
    assert.ok(typeof g.name === 'string' && g.name);
    assert.ok(Number.isFinite(g.vram) && g.vram > 0, g.id + ' vram');
    assert.ok(Number.isFinite(g.secure) && g.secure > 0, g.id + ' secure rate');
    assert.ok(g.community === null || (Number.isFinite(g.community) && g.community > 0), g.id + ' community rate');
    assert.ok(Number.isInteger(g.max) && g.max >= 1, g.id + ' max');
  });
  assert.equal(gpuData.gpus.find((g) => g.id === 'mi300x').community, null);
});

test('every model entry is well formed', () => {
  const ids = new Set();
  MODELS.forEach((m) => {
    assert.ok(!ids.has(m.id), 'duplicate id ' + m.id);
    ids.add(m.id);
    assert.ok(Number.isFinite(m.params) && m.params > 0, m.id + ' params');
    if (m.image) return;
    assert.ok(Number.isInteger(m.layers) && m.layers > 0, m.id + ' layers');
    if (m.mla) {
      assert.ok(m.mla.kvLoraRank > 0 && m.mla.ropeDim > 0, m.id + ' mla');
    } else {
      assert.ok(m.kvHeads > 0 && m.headDim > 0, m.id + ' kv heads / head dim');
    }
  });
});

test('quantization table', () => {
  const byId = Object.fromEntries(QUANTS.map((q) => [q.id, q.bpp]));
  assert.equal(byId.fp16, 2);
  assert.equal(byId.int8, 1);
  assert.equal(byId.int4, 0.5625);
  assert.equal(byId.awq, 0.5625);
  QUANTS.forEach((q) => assert.ok(q.bpp > 0 && q.bpp <= 4, q.id));
});
