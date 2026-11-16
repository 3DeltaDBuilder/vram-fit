'use strict';
/*
 * vram-fit: pure estimation functions. No I/O, no dependencies.
 * Data (GPUs, models, quantizations) lives in ../data/*.json and is loaded by the caller
 * or through the helpers exported here.
 */

const USABLE_FRACTION = 0.9; // at or below 90% of VRAM is "fits"; above that, up to 100%, is "tight"
const OVERHEAD_GB = 1.5; // CUDA context and framework, per GPU
const ACTIVATION_FRACTION = 0.1; // inference activations, as a fraction of weights
const FINETUNE_FACTOR = 2.0; // rough: fine-tuning roughly doubles weights, KV cache and activations
const GPU_COUNTS = [1, 2, 4, 8];
const GIB = Math.pow(1024, 3);
const LLAMA31_8B_KV_BYTES = 131072; // bytes per token, anchor for the custom-model extrapolation

const LIMITS = {
  paramsMin: 0.1, // billions
  paramsMax: 2000,
  contextMin: 1,
  contextMax: 10000000,
  batchMin: 1,
  batchMax: 1024
};

class InputError extends Error {
  constructor(message) {
    super(message);
    this.name = 'InputError';
  }
}

function byId(list, id) {
  for (let i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
  return null;
}

function checkNumber(name, value, min, max, integer) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new InputError(name + ' must be a finite number (got ' + String(value) + ')');
  }
  if (integer && !Number.isInteger(value)) {
    throw new InputError(name + ' must be a whole number (got ' + value + ')');
  }
  if (value <= 0) {
    throw new InputError(name + ' must be greater than zero (got ' + value + ')');
  }
  if (value < min || value > max) {
    throw new InputError(name + ' must be between ' + min + ' and ' + max + ' (got ' + value + ')');
  }
}

/*
 * KV cache bytes per token, FP16 cache.
 *   standard attention: 2 (K and V) * layers * kvHeads * headDim * 2 bytes
 *   multi-head latent attention (model.mla): layers * (kvLoraRank + ropeDim) * 2 bytes
 *   no architecture known (custom --params): 131072 * (params / 8) ^ 0.45, scaled from Llama 3.1 8B
 */
function kvBytesPerToken(model, customParams) {
  if (model && model.mla) {
    return model.layers * (model.mla.kvLoraRank + model.mla.ropeDim) * 2;
  }
  if (model && model.layers && model.kvHeads && model.headDim) {
    return 2 * model.layers * model.kvHeads * model.headDim * 2;
  }
  const p = Math.max(customParams, 0.1);
  return LLAMA31_8B_KV_BYTES * Math.pow(p / 8, 0.45);
}

/*
 * opts: { model | customParams, quant: {bpp}, context, batch, finetune }
 * Weights are decimal GB (billions of parameters * bytes per parameter).
 * KV cache is divided by 1024^3 (GiB) and added as-is, matching the web calculator.
 */
function estimate(opts) {
  const model = opts.model || null;
  const params = model ? model.params : opts.customParams;
  const quant = opts.quant;
  if (!quant || typeof quant.bpp !== 'number' || !(quant.bpp > 0)) {
    throw new InputError('a quantization with a positive bytes-per-parameter value is required');
  }
  checkNumber('params', params, LIMITS.paramsMin, LIMITS.paramsMax, false);
  const isImage = !!(model && model.image);
  let context = opts.context;
  let batch = opts.batch;
  if (!isImage) {
    checkNumber('context', context, LIMITS.contextMin, LIMITS.contextMax, true);
    checkNumber('batch', batch, LIMITS.batchMin, LIMITS.batchMax, true);
  }
  const kvBytes = isImage ? 0 : kvBytesPerToken(model, params);
  const weights = params * quant.bpp;
  const kv = isImage ? 0 : (kvBytes * context * batch) / GIB;
  const activation = weights * ACTIVATION_FRACTION;
  const finetune = !!opts.finetune;
  const f = finetune ? FINETUNE_FACTOR : 1;
  const w = weights * f;
  const k = kv * f;
  const a = activation * f;
  return {
    weights: w,
    kv: k,
    activation: a,
    overhead: OVERHEAD_GB,
    total: w + k + a + OVERHEAD_GB,
    kvBytesPerToken: kvBytes,
    kvExtrapolated: !isImage && !(model && (model.mla || (model.layers && model.kvHeads && model.headDim))),
    isImage: isImage,
    finetune: finetune
  };
}

/*
 * For every GPU with a price in the chosen cloud, find the smallest identical GPU count among
 * 1, 2, 4, 8 (capped at the GPU's per-pod maximum) that holds the estimate.
 * Per-GPU overhead is repeated for each GPU.
 * Status: fits (<= 90% of total VRAM), tight (<= 100%), multi / multi-tight when count > 1.
 * Rows are sorted by hourly cost, then by larger total VRAM.
 */
function evaluate(est, cloud, gpus) {
  if (cloud !== 'secure' && cloud !== 'community') {
    throw new InputError('cloud must be "secure" or "community" (got ' + String(cloud) + ')');
  }
  const shared = est.weights + est.kv + est.activation;
  const rows = [];
  gpus.forEach(function (g) {
    const price = g[cloud];
    if (price == null) return; // not offered in this cloud
    let count = 0;
    let status = null;
    let needed = 0;
    for (let ci = 0; ci < GPU_COUNTS.length; ci++) {
      const n = GPU_COUNTS[ci];
      if (n > (cloud === 'community' && g.maxCommunity ? g.maxCommunity : g.max)) break;
      needed = shared + OVERHEAD_GB * n;
      const cap = g.vram * n;
      if (needed <= cap * USABLE_FRACTION) { count = n; status = 'fits'; break; }
      if (needed <= cap) { count = n; status = 'tight'; break; }
    }
    if (!count) return;
    if (count > 1) status = status === 'tight' ? 'multi-tight' : 'multi';
    rows.push({
      gpu: g,
      count: count,
      status: status,
      needed: needed,
      totalVram: g.vram * count,
      price: price,
      hourly: price * count
    });
  });
  rows.sort(function (a, b) { return a.hourly - b.hourly || b.gpu.vram - a.gpu.vram; });
  return rows;
}

module.exports = {
  USABLE_FRACTION,
  OVERHEAD_GB,
  ACTIVATION_FRACTION,
  FINETUNE_FACTOR,
  GPU_COUNTS,
  LIMITS,
  InputError,
  byId,
  kvBytesPerToken,
  estimate,
  evaluate
};
