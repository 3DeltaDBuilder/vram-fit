#!/usr/bin/env node
'use strict';

const path = require('path');
const lib = require('../src/estimate.js');
const gpuData = require('../data/gpus.json');
const MODELS = require('../data/models.json');
const QUANTS = require('../data/quants.json');
const pkg = require('../package.json');

const WEB_URL = 'https://3deltadbuilder.github.io/calculator.html';

const STATUS_LABEL = { fits: 'fits', tight: 'tight', multi: 'multi', 'multi-tight': 'multi, tight' };

const OPTIONS = {
  '--params': { key: 'params', value: true },
  '--model': { key: 'model', value: true },
  '--quant': { key: 'quant', value: true },
  '--context': { key: 'context', value: true },
  '--batch': { key: 'batch', value: true },
  '--cloud': { key: 'cloud', value: true },
  '--finetune': { key: 'finetune', value: false },
  '--json': { key: 'json', value: false },
  '--list-models': { key: 'listModels', value: false },
  '--help': { key: 'help', value: false },
  '-h': { key: 'help', value: false },
  '--version': { key: 'version', value: false }
};

function usage() {
  return [
    'vram-fit ' + pkg.version + ': estimate GPU memory for running a model, and list GPUs that fit.',
    '',
    'Usage:',
    '  vram-fit --model <id> [--quant <id>] [--context <tokens>] [--batch <n>] [options]',
    '  vram-fit --params <billions> [--quant <id>] [--context <tokens>] [--batch <n>] [options]',
    '  vram-fit --list-models',
    '',
    'Options:',
    '  --model <id>       model from the built-in table (see --list-models)',
    '  --params <b>       parameter count in billions, ' + lib.LIMITS.paramsMin + ' to ' + lib.LIMITS.paramsMax + ' (custom model)',
    '  --quant <id>       ' + QUANTS.map(function (q) { return q.id; }).join(', ') + ' (default fp16)',
    '  --context <n>      context length in tokens, whole number (default 4096)',
    '  --batch <n>        simultaneous sequences held in the KV cache, 1 to ' + lib.LIMITS.batchMax + ' (default 1)',
    '  --finetune         rough fine-tuning allowance: weights, KV cache and activations x ' + lib.FINETUNE_FACTOR,
    '  --cloud <name>     secure or community rates (default secure)',
    '  --json             machine-readable output',
    '  --list-models      list the built-in models',
    '  --help, --version',
    '',
    'Exit codes: 0 ok, 2 bad input.',
    'This is an estimate, not a measurement. Rates are dated snapshots (' + gpuData.ratesDate + ').'
  ].join('\n');
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    let arg = argv[i];
    let inline = null;
    const eq = arg.indexOf('=');
    if (arg.indexOf('--') === 0 && eq > 0) {
      inline = arg.slice(eq + 1);
      arg = arg.slice(0, eq);
    }
    const spec = OPTIONS[arg];
    if (!spec) throw new lib.InputError('unknown option: ' + arg + ' (try --help)');
    if (spec.value) {
      let v = inline;
      if (v === null) {
        i++;
        if (i >= argv.length) throw new lib.InputError(arg + ' needs a value');
        v = argv[i];
      }
      if (v === '') throw new lib.InputError(arg + ' needs a non-empty value');
      if (Object.prototype.hasOwnProperty.call(out, spec.key)) throw new lib.InputError(arg + ' given more than once');
      out[spec.key] = v;
    } else {
      if (inline !== null) throw new lib.InputError(arg + ' does not take a value');
      out[spec.key] = true;
    }
  }
  return out;
}

// Strict decimal parser: rejects '', 'abc', '0x10', 'NaN', 'Infinity', '1,000'. Accepts '70', '7.6', '1e3'.
function parseNumber(name, text) {
  if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(text)) {
    throw new lib.InputError(name + ' must be a number (got "' + text + '")');
  }
  const n = Number(text);
  if (!Number.isFinite(n)) throw new lib.InputError(name + ' is not a finite number (got "' + text + '")');
  return n;
}

function resolveInputs(args) {
  if (args.model && args.params) throw new lib.InputError('use either --model or --params, not both');
  if (!args.model && !args.params) throw new lib.InputError('give --model <id> or --params <billions> (or --list-models)');

  let model = null;
  let customParams = null;
  if (args.model) {
    model = lib.byId(MODELS, args.model);
    if (!model) throw new lib.InputError('unknown model "' + args.model + '" (see --list-models)');
  } else {
    customParams = parseNumber('--params', args.params);
  }

  const quantId = args.quant || 'fp16';
  const quant = lib.byId(QUANTS, quantId);
  if (!quant) throw new lib.InputError('unknown quantization "' + quantId + '" (one of: ' + QUANTS.map(function (q) { return q.id; }).join(', ') + ')');

  const context = args.context === undefined ? 4096 : parseNumber('--context', args.context);
  const batch = args.batch === undefined ? 1 : parseNumber('--batch', args.batch);
  const cloud = args.cloud || 'secure';
  if (cloud !== 'secure' && cloud !== 'community') {
    throw new lib.InputError('--cloud must be "secure" or "community" (got "' + cloud + '")');
  }
  return { model, customParams, quant, context, batch, cloud, finetune: !!args.finetune };
}

function round(n, d) {
  const f = Math.pow(10, d);
  return Math.round(n * f) / f;
}

function buildResult(inp, est, rows) {
  return {
    ratesDate: gpuData.ratesDate,
    ratesSource: gpuData.source,
    cloud: inp.cloud,
    input: {
      model: inp.model ? inp.model.id : null,
      paramsBillions: inp.model ? inp.model.params : inp.customParams,
      quant: inp.quant.id,
      bytesPerParam: inp.quant.bpp,
      context: est.isImage ? null : inp.context,
      batch: est.isImage ? null : inp.batch,
      finetune: est.finetune
    },
    estimateGB: {
      weights: round(est.weights, 2),
      kvCache: round(est.kv, 2),
      activations: round(est.activation, 2),
      overhead: round(est.overhead, 2),
      total: round(est.total, 2)
    },
    kvBytesPerToken: est.isImage ? null : Math.round(est.kvBytesPerToken),
    notes: notesFor(inp, est),
    gpus: rows.map(function (r) {
      return {
        id: r.gpu.id,
        name: r.gpu.name,
        count: r.count,
        status: r.status,
        vramPerGpuGB: r.gpu.vram,
        totalVramGB: r.totalVram,
        neededGB: round(r.needed, 2),
        ratePerGpuHour: r.price,
        hourly: round(r.hourly, 2)
      };
    })
  };
}

function notesFor(inp, est) {
  const notes = [];
  if (est.finetune) notes.push('Fine-tuning allowance: weights, KV cache and activations are doubled. Real usage depends on the method (full fine-tune, LoRA, QLoRA).');
  if (inp.model && inp.model.moe) notes.push(inp.model.name + ' is a mixture-of-experts model: all expert weights must be resident in VRAM.');
  if (est.kvExtrapolated) notes.push('Custom model: layer and head counts are unknown, so the KV cache is extrapolated from Llama 3.1 8B (params^0.45). Treat it as rough.');
  if (est.isImage) notes.push('Image model: only weights, activations and overhead are estimated.');
  notes.push('Estimate only. Not a benchmark; says nothing about speed. Estimates can be several percent low.');
  return notes;
}

function gb(n) { return n.toFixed(2).padStart(8) + ' GB'; }

function renderText(res, inp, est) {
  const lines = [];
  const who = inp.model ? inp.model.name + ' (' + inp.model.params + 'B)' : 'Custom model (' + inp.customParams + 'B)';
  lines.push(who + ', ' + inp.quant.name + (est.isImage ? '' : ', context ' + inp.context + ', batch ' + inp.batch) + (est.finetune ? ', fine-tune allowance' : ''));
  lines.push('');
  lines.push('  Weights      ' + gb(est.weights));
  if (!est.isImage) lines.push('  KV cache     ' + gb(est.kv) + '  (' + Math.round(est.kvBytesPerToken).toLocaleString('en-US') + ' B/token)');
  lines.push('  Activations  ' + gb(est.activation));
  lines.push('  Overhead     ' + gb(est.overhead) + '  (per GPU)');
  lines.push('  Total        ' + gb(est.total));
  lines.push('');
  const cloudLabel = inp.cloud === 'community' ? 'Community Cloud' : 'Secure Cloud';
  if (!res.gpus.length) {
    lines.push('No GPU configuration (1, 2, 4 or 8 identical GPUs, up to each GPU\'s per-pod maximum) in the ' + cloudLabel + ' list has enough memory.');
    lines.push('Try a smaller quantization, a shorter context, or a smaller batch.');
  } else {
    lines.push('GPUs that fit, ' + cloudLabel + ', cheapest first (RunPod rates as of ' + gpuData.ratesDate + '):');
    lines.push('');
    const nameW = Math.max(3, Math.max.apply(null, res.gpus.map(function (g) { return displayName(g).length; })));
    lines.push('  ' + 'GPU'.padEnd(nameW) + '  ' + 'VRAM'.padStart(8) + '  ' + 'Need'.padStart(9) + '  ' + 'Status'.padEnd(12) + '  ' + '$/hr'.padStart(7));
    res.gpus.forEach(function (g) {
      lines.push('  ' + displayName(g).padEnd(nameW) + '  ' + (g.totalVramGB + ' GB').padStart(8) + '  ' + (g.neededGB.toFixed(1) + ' GB').padStart(9) + '  ' + STATUS_LABEL[g.status].padEnd(12) + '  ' + ('$' + g.hourly.toFixed(2)).padStart(7));
    });
    lines.push('');
    lines.push('fits = needs at most 90% of VRAM; tight = 90% to 100%; multi = split across identical GPUs.');
  }
  lines.push('');
  res.notes.forEach(function (n) { lines.push('Note: ' + n); });
  lines.push('Rates are dated snapshots and change. Check the provider. Web version: ' + WEB_URL);
  return lines.join('\n');
}

function displayName(g) {
  return (g.count > 1 ? g.count + 'x ' : '') + g.name;
}

function listModels(json) {
  if (json) {
    return JSON.stringify(MODELS.map(function (m) {
      return { id: m.id, name: m.name, paramsBillions: m.params, image: !!m.image, moe: !!m.moe };
    }), null, 2);
  }
  const idW = Math.max.apply(null, MODELS.map(function (m) { return m.id.length; }));
  const nameW = Math.max.apply(null, MODELS.map(function (m) { return m.name.length; }));
  const lines = ['  ' + 'ID'.padEnd(idW) + '  ' + 'Name'.padEnd(nameW) + '  Params'];
  MODELS.forEach(function (m) {
    lines.push('  ' + m.id.padEnd(idW) + '  ' + m.name.padEnd(nameW) + '  ' + m.params + 'B' + (m.moe ? '  (MoE)' : '') + (m.image ? '  (image)' : ''));
  });
  return lines.join('\n');
}

function main(argv) {
  let args;
  try {
    args = parseArgs(argv);
    if (args.help) { process.stdout.write(usage() + '\n'); return 0; }
    if (args.version) { process.stdout.write(pkg.version + '\n'); return 0; }
    if (args.listModels) { process.stdout.write(listModels(args.json) + '\n'); return 0; }
    const inp = resolveInputs(args);
    const est = lib.estimate({ model: inp.model, customParams: inp.customParams, quant: inp.quant, context: inp.context, batch: inp.batch, finetune: inp.finetune });
    const rows = lib.evaluate(est, inp.cloud, gpuData.gpus);
    const res = buildResult(inp, est, rows);
    process.stdout.write((args.json ? JSON.stringify(res, null, 2) : renderText(res, inp, est)) + '\n');
    return 0;
  } catch (e) {
    if (e instanceof lib.InputError) {
      process.stderr.write(path.basename(process.argv[1] || 'vram-fit').replace(/\.js$/, '') + ': ' + e.message + '\n');
      return 2;
    }
    throw e;
  }
}

process.exitCode = main(process.argv.slice(2));
