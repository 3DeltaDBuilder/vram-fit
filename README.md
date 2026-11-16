# vram-fit

A command-line estimator for how much GPU memory a language model needs, and which rentable GPUs hold it. Zero dependencies, Node 18 or newer.

> **Disclosure:** The maintainer has a RunPod referral link and may earn RunPod credits if you sign up and use RunPod. This tool does not require RunPod. Where RunPod rates appear they are dated snapshots, shown the same way whether or not you use the link.
>
> Referral link: I may earn RunPod credits if you sign up and use the platform. https://console.runpod.io/signup?ref=6ah8jpht

```
$ vram-fit --model llama31-70b --quant int4 --context 8192

Llama 3.1 70B (70B), INT4 / GPTQ 4-bit, context 8192, batch 1

  Weights         39.38 GB
  KV cache         2.50 GB  (327,680 B/token)
  Activations      3.94 GB
  Overhead         1.50 GB  (per GPU)
  Total           47.31 GB

GPUs that fit, Secure Cloud, cheapest first (RunPod rates as of 2026-11-15):
  ...
```

A browser version of the same arithmetic: https://3deltadbuilder.github.io/calculator.html

## What it does

Given a model (from the built-in table, or a parameter count), a quantization, a context length and a batch size, it prints estimated weights, KV cache, activations, framework overhead and a total, then lists GPU configurations with enough memory, cheapest first, using the dated rates in `data/gpus.json`.

## What it does not do

- It does not measure anything. These are estimates from a formula, not benchmarks.
- It says nothing about speed (tokens per second, latency, throughput).
- It does not say anything about output quality at a given quantization.
- Training is covered only by a rough 2x allowance (`--finetune`). LoRA and QLoRA are not modelled separately.
- Image models (SDXL) get weights, activations and overhead only. Resolution, VAE, refiner, LoRAs and ControlNet are not modelled.
- Estimates can be several percent low. In one comparison the weights estimate for Qwen2.5-32B AWQ was 18.28 GB against 19.33 GB of actual safetensors files (about 5% low); the cause has not been verified. Leave margin, and measure on the real hardware before you commit to a purchase or a long rental.
- It does not read a model's `config.json` or fetch anything from the network. Architecture numbers come from `data/models.json`, which is entered by hand; check them against the model's own `config.json` if exactness matters.

## Install and run

No install step is needed beyond Node 18+.

```
git clone <this repository>
cd vram-fit
node bin/vram-fit.js --model qwen25-32b --quant awq --context 16384
```

Or link the command: `npm link` (then `vram-fit ...`).

```
vram-fit --params 70 --quant int4 --context 8192 [--batch 1] [--finetune] [--cloud secure|community] [--json]
vram-fit --model llama31-70b --quant int4 --context 8192
vram-fit --list-models
```

| Option | Meaning | Default |
|---|---|---|
| `--model <id>` | Model from the built-in table (`--list-models`) | |
| `--params <b>` | Parameter count in billions, 0.1 to 2000 (custom model) | |
| `--quant <id>` | `fp32 fp16 int8 int4 awq q4km q5km q6k q80` | `fp16` |
| `--context <n>` | Context length in tokens, whole number, 1 to 10,000,000 | 4096 |
| `--batch <n>` | Simultaneous sequences held in the KV cache, 1 to 1024 | 1 |
| `--finetune` | Rough fine-tuning allowance (see below) | off |
| `--cloud <name>` | `secure` or `community` rates | `secure` |
| `--json` | Machine-readable output | off |

Give exactly one of `--model` and `--params`. With `--params`, the layer and head counts are unknown, so the KV cache is extrapolated (see formulas) and the output says so. The example `--params 70` therefore gives a slightly different KV figure than `--model llama31-70b`, which uses the real architecture.

Exit codes: `0` ok, `2` bad input (unknown option, missing value, zero, negative, NaN, infinite, non-numeric or absurd values). The message goes to stderr. A workload that fits on no listed GPU is not an error: it prints an explanation and exits 0.

## Formulas

All of it is in `src/estimate.js` (pure functions, no I/O).

```
weights      = params_in_billions x bytes_per_param                 (decimal GB)
kv_bytes     = 2 x layers x kv_heads x head_dim x 2                 (per token, FP16 cache)
kv_cache     = kv_bytes x context x batch / 1024^3
activations  = 0.10 x weights
overhead     = 1.5 GB per GPU
total        = weights + kv_cache + activations + overhead
```

Special cases:

- Multi-head latent attention (DeepSeek V3): `kv_bytes = layers x (kv_lora_rank + rope_dim) x 2 = 61 x (512 + 64) x 2 = 70,272` bytes per token.
- Custom `--params` model: `kv_bytes = 131,072 x (params / 8) ^ 0.45`, scaled from Llama 3.1 8B (131,072 bytes per token). This is a rough extrapolation.
- `--finetune`: weights, KV cache and activations are multiplied by 2; overhead is not. This is a rough allowance for gradients and optimizer state, not a measurement.
- Image models: no KV cache; context and batch are ignored.

Units: weights are decimal GB (billions of parameters times bytes). The KV cache is divided by 1024^3, so it is in GiB, and is added to the total as-is. This matches the browser calculator. The mixed units understate the KV term by about 7% (1 GiB is 1.074 GB); for example 24.0 GiB of KV cache is 25.8 GB.

Bytes per parameter (`data/quants.json`): FP32 4.0, FP16/BF16 2.0, INT8 1.0, INT4/GPTQ-4 0.5625, AWQ 0.5625, GGUF Q4_K_M 0.604, Q5_K_M 0.711, Q6_K 0.821, Q8_0 1.0625.

### Which GPUs fit

For each GPU with a rate in the chosen cloud, the tool tries 1, 2, 4 and 8 identical GPUs (never 3, 5, 6 or 7), up to the per-pod maximum recorded for that GPU, and takes the smallest count that holds the estimate. The per-GPU overhead is added once per GPU:

```
needed = weights + kv_cache + activations + 1.5 x gpu_count
```

| Status | Rule |
|---|---|
| `fits` | `needed` is at most 90% of total VRAM |
| `tight` | above 90%, up to 100% |
| `multi` | fits with more than one GPU |
| `multi, tight` | more than one GPU, between 90% and 100% |

GPUs with a null rate for the chosen cloud (for example MI300X in the community cloud) are skipped. Rows are sorted by hourly cost (per-GPU rate times count), then by larger total VRAM. Splitting a model across GPUs is bandwidth-sensitive; NVLink-class GPUs generally handle it better than PCIe cards. The tool does not model that.

## Pricing and the rates file

Referral link: I may earn RunPod credits if you sign up and use the platform. https://console.runpod.io/signup?ref=6ah8jpht

The tool works without a RunPod account and without the link. Rates in `data/gpus.json` are RunPod on-demand per-GPU hourly rates in USD, fetched from the RunPod API on **2026-11-15** (`"ratesDate"` in the file). Rates change; check the provider before you rent. The `--json` output includes the date and the source note.

To update the rates, edit `data/gpus.json`:

1. Change each GPU's `secure` and `community` values (use `null` where a GPU is not offered in that cloud).
2. Set `ratesDate` to the date you fetched them, in `YYYY-MM-DD` form.
3. Update the `source` note if you got the numbers somewhere else.
4. Run `node --test`. The tests check that every entry is well formed; some assertions pin the shipped 2026-11-15 values (the date and the cheapest 70B entry), so change those too when you intentionally refresh the data.

Other GPU providers, or a GPU not in the list, can be added the same way: `id`, `name`, `vram` (GB), `secure`, `community`, `max` (largest count per pod), `nvlink`.

## Measured versus estimated

Measured rows are added only from real runs. Current entries:

| Model and setup | Estimated | Measured | Difference | Note |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct-AWQ, weights only | 18.28 GB | 19.33 GB (sum of 5 safetensors shards, 19,328,993,904 bytes, from the Hugging Face API) | about 5.4% low | File size on disk, not GPU memory. Cause not verified. |

No GPU-memory measurements are recorded yet.

## Tests

```
node --test
```

The tests use `node:test` and cover the golden figures (Llama 3.1 70B INT4 at 8K = 47.31 GB, Qwen2.5-32B AWQ at 16K = 25.61 GB, Mixtral 8x7B INT4 at 4K = 30.90 GB, Llama 3.1 8B KV = 131,072 bytes per token and 16.00 GiB at 131,072 tokens, DeepSeek V3 = 70,272 bytes per token, Qwen2.5-14B KV at 128K = 24.0), the 90% boundary, the 1/2/4/8 GPU counts, null-rate skipping, input validation and exit codes.

## License

MIT. See `LICENSE`.
