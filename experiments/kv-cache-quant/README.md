# KV-cache Quantization: Fit More Context in the Same VRAM

> Experiment #4 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**Does quantizing the KV cache let you fit more context without hurting quality?**
f16, q8_0, and q4_0 KV caches on the same model at 8192-token context. Smaller
caches mean more context fits in the same VRAM — this measures exactly what you give
up (if anything) in speed and output quality.

Status: **scaffolded, results pending** (waiting on a hardware run).

## What it measures (per KV-cache type)

- **VRAM footprint** — nvidia-smi used-memory delta on model load at 8192-token context.
- **Speed** — generation tok/s on the shared 12-prompt suite.
- **Output similarity** — how close each config's greedy output stays to the f16
  baseline (mean difflib ratio over the suite). 100% = the quantized KV cache
  changed nothing the model says; a drop marks a quality cost.

> A note on rigor: output similarity is a self-contained and relatable proxy metric.
> The gold-standard quality measure is perplexity or KL-divergence vs FP16.
> This experiment favors a signal you can feel.

## Run it

Download `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` into your models dir
(`SPECBENCH_MODELS_DIR`), then:

```bash
python scripts/bench.py --backend cuda --gpu 0 --ctx 8192
python scripts/aggregate.py        # -> results/data.json
python scripts/inject.py           # bakes into site/index.html
```

Run it per device (`--backend cuda|vulkan|cpu`, `--gpu 0|1`) if you want to compare
KV-cache VRAM savings across backends. Best on an otherwise-idle GPU so the VRAM
delta is clean.

## Layout

```
scripts/  bench.py · aggregate.py · inject.py · make_submission.py
results/  results.json + data.json + community/  (logs gitignored)
site/     index.html — VRAM + speed + similarity charts (shows "results pending" until run)
```

Built on [`harness/`](../../harness/) (`LlamaServer` + `measure` + shared `prompts.py`).

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
