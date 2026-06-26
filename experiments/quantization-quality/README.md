# GGUF Quantization — Speed vs Quality

> Experiment #3 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**How low can you quantize before it breaks?** Quantizing a GGUF model buys speed
and VRAM — but somewhere down there, quality falls off a cliff. This sweeps Q2_K →
Q8_0 on one model and measures exactly where.

## What it measures (per quant)

- **Speed** — generation tok/s on the shared 12-prompt suite.
- **Footprint** — file size + VRAM used (nvidia-smi delta on load).
- **Quality**, two ways:
  - **Graded math pass-rate** — an 8-question set with known integer answers
    (relatable: "at Q2 it got 3/8 wrong").
  - **Output similarity** — how close each quant's greedy output stays to the
    highest-precision quant present (mean difflib ratio over the suite). 100% =
    quantization changed nothing it said; the drop marks the cliff.

> A note on rigor: output-similarity and a small graded set are self-contained and
> relatable, but they're proxies. The gold-standard quality metrics are
> **perplexity** and **KL-divergence vs FP16** (`llama-perplexity --kl-divergence`).
> A future pass can add those; this experiment favors a metric you can feel.

## Run it

Download the quants into your models dir (`SPECBENCH_MODELS_DIR`), named
`Qwen2.5-Coder-7B-Instruct-<QUANT>.gguf` (e.g. bartowski's GGUF repo). Then:

```bash
python scripts/bench.py --backend cuda --gpu 0    # missing quants are skipped
python scripts/aggregate.py        # -> results/data.json
python scripts/inject.py           # bakes into site/index.html
```

Run it per device (`--backend cuda|vulkan|cpu`, `--gpu 0|1`) if you want the
speed/quality tradeoff on each. Best on an otherwise-idle GPU so the VRAM delta is
clean.

## Layout

```
scripts/  bench.py · aggregate.py · inject.py · make_submission.py · quality_set.py (graded math)
results/  results.json + data.json + community/  (logs/screenshots gitignored)
site/     index.html — quality-vs-quant + speed charts
```

Built on [`harness/`](../../harness/) (`LlamaServer` + `measure` + shared `prompts.py`).

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
