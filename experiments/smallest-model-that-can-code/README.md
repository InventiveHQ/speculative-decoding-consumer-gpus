# Smallest Model That Can Code

> Experiment #11 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**How small a model can actually solve problems?** Tiny models are fast and cheap
— but somewhere down there, they stop getting things right. This sweeps
Qwen2.5-Coder-Instruct from 0.5B to 14B parameters to find the knee: the smallest
model that reliably passes, traded against speed and size.

## What it measures (per model size)

- **Speed** — generation tok/s on the shared 12-prompt suite.
- **Footprint** — file size (MB) + VRAM used (nvidia-smi delta on load).
- **Correctness** — graded pass-rate on a 10-question math/reasoning set with
  known integer answers. Relatable: "at 0.5B it gets 3/10; at 7B it nails 9/10."

> **A note on code correctness:** executing untrusted model-generated code is unsafe
> in a benchmark harness, so correctness here is approximated by a graded
> math/reasoning set (exact-answer matching only). A real code-execution sandbox is
> a planned future addition.

## Run it

Download the Qwen2.5-Coder-Instruct GGUF files into your models dir
(`SPECBENCH_MODELS_DIR`), named as listed in `scripts/bench.py`. Then:

```bash
python scripts/bench.py --backend cuda --gpu 0    # missing models are skipped
python scripts/aggregate.py        # -> results/data.json
python scripts/inject.py           # bakes into site/index.html
```

Run per device (`--backend cuda|vulkan|cpu`, `--gpu 0|1`) to compare hardware.
Best on an otherwise-idle GPU so the VRAM delta is clean.

## Models

| Label | File |
|-------|------|
| 0.5B  | Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf   |
| 1.5B  | Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf |
| 3B    | Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf   |
| 7B    | Qwen2.5-Coder-7B-Instruct-Q4_K_S.gguf   |
| 14B   | Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf  |

## Layout

```
scripts/  bench.py · aggregate.py · inject.py · make_submission.py · quality_set.py (graded set)
results/  results.json + data.json + community/  (logs gitignored)
site/     index.html — pass-rate + tok/s charts
```

Built on [`harness/`](../../harness/) (`LlamaServer` + `measure` + shared `prompts.py`).

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
