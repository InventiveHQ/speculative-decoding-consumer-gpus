# CPU Thread Scaling for LLM Inference

> Experiment #9 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**More CPU threads made my LLM slower.** Setting `-t 16` when your CPU has 8 physical
cores can halve your tok/s. This sweeps thread counts from 1 → 16 and shows exactly
where the hyperthreading penalty kicks in.

Status: **scaffolded, results pending** (waiting on a hardware run).

## What it measures

- **Tok/s per thread count** — 1, 2, 4, 6, 8, 12, 16 (capped at logical core count).
- **Peak thread count** — where throughput is highest.
- **Hyperthreading penalty** — how much tok/s drops past the physical-core boundary.

Both compute (`-t`) and batch (`-tb`) threads are swept together. Backend: CPU-only
(`-ngl 0 --device none`), so no GPU is required.

## Run it

```bash
export SPECBENCH_MODELS_DIR=/path/to/models
export SPECBENCH_VULKAN_EXE=/path/to/llama-server
# SPECBENCH_MODEL_BASE defaults to Qwen2.5-Coder-3B-Instruct
# expects <base>-Q4_K_M.gguf in MODELS_DIR

python scripts/bench.py
python scripts/aggregate.py        # -> results/data.json
python scripts/inject.py           # bakes into site/index.html
```

Share your results by running:

```bash
python scripts/make_submission.py --name "my-cpu-label"
# then open a PR adding results/community/my-cpu-label.json
```

## Layout

```
scripts/  bench.py · aggregate.py · inject.py · make_submission.py
results/  results.json + data.json + community/  (logs gitignored)
site/     index.html — line chart tok/s vs threads (shows "results pending" until run)
```

Built on [`harness/`](../../harness/) (`LlamaServer` + `run_suite` + shared `prompts.py`).

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
