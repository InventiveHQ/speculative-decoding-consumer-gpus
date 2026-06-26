# Tokens per watt: cap your GPU's power, keep the speed?

> Experiment #8 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**Tokens per watt: I capped my GPU to 70% power and barely lost speed.** LLM decoding
is memory-bandwidth-bound, not compute-bound. That means cutting the GPU's power
limit often costs almost nothing in tok/s while shrinking heat, noise, and electricity
draw. This experiment measures exactly where the trade-off lives.

> **Hardware notice:** `bench.py` modifies GPU power limits via `nvidia-smi -pl`.
> This requires **administrator/elevated privileges** (Windows: run from an Admin
> prompt; Linux: run with sudo). The script always restores the original power limit
> in a `finally` block. If it is killed ungracefully, restore manually:
> `nvidia-smi -pl <default_watts> -i <gpu>`.

## What it measures

Fixed model (`Qwen2.5-Coder-7B-Instruct Q4_K_M`), CUDA backend. For each power-limit
step (100 → 90 → 80 → 70 → 60 → 50% of the GPU's default TDP):

- **tok/s** — generation throughput on the shared 12-prompt suite.
- **avg_power_w** — mean GPU power draw sampled every 0.5 s during the full suite run.
- **tok/s / W** — the efficiency figure: tokens per watt of actual draw.

The GPU's default power limit is queried automatically; steps are clamped to the
hardware minimum. Duplicate watt values (from rounding) are skipped.

## Run it

```bash
# Set env vars (or rely on defaults)
export SPECBENCH_MODELS_DIR=/path/to/models
export SPECBENCH_CUDA_EXE=/path/to/llama-server
export SPECBENCH_MODEL_BASE=Qwen2.5-Coder-7B-Instruct   # optional

# Windows — run from an Administrator command prompt
python scripts\bench.py --gpu 0
python scripts\aggregate.py
python scripts\inject.py
```

Missing model: the script expects `<SPECBENCH_MODELS_DIR>/<MODEL_BASE>-Q4_K_M.gguf`.
Download it from e.g. [bartowski's Qwen2.5-Coder-7B-Instruct GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF).

## Layout

```
scripts/  bench.py · aggregate.py · inject.py · make_submission.py
results/  results.json + data.json + community/  (logs gitignored)
site/     index.html — tok/s + tok/s/W charts
```

Built on [`harness/`](../../harness/) (`LlamaServer` + `measure` + shared `prompts.py`).

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
