# Speculative Decoding: Draft Model Size Sweep

> Experiment #10 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**Bigger draft model = faster? Not always.** Speculative decoding uses a cheap draft
model to propose tokens that a larger target model verifies in one pass. A bigger draft
accepts more tokens per round — but costs more per draft step. There is a sweet spot.
This sweep holds the target fixed and measures actual speedup for three draft sizes
(0.5B, 1.5B, 3B) to find it. Sequel to [experiment #1](../speculative-decoding/).

Status: **scaffolded, results pending** (waiting on a hardware run).

## What it measures (per draft size)

- **Speedup** — draft tok/s / baseline tok/s (above 1.0× = faster than no speculation).
- **Raw tok/s** — absolute generation rate on the shared 12-prompt suite.
- **VRAM used** — nvidia-smi used-memory delta on model load.
- **Acceptance rate** — best-effort parse from server log (`draft acceptance = ...` lines).

## Run it

Download the target and any draft GGUFs into your models dir (`SPECBENCH_MODELS_DIR`):

| Role   | File                                          |
|--------|-----------------------------------------------|
| Target | `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf`    |
| Draft  | `Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf`     |
| Draft  | `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf`   |
| Draft  | `Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf`     |

Missing draft models are skipped automatically. Then:

```bash
python scripts/bench.py --gpu 0         # CUDA, GPU 0; drafts found are swept
python scripts/aggregate.py             # -> results/data.json
python scripts/inject.py                # bakes into site/index.html
```

Override defaults with env vars:

```bash
SPECBENCH_MODELS_DIR=/path/to/models \
SPECBENCH_CUDA_EXE=/path/to/llama-server.exe \
python scripts/bench.py
```

## Layout

```
scripts/  bench.py · aggregate.py · inject.py · make_submission.py
results/  results.json + data.json + community/  (logs gitignored)
site/     index.html — speedup chart + raw tok/s (shows "results pending" until run)
```

Built on [`harness/`](../../harness/) (`LlamaServer` + `run_suite` + shared `prompts.py`).

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
