# The context-length tax: what 2K→32K costs you

> Experiment #6 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**Bigger declared context costs throughput and VRAM even when you don't fill it.**
This sweep fixes one model and quant (Qwen2.5-Coder-7B-Instruct Q4_K_M) and varies
only the context-window setting from 2 048 → 32 768 tokens, measuring the speed,
memory, and latency penalty at each level.

## What it measures (per context size)

- **Speed** — generation tok/s on the shared 12-prompt suite.
- **VRAM** — nvidia-smi used-memory delta between idle and loaded (the KV-cache
  reservation is the dominant contributor; it scales with context).
- **TTFT proxy** — `timings.prompt_ms` from a fixed ~200-token prompt (prompt-eval
  latency; rises with context because the attention mask grows).

## Why it matters

llama.cpp (and most runtimes) allocate the full KV-cache up front based on the
declared `-c` value, regardless of how long your actual prompts are. Every extra
kilotoken of headroom is VRAM you pay for immediately and potentially throughput
you lose due to memory pressure. This experiment makes that cost concrete.

## Run it

Place `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` in your models directory, set
`SPECBENCH_MODELS_DIR`, then:

```bash
python scripts/bench.py --backend cuda --gpu 0
python scripts/aggregate.py        # -> results/data.json
python scripts/inject.py           # bakes into site/index.html
```

Optional env vars: `SPECBENCH_CUDA_EXE`, `SPECBENCH_VULKAN_EXE`,
`SPECBENCH_MODEL_BASE` (default `Qwen2.5-Coder-7B-Instruct`). Run on an otherwise
idle GPU so the VRAM delta is clean.

## Layout

```
scripts/  bench.py · aggregate.py · inject.py · make_submission.py
results/  results.json + data.json + community/  (logs gitignored)
site/     index.html — tok/s + VRAM + TTFT charts
```

Built on [`harness/`](../../harness/) (`LlamaServer` + `measure` + shared `prompts.py`).

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
