# Flash Attention: is -fa actually free?

> Experiment #7 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**Does llama.cpp's `-fa` flag actually help on a consumer GPU?** Flash attention
promises lower KV-cache memory and sometimes faster generation — but the gains depend
on context length and hardware. This experiment sweeps `4096 / 16384 / 32768` context
lengths with flash-attention off and on, measures generation tok/s and VRAM used, and
shows exactly where -fa helps vs where it's a wash.

Status: **scaffolded, results pending** (waiting on a hardware run).

## What it measures (per context × fa cell)

- **Speed** — mean generation tok/s on the shared 12-prompt suite.
- **VRAM used** — nvidia-smi delta on model load (approximate).
- **VRAM saved** — VRAM(fa=off) − VRAM(fa=on) at each context length.

The story: at short context, flash-attention often makes little difference. At long
context (16K–32K) the fused kernel can cut KV-cache memory noticeably and give a
measurable speed boost. We measure the real tradeoff so you know when `-fa` is worth
enabling.

## Run it

Place your model in `SPECBENCH_MODELS_DIR` as `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf`
(or set `SPECBENCH_MODEL_BASE` to a different base name). Then:

```bash
python scripts/bench.py --gpu 0          # sweeps 6 cells (3 ctx x fa off/on)
python scripts/aggregate.py              # -> results/data.json
python scripts/inject.py                 # bakes into site/index.html
```

Key env vars (mirror the rest of the series):

| Variable | Default | Purpose |
|---|---|---|
| `SPECBENCH_MODELS_DIR` | `models/` | dir containing the .gguf |
| `SPECBENCH_MODEL_BASE` | `Qwen2.5-Coder-7B-Instruct` | model name without quant suffix |
| `SPECBENCH_CUDA_EXE` | `runtimes/cuda/llama-server.exe` | path to llama-server |

> **Note on the flag:** the llama.cpp flash-attn argument has been `-fa` /
> `--flash-attn`, historically boolean and in recent builds taking `on/off/auto`.
> Check your build's `--help` output and adjust `bench.py`'s `extra_args` if needed.

Run on an otherwise-idle GPU so the VRAM delta is clean.

## Layout

```
scripts/  bench.py · aggregate.py · inject.py · make_submission.py
results/  results.json + data.json + community/  (logs gitignored)
site/     index.html — grouped bar charts: tok/s + VRAM saved (shows pending until run)
```

Built on [`harness/`](../../harness/) (`LlamaServer` + `measure` + shared `prompts.py`).

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
