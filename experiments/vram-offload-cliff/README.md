# The VRAM Cliff: What Happens When Your Model Doesn't Fit

> Experiment #5 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**Each layer left on the CPU drags throughput down a cliff.** This sweeps `-ngl`
(GPU-offloaded layers) from 0 (all CPU) to 99 (all GPU) on a fixed model and quant,
measuring tok/s and VRAM at every step to find exactly where the cliff is.

## What it measures

- **Speed** — generation tok/s on the shared 12-prompt suite, per `-ngl` setting.
- **VRAM** — GPU memory used (nvidia-smi delta after model load).

The sweep uses a model large enough (Qwen2.5-Coder-14B Q4_K_M) that low `-ngl`
values genuinely spill layers to CPU, making the cliff visible and dramatic.

## Run it

Place `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` (or set `SPECBENCH_MODEL_BASE`) in
your models dir (`SPECBENCH_MODELS_DIR`), then:

```bash
python scripts/bench.py --backend cuda --gpu 0
python scripts/aggregate.py        # -> results/data.json
python scripts/inject.py           # bakes into site/index.html
```

Override the default model via `SPECBENCH_MODEL_BASE`; override the exe via
`SPECBENCH_CUDA_EXE`. Run on an otherwise-idle GPU for a clean VRAM delta.

## Layout

```
scripts/  bench.py · aggregate.py · inject.py · make_submission.py
results/  results.json + data.json + community/  (logs gitignored)
site/     index.html — tok/s vs ngl + VRAM vs ngl
```

Built on [`harness/`](../../harness/) (`LlamaServer` + `run_suite` + shared `prompts.py`).

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
