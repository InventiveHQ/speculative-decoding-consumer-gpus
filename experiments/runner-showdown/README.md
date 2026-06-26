# Runner Showdown — Ollama vs llama.cpp vs LM Studio

> Experiment #2 in the [local-llm-benchmarks](../../) series. Built on the shared
> [`harness/`](../../harness/) · contribute: [CONTRIBUTING.md](../../CONTRIBUTING.md)

**They're all llama.cpp underneath — so why is one slower?** LM Studio and Ollama
both run on llama.cpp (Ollama for GGUF models; LM Studio bundles llama.cpp, plus
MLX on Macs). So this isn't three engines racing — it's **raw llama.cpp (the floor)
vs two convenience layers on top**, and the question is what that convenience costs
in throughput. Lots of posts compare these on features; almost none measure the
speed tax. This one does.

The deltas come from three things, all of which the wrapper decides for you:
1. **which llama.cpp version** it bundles (often behind upstream),
2. **default flags** (flash-attention, KV-cache type, batch size, context, `-ngl`),
3. **wrapper overhead** (its HTTP server, prompt templating, scheduling).

We can't force Ollama/LM Studio onto a specific llama.cpp build — and that's the
point: their bundled version + defaults *are* part of what you're measuring. We hold
model + quant + prompts + sampling constant and report wall-clock tok/s.

## What it measures

Each runner serves the *identical* model + quant (Qwen2.5-Coder-7B, Q4) over its
OpenAI-compatible endpoint. We replay the shared 12-prompt suite (greedy, 256
tokens) against each and record **wall-clock tokens/sec** (`completion_tokens /
elapsed`) — the fair cross-runner metric, since it counts each wrapper's real
overhead (templating, HTTP, scheduling), not just the model's raw generation rate.
Output: throughput per runner, per task, and each runner's overhead vs the fastest.

## Run it — once per device

We test every device you can target (5060 Ti, 1080 Ti, CPU). For each device:
pin all three runners to it, then run `bench.py --device <label>` (it probes
whichever runners are reachable and tags the run). Repeat, then aggregate.

**Device pinning differs per runner** (because we only fully control llama.cpp):

| Runner | 5060 Ti | 1080 Ti | CPU |
|---|---|---|---|
| llama.cpp | `CUDA_VISIBLE_DEVICES=0 llama-server -ngl 999 -mg 0 --port 8080` | `CUDA_VISIBLE_DEVICES=1 …` | `llama-server -ngl 0 --device none …` |
| Ollama | `CUDA_VISIBLE_DEVICES=0 ollama serve` | `CUDA_VISIBLE_DEVICES=1 ollama serve` | `CUDA_VISIBLE_DEVICES= ollama serve` |
| LM Studio | GUI: select GPU / set GPU offload | GUI: select the 1080 Ti (may not be exposed) | GUI: GPU offload = 0 layers |

(Restart Ollama/llama-server between devices so the env takes effect. All three
must serve the **same** model + quant.)

```bash
python scripts/bench.py --device 5060ti     # -> results/results-5060ti.json
python scripts/bench.py --device 1080ti     # -> results/results-1080ti.json
python scripts/bench.py --device cpu        # -> results/results-cpu.json
python scripts/aggregate.py                 # -> results/data.json (runner x device matrix)
python scripts/inject.py                    # bakes into site/index.html
```

`bench.py` skips any runner not reachable on a device, so gaps are fine — a likely
one is **a wrapper's bundled llama.cpp lacking Blackwell `sm_120` support**, which
would show as `n/a` (or a CPU-fallback collapse) on the 5060 Ti. That gap is a
finding, not a bug. Endpoints/model ids: edit `RUNNERS` in `scripts/bench.py`,
`--config runners.json`, or `--only ollama,llama.cpp`.

## Layout

```
scripts/   bench.py (harness-based) · aggregate.py · inject.py · make_submission.py
results/   results.json + data.json + community/ submissions  (logs/screenshots gitignored)
site/      index.html — the writeup
```

Built on [`harness/`](../../harness/): `llama_client.py` (`chat`/`measure` against any
OpenAI-compatible endpoint, reporting wall-clock + server tok/s) and the shared
`prompts.py`.

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
