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

Status: **scaffolded, results pending** (waiting on a hardware run).

## What it measures

Each runner serves the *identical* model + quant (Qwen2.5-Coder-7B, Q4) over its
OpenAI-compatible endpoint. We replay the shared 12-prompt suite (greedy, 256
tokens) against each and record **wall-clock tokens/sec** (`completion_tokens /
elapsed`) — the fair cross-runner metric, since it counts each wrapper's real
overhead (templating, HTTP, scheduling), not just the model's raw generation rate.
Output: throughput per runner, per task, and each runner's overhead vs the fastest.

## Run it

Start all three with the *same* model, then benchmark (from this folder):

```bash
llama-server -m Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf -ngl 999 --port 8080 --jinja
ollama serve                      # first time: ollama pull qwen2.5-coder:7b
# LM Studio: load the same model, enable its local server (port 1234)

python scripts/bench.py --out results/results.json   # skips any runner that isn't up
python scripts/aggregate.py        # -> results/data.json
python scripts/inject.py           # bakes into site/index.html
```

Endpoints/model ids are configurable: edit the `RUNNERS` list in `scripts/bench.py`,
pass `--config runners.json`, or `--only ollama,llama.cpp`.

## Layout

```
scripts/   bench.py (harness-based) · aggregate.py · inject.py · make_submission.py
results/   results.json + data.json + community/ submissions  (logs/screenshots gitignored)
site/      index.html — the writeup (shows "results pending" until data lands)
```

Built on [`harness/`](../../harness/): `llama_client.py` (`chat`/`measure` against any
OpenAI-compatible endpoint, reporting wall-clock + server tok/s) and the shared
`prompts.py`.

## License

Code [MIT](../../LICENSE); data & writeup [CC BY 4.0](../../DATA-LICENSE.md).
