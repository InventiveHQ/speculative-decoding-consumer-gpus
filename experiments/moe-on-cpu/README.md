# Experiment #12 — MoE on CPU: 13B-class answers at 3B speed

**Question:** a Mixture-of-Experts model has a huge total parameter count but only
activates a fraction per token. On a CPU — where single-user decoding is
memory-bandwidth-bound — does it run at the speed of its **total** size, or its
**active** size?

**Finding:** active size. A **Qwen3-30B-A3B** MoE (30.5B total, ~3.3B active per
token) decodes at roughly the same tok/s as a dense **3B** on the same CPU, while
answering like a model many times larger. You get big-model quality at small-model
CPU speed — the most practical reason to run MoE on a RAM-rich machine with no GPU.

> **Why not dFlash / speculation?** NVIDIA's dFlash (and EAGLE) get their headline
> multiples from a *trained, target-conditioned draft head* running in
> **TensorRT-LLM on Blackwell GPUs at high concurrency** — not something that exists
> in llama.cpp or on CPU, and there's no published head for this model. And per
> [experiment #1](../speculative-decoding/), at batch-1 on one device a generic draft
> is overhead-governed anyway. So this experiment is about the MoE *architecture's*
> own speed/quality tradeoff, measured honestly on the stack you can actually run.

## What it measures

For each model, **CPU only** (`-ngl 0`):
- generation **tok/s** (shared 12-prompt suite, `harness/prompts.py`)
- a **quality** pass-rate (graded math/reasoning set, exact-answer matching — model
  code is never executed)
- **active vs total** parameters, to plot speed against the right axis

Models (all Q4_K_M): dense 1.5B / 3B / 7B (Qwen2.5-Coder), dense 8B (Llama-3.1),
and the **30B-A3B MoE** (Qwen3-VL-30B-A3B-Instruct — its text decoder is the A3B MoE).

## Run it

```bash
# point at your GGUF dir(s); os.pathsep-separated (";" on Windows, ":" on POSIX)
export SPECBENCH_MODELS_DIR="/path/to/models"
export SPECBENCH_CPU_EXE="/path/to/llama-server"   # build must support qwen3moe
cd experiments/moe-on-cpu
python scripts/bench.py --threads 6 --out results/results.json
python scripts/aggregate.py && python scripts/inject.py
```

Missing models are skipped — even one MoE + one dense point is a useful data point.
A recent llama.cpp build is required (the MoE arch is new).

## Contribute

```bash
python scripts/make_submission.py --name "ryzen7950x-ddr5" --notes "DDR5-6000, llama.cpp bXXXX"
```

Opens a PR-ready `results/community/<name>.json` with your auto-captured rig. Memory
**bandwidth** is the lever here, so DDR5 / more channels should shift the whole curve
up — submissions across memory configs are especially welcome. See
[CONTRIBUTING.md](../../CONTRIBUTING.md).
