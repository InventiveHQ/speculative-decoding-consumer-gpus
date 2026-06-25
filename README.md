# Speculative Decoding on Consumer GPUs

**How much of NVIDIA's datacenter speculative-decoding speedup actually survives on
hardware you can buy?** This repo is the reproducible benchmark behind the
[InventiveHQ](https://inventivehq.com) article — and an open, growing map of where
speculative decoding helps (and hurts) across real consumer GPUs and CPUs.

NVIDIA's [dFlash post](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/)
shows up to **15× higher throughput** on eight DGX B300s. We ran the same core
idea — speculative decoding — on one Windows box, across three methods (baseline,
draft-model, model-free n-gram), two llama.cpp backends (CUDA + Vulkan), greedy
decoding, 256 tokens/prompt, over a 12-prompt suite (code / math / reasoning /
summarization / chat).

> 📈 **Got a GPU? Add your numbers.** Running the suite takes a few minutes — see
> [CONTRIBUTING.md](CONTRIBUTING.md). PRs adding `results/community/*.json` welcome.

## Hardware we've measured so far

| Device | Arch | Baseline (7B) | Best speculative win |
|---|---|---|---|
| RTX 5060 Ti (16 GB, 2025, ~$570) | Blackwell sm_120 | ~85 tok/s | 14B math **1.85×** (Vulkan/CUDA); 7B CUDA draft **collapses to 0.27×** |
| GTX 1080 Ti (11 GB, 2017) | Pascal sm_61 | ~50–57 tok/s | 7B math **1.16×** (CUDA draft) |
| CPU (3B, Core i7-8700) | — | ~13 tok/s | **2.0×** on math (draft) |

## Headline findings

1. **Speculation has overhead; it can backfire on a fast model.** On Blackwell's
   quick 7B (~85 tok/s), the stock CUDA build's draft path *collapsed* to ~0.27×
   of baseline (a 3–4× slowdown) at healthy acceptance. The **same CUDA build,
   same card**, on a slower 14B hit ~1.85× on math — the heavier model amortizes
   the per-cycle overhead. Pascal's slower 7B (~50 tok/s) never collapsed either.
   The **Vulkan** build has much lower speculative overhead and stayed ≥ baseline
   where CUDA cratered → on a fast small model, prefer Vulkan or n-gram.
2. **Task-dependent**, exactly like NVIDIA's SPEED-Bench shape: code/math win,
   open-ended chat loses (the draft nails structured output, flails on prose).
3. **N-gram is the free safety net** (~1.0× everywhere, no extra model/RAM).
4. **Memory-bound decoding narrows the generation gap**: the 2017 1080 Ti is only
   ~1.5× behind the 2025 5060 Ti at batch 1.
5. **Bigger model = bigger win, and it fits**: a full-quality 14B (Q4_K_M) + draft
   fits 16 GB (~10 GB at 4K ctx), and speculation helps it *more* than the 7B —
   the consumer echo of why dFlash targets a 120B.
6. **Works on CPU too** (Qwen2.5-Coder 3B, `-ngl 0 --device none`): the slowest
   target of all, where speculation pays off most on code/math.

Every speculative run's output hash matched its baseline → **lossless** (it
changes speed, not answers).

## Quickstart

```bash
# 1. Put llama.cpp binaries in runtimes/{cuda,vulkan}/llama-server.exe
#    (or point at your own with SPECBENCH_CUDA_EXE / SPECBENCH_VULKAN_EXE)
# 2. Put the GGUF models in models/  (or set SPECBENCH_MODELS_DIR)  — see CONTRIBUTING.md
# 3. Run:
python scripts/bench.py --gpu 0 --backends cuda,vulkan --out results/results.json
python scripts/aggregate.py        # -> results/data.json
python scripts/inject.py           # bakes results into site/index.html
# open site/index.html
```

Full instructions (downloads, CPU runs, bigger models, submitting results) are in
**[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Layout

```
scripts/
  prompts.py        the 12-prompt suite (5 task categories)
  bench.py          GPU runs: launches llama-server per (backend x method), records tok/s
  cpu_bench.ps1     CPU run (PowerShell HTTP client; robust on long CPU requests)
  aggregate.py      -> results/data.json (GPUs + bigmodel + cpu)
  inject.py         injects data.json into site/index.html
  make_submission.py  bundles your results + hardware -> results/community/<name>.json
results/            raw results_*.json + community/ submissions  (logs/screenshots gitignored)
site/index.html     the self-contained article (hand-coded animated SVG explainer + charts)
models/, runtimes/  large files — NOT in git; download them yourself (see CONTRIBUTING.md)
```

## Methodology

- One device per run, all layers on it (`--split-mode none -mg 0`; CPU uses
  `-ngl 0 --device none`), context 4096, greedy (temp 0, top-k 1), 256 generated
  tokens/prompt, the bundled 12-prompt suite.
- Throughput = the server's `timings.predicted_per_second` (generation only).
  Acceptance rates are in `results/server-*.log` (`draft acceptance = …`).
- GPU runs use `bench.py` (Python). The CPU run uses `cpu_bench.ps1` because
  Python's urllib intermittently dropped long CPU requests; both emit the same
  `results_*.json` schema.
- Reference binaries: llama.cpp release `b9789` (CUDA 12.4) + a recent Vulkan
  build. The CUDA-on-fast-model behaviour is version-specific — **re-test on newer
  releases** and send a PR if it changes.

## License

- **Code** (`scripts/`, `site/` markup & JS): [MIT](LICENSE).
- **Data & article** (`results/`, `site/index.html` content):
  [CC BY 4.0](DATA-LICENSE.md).
