# Local LLM Benchmarks

Reproducible, honest benchmarks of local LLM inference on **hardware you can
actually buy** — by [InventiveHQ](https://inventivehq.com) and contributors. Every
experiment ships its harness, raw results, and a self-contained writeup, and takes
community submissions so the numbers grow across real-world hardware.

The throughline: vendor blogs benchmark on 8×H100s and report best-case numbers.
We run the same ideas on consumer GPUs and CPUs, and publish what actually happens
— including the times it backfires.

> 👉 **Start here:** [`index.html`](index.html) is the **findings roundup** — all 12 results on one page (open it locally, or it's the GitHub Pages landing page).

## Experiments

| # | Experiment | The hook | Status |
|---|---|---|---|
| 1 | [Speculative decoding](experiments/speculative-decoding/) | NVIDIA's 15× dFlash trick on a $570 GPU (and an 8-year-old one, and a CPU) — where it helps, where it backfires | ✅ Published |
| 2 | [Runner showdown](experiments/runner-showdown/) (Ollama vs llama.cpp vs LM Studio) | Same llama.cpp engine: **LM Studio ≈ raw (0.3%), Ollama ~10% slower** | ✅ Results (RTX 5060 Ti) |
| 3 | [GGUF quantization](experiments/quantization-quality/): speed vs quality | Output fidelity vs Q8: **Q2 21% → Q4 57% → Q6 80%** — the cliff is below Q4; sweet spot Q4–Q5 | ✅ Results (RTX 5060 Ti) |
| 4 | [KV-cache quantization](experiments/kv-cache-quant/) | q8_0 KV ≈ lossless (82% similar); **q4_0 KV wrecks output quality (8%)** | ✅ Results (RTX 5060 Ti) |
| 5 | [The VRAM cliff](experiments/vram-offload-cliff/) (`-ngl` sweep) | **2.9 → 43 tok/s** as a 14B moves to GPU — the last layers matter most | ✅ Results (RTX 5060 Ti) |
| 6 | [Context-length tax](experiments/context-length-tax/) | 2K→32K: **~0 tok/s cost, +1.7 GB VRAM** (the tax is memory, not speed) | ✅ Results (RTX 5060 Ti) |
| 7 | [Flash attention](experiments/flash-attention/) | **`-fa` was a no-op** — identical speed & VRAM (already default-on this build) | ✅ Results (RTX 5060 Ti) |
| 8 | [Tokens per watt](experiments/tokens-per-watt/) | **−17% power → ~0 speed lost, +16% efficiency** | ✅ Results (RTX 5060 Ti) |
| 9 | [CPU thread scaling](experiments/cpu-thread-scaling/) | Peaks near the **6 physical cores**; 12 threads slightly slower than 8 | ✅ Results (i7-8700) |
| 10 | [Draft-model size sweep](experiments/draft-size-sweep/) | **0.5B draft wins (1.37×)** despite lowest acceptance — bigger ≠ faster | ✅ Results (RTX 5060 Ti) |
| 11 | [Smallest model that can code](experiments/smallest-model-that-can-code/) | Graded pass-rate **1→6 of 10** from 0.5B→14B — need ≥7B to reason | ✅ Results (RTX 5060 Ti) |
| 12 | [MoE on CPU](experiments/moe-on-cpu/) | A **30B-A3B MoE runs at 3B-dense speed** on an i7-8700 (10.6 tok/s, 9.2× faster than a dense 30B) — and scores **8/10 vs the 3B's 1/10** | ✅ Results (i7-8700 CPU) |

## Repo layout

```
harness/                     shared engine — prompt suite + llama-server client/runner
experiments/
  speculative-decoding/       experiment #1: scripts/ results/ site/ README.md
  <next>/                     each new experiment is a folder like this
CONTRIBUTING.md  LICENSE  DATA-LICENSE.md   (apply to the whole series)
```

## Contribute your hardware's numbers

Every experiment has a `results/community/` inbox. Run the suite, bundle your
results, open a PR — CI validates it and rebuilds the experiment's page
automatically. See **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## License

- **Code** (`harness/`, `experiments/**/scripts`, site markup/JS): [MIT](LICENSE)
- **Data & writeups** (`experiments/**/results`, article content): [CC BY 4.0](DATA-LICENSE.md)

---

> This repo began as `speculative-decoding-consumer-gpus` and was renamed to
> `local-llm-benchmarks` to host the full series — old links redirect automatically.
