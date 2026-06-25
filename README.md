# Local LLM Benchmarks

Reproducible, honest benchmarks of local LLM inference on **hardware you can
actually buy** — by [InventiveHQ](https://inventivehq.com) and contributors. Every
experiment ships its harness, raw results, and a self-contained writeup, and takes
community submissions so the numbers grow across real-world hardware.

The throughline: vendor blogs benchmark on 8×H100s and report best-case numbers.
We run the same ideas on consumer GPUs and CPUs, and publish what actually happens
— including the times it backfires.

## Experiments

| # | Experiment | The hook | Status |
|---|---|---|---|
| 1 | [Speculative decoding](experiments/speculative-decoding/) | NVIDIA's 15× dFlash trick on a $570 GPU (and an 8-year-old one, and a CPU) — where it helps, where it backfires | ✅ Published |
| 2 | [Runner showdown](experiments/runner-showdown/) (Ollama vs llama.cpp vs LM Studio) | "Which wrapper is fastest — and what does the convenience cost you?" | 🚧 Scaffolded — results pending |
| 3 | [GGUF quantization](experiments/quantization-quality/): speed vs quality | "Q2 → Q8 — where does quality fall off a cliff?" | 🚧 Scaffolded — results pending |
| 4 | [KV-cache quantization](experiments/kv-cache-quant/) | "f16 vs q8_0 vs q4_0 KV cache — fit more context per GB, does quality hold?" | 🚧 Scaffolded — results pending |
| 5 | [The VRAM cliff](experiments/vram-offload-cliff/) (`-ngl` sweep) | "What actually happens when your model doesn't fit in VRAM" | 🚧 Scaffolded — results pending |
| 6 | [Context-length tax](experiments/context-length-tax/) | "What 2K→32K context costs in throughput and VRAM" | 🚧 Scaffolded — results pending |
| 7 | [Flash attention](experiments/flash-attention/) | "Is `-fa` actually free on a consumer GPU?" | 🚧 Scaffolded — results pending |
| 8 | [Tokens per watt](experiments/tokens-per-watt/) | "Cap your GPU's power — keep the speed?" | 🚧 Scaffolded — results pending |
| 9 | [CPU thread scaling](experiments/cpu-thread-scaling/) | "More CPU threads made my LLM slower" | 🚧 Scaffolded — results pending |
| 10 | [Draft-model size sweep](experiments/draft-size-sweep/) | "Bigger draft model = faster? Not always." | 🚧 Scaffolded — results pending |
| 11 | [Smallest model that can code](experiments/smallest-model-that-can-code/) | "How small a model can actually solve problems?" | 🚧 Scaffolded — results pending |

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
