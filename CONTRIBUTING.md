# Contributing your results

This repo is a **series** of local-LLM benchmarks, and the whole point is to build
a community map of how each one behaves across real hardware. Your numbers — GPU or
CPU — are welcome on any experiment.

Each experiment lives in `experiments/<name>/` with its own `scripts/`, `results/`,
and `results/community/` inbox. The example below uses experiment #1,
[`speculative-decoding`](experiments/speculative-decoding/); the flow is the same
for the others.

## 1. Set up (one-time)

1. Download llama.cpp binaries from
   [ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)
   — the **CUDA** and **Vulkan** builds for your OS. Point the harness at them with
   env vars (so you can reuse one copy across experiments):
   ```
   SPECBENCH_CUDA_EXE   = .../cuda/llama-server(.exe)
   SPECBENCH_VULKAN_EXE = .../vulkan/llama-server(.exe)
   SPECBENCH_MODELS_DIR = .../your/models      # where the .gguf files live
   ```
   (On non-NVIDIA hardware, use whichever backend you have.)
2. Download the models named in the experiment's README into `SPECBENCH_MODELS_DIR`.
   For #1: Qwen2.5-Coder 7B Q4_K_S (target) + 0.5B Q8_0 (draft); optional 14B/3B.
   A same-family draft (shared tokenizer) is required.

## 2. Run it

```bash
cd experiments/speculative-decoding
python scripts/bench.py --gpu 0 --backends cuda,vulkan --out results/results.json
# optional: second GPU, bigger model, CPU — see this experiment's README
```

Run only what you can — a single `results/results.json` is a great contribution.

## 3. Bundle + submit

```bash
python scripts/make_submission.py --name "rtx4090-ryzen7950x" --notes "llama.cpp b9999, Q4_K_M"
```

That writes `results/community/<name>.json` with your results plus an **automatically
captured hardware snapshot** — CPU (model + core/thread count), RAM (size, speed,
type), every GPU with its **PCIe link gen × width**, motherboard, and storage type.
It's detected cross-platform by [`harness/hwinfo.py`](harness/hwinfo.py) (Windows /
Linux / macOS), so you don't fill it in by hand. Skim the `"hardware"` block before
you open the PR — it's plain JSON; delete any field you'd rather not share. Open a
**pull request** adding just that file. CI validates it and rebuilds the experiment's
page automatically on merge.

By submitting you agree to license your data under **CC BY 4.0** (see
[DATA-LICENSE.md](DATA-LICENSE.md)).

## Good-submission checklist

- Note your **llama.cpp version** and exact **quant** in `--notes`.
- Keep the default methodology (greedy, 256 tokens, the bundled 12-prompt suite from
  `harness/prompts.py`) so results stay comparable — flag any deviations in notes.
- Acceptance rates and server logs land in `results/server-*.log` if you want to
  dig into *why* you saw what you saw.

## Adding a whole new experiment?

Build it on `harness/` (see [harness/README.md](harness/README.md)) as
`experiments/<name>/` with `scripts/` + `results/` + `site/` + a `README.md`, then
open a PR. Reusing `harness/prompts.py` and `harness/llama_client.py` keeps every
experiment comparable and the CI rebuild automatic.
