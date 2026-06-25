# Contributing your results

The whole point of this repo is to build a **community map** of where speculative
decoding helps (and hurts) across real consumer hardware. If you have a GPU — or
even just a CPU — your numbers are welcome.

## 1. Set up (one-time)

1. Download a llama.cpp release from
   [ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases).
   Grab the **CUDA** build and the **Vulkan** build for your OS, and unzip them to:
   - `runtimes/cuda/llama-server.exe`
   - `runtimes/vulkan/llama-server.exe`

   (On non-NVIDIA hardware, just use whichever backend you have. You can point at
   existing binaries with `SPECBENCH_CUDA_EXE` / `SPECBENCH_VULKAN_EXE` env vars.)
2. Download the models into `models/` (or set `SPECBENCH_MODELS_DIR`):
   - `Qwen2.5-Coder-7B-Instruct-Q4_K_S.gguf`  (target)
   - `Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf`  (draft)
   - optional: `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf`, `Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf`

   All are on Hugging Face (e.g. the `bartowski` GGUF repos). Same-family draft is
   required — the draft and target must share a tokenizer.

## 2. Run the benchmark

```bash
# Your main GPU (index 0), both backends, 3 methods
python scripts/bench.py --gpu 0 --backends cuda,vulkan --out results/results.json

# A second GPU? (index 1)
python scripts/bench.py --gpu 1 --backends cuda,vulkan --out results/results_1080ti.json

# A bigger model (fits if you have the VRAM)
python scripts/bench.py --gpu 0 --target qwen14b --backends cuda,vulkan --out results/results_14b.json

# CPU only (Windows: use the PowerShell runner — it's robust on long CPU requests)
powershell -ExecutionPolicy Bypass -File scripts/cpu_bench.ps1
```

Only run what you can — a single `results/results.json` is a perfectly good
contribution.

## 3. Bundle + submit

```bash
python scripts/make_submission.py --name "rtx4090-ryzen7950x" --notes "llama.cpp b9999, Q4_K_M"
```

That writes `results/community/<name>.json` with your hardware info and results.
Open a **pull request** adding just that file. We'll merge it and (over time) add
a community comparison view.

By submitting you agree to license your data under **CC BY 4.0** (see
[DATA-LICENSE.md](DATA-LICENSE.md)).

## What makes a good submission

- Note your **llama.cpp version** and exact **quant** in `--notes`.
- Keep the default methodology (greedy, 256 tokens, the bundled 12-prompt suite)
  so results are comparable. If you change something, say so in the notes.
- The harness records the server's `draft acceptance` lines in
  `results/server-*.log` — handy if you want to discuss *why* you saw what you saw.
