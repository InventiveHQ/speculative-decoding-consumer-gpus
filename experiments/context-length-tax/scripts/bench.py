"""Context-length sweep: tok/s, VRAM, and TTFT proxy as declared context grows.

One model, one quant (Q4_K_M), backend cuda. We launch a fresh llama-server for each
context-size setting and measure:
  - generation tok/s (shared 12-prompt suite),
  - VRAM used (nvidia-smi delta on load), and
  - prompt-eval latency a.k.a. TTFT proxy (resp["timings"]["prompt_ms"]).

The story: bigger declared context costs throughput and VRAM even when you don't fill
it — this sweep quantifies the % drop and GB cost from 2K to 32K.

  python scripts/bench.py --backend cuda --gpu 0 --out results/results.json

Set SPECBENCH_MODELS_DIR to your models directory and place
<MODEL_BASE>-Q4_K_M.gguf there. SPECBENCH_CUDA_EXE / SPECBENCH_VULKAN_EXE override
the llama-server binary path.
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS                       # noqa: E402
import llama_client as lc                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.environ.get("SPECBENCH_MODELS_DIR", os.path.join(ROOT, "models"))
EXE = {
    "cuda":   os.environ.get("SPECBENCH_CUDA_EXE",   os.path.join(ROOT, "runtimes", "cuda",   "llama-server.exe")),
    "vulkan": os.environ.get("SPECBENCH_VULKAN_EXE", os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe")),
    "cpu":    os.environ.get("SPECBENCH_VULKAN_EXE", os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe")),
}
MAX_TOKENS = 256
MODEL_BASE = os.environ.get("SPECBENCH_MODEL_BASE", "Qwen2.5-Coder-7B-Instruct")
QUANT = "Q4_K_M"

# Context sizes to sweep (tokens).
CTX_SIZES = [2048, 4096, 8192, 16384, 32768]

# A ~200-token prompt used to measure prompt-evaluation latency (TTFT proxy).
# Longer prompt = more signal; keep well under the smallest ctx (2048).
TTFT_PROMPT = (
    "Read the following technical passage carefully, then write a concise two-sentence "
    "summary focusing on the architectural implications for inference systems. "
    "The transformer architecture, introduced in 2017, replaced recurrent networks for "
    "most sequence tasks by relying entirely on attention. Self-attention lets every "
    "token attend to every other token in the sequence, which removes the sequential "
    "dependency that made RNNs slow to train. Multi-head attention runs several "
    "attention operations in parallel, each learning a different relationship, and "
    "their outputs are concatenated and projected. Positional encodings inject order "
    "information because attention itself is permutation-invariant. The encoder-decoder "
    "design was later split into encoder-only models for understanding and decoder-only "
    "models for generation. Decoder-only models generate one token at a time, feeding "
    "each output back as input, which is why inference is memory-bound and latency-bound "
    "rather than compute-bound at small batch sizes. Modern deployment at scale requires "
    "batching many requests together, using KV-cache compression, and optimizing attention "
    "kernels such as FlashAttention. Each of these techniques trades some form of memory "
    "for speed, or accuracy for throughput, making hardware-software co-design critically "
    "important for production LLM serving."
)


def gpu_used_mb(idx):
    """Return current GPU memory used in MiB, or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(idx)],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def bench_ctx(c, backend, gpu, model_path):
    """Launch a server with context size c and record tps, vram_mb, ttft_ms."""
    print(f"\n=== ctx={c} ===", flush=True)
    before = gpu_used_mb(gpu) if backend != "cpu" else None
    srv = lc.LlamaServer(
        EXE[backend], model_path, backend=backend, gpu_index=gpu, ctx=c,
        log_path=os.path.join(ROOT, "results", f"server-ctx{c}.log"))
    try:
        srv.start()
        time.sleep(1)
        after = gpu_used_mb(gpu) if backend != "cpu" else None
        vram = (after - before) if (before is not None and after is not None) else None

        # Warmup pass (not measured)
        try:
            lc.chat(srv.port, "Say hello.", max_tokens=16)
        except Exception:
            pass

        # TTFT proxy: prompt-eval time for a fixed long prompt.
        ttft_ms = None
        try:
            resp = lc.chat(srv.port, TTFT_PROMPT, max_tokens=32)
            ttft_ms = (resp.get("timings") or {}).get("prompt_ms")
            if ttft_ms is not None:
                ttft_ms = round(ttft_ms, 1)
        except Exception as e:
            print(f"  TTFT measurement failed: {e}", flush=True)

        # Generation throughput: shared 12-prompt suite (warmup already done above).
        records = lc.run_suite(srv.port, PROMPTS, max_tokens=MAX_TOKENS, warmup=False)
        tps_vals = [r["tps"] for r in records if r.get("tps") is not None]
        avg_tps = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None

        for r in records:
            tok = round(r["tps"], 1) if r.get("tps") else "—"
            print(f"  {r['id']:<22} {tok} tok/s", flush=True)
        print(f"  avg tps={avg_tps}  vram_delta={vram} MiB  ttft={ttft_ms} ms", flush=True)

        return {"ctx": c, "tps": avg_tps, "vram_mb": vram, "ttft_ms": ttft_ms}
    finally:
        srv.stop()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser(
        description="Sweep context sizes and measure tok/s, VRAM, and TTFT.")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "vulkan", "cpu"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    args = ap.parse_args()

    model_path = os.path.join(MODELS_DIR, f"{MODEL_BASE}-{QUANT}.gguf")
    if not os.path.exists(model_path):
        sys.exit(
            f"Model not found: {model_path}\n"
            f"Set SPECBENCH_MODELS_DIR to the directory containing "
            f"{MODEL_BASE}-{QUANT}.gguf")

    res = {
        "meta": {
            "model": MODEL_BASE, "quant": QUANT,
            "backend": args.backend, "gpu": args.gpu,
            "max_tokens": MAX_TOKENS, "ctx_sizes": CTX_SIZES,
        },
        "points": [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for c in CTX_SIZES:
        pt = bench_ctx(c, args.backend, args.gpu, model_path)
        if pt:
            res["points"].append(pt)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
