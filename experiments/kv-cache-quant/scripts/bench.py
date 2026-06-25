"""KV-cache quantization sweep: fit more context in the same VRAM.

For a single fixed model (Q4_K_M quant), this sweeps three KV-cache types
(f16, q8_0, q4_0) at a large context window and measures:
  - VRAM used (nvidia-smi delta around load),
  - generation tok/s (shared 12-prompt suite),
  - greedy outputs per prompt (for quality/similarity comparison vs f16).

  python scripts/bench.py --backend cuda --gpu 0 --out results/results.json

Set SPECBENCH_MODELS_DIR to the folder containing your GGUF files, and
SPECBENCH_MODEL_BASE (default Qwen2.5-Coder-7B-Instruct) for the model name.
The script looks for <base>-Q4_K_M.gguf.
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS                       # noqa: E402
import llama_client as lc                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.environ.get("SPECBENCH_MODELS_DIR", os.path.join(ROOT, "models"))
EXE = {
    "cuda":   os.environ.get("SPECBENCH_CUDA_EXE",   os.path.join(ROOT, "runtimes", "cuda", "llama-server.exe")),
    "vulkan": os.environ.get("SPECBENCH_VULKAN_EXE", os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe")),
    "cpu":    os.environ.get("SPECBENCH_VULKAN_EXE", os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe")),
}
MAX_TOKENS = 256
MODEL_BASE = os.environ.get("SPECBENCH_MODEL_BASE", "Qwen2.5-Coder-7B-Instruct")

# KV-cache types to sweep: f16 is the baseline (full precision).
KV_TYPES = ["f16", "q8_0", "q4_0"]


def gpu_used_mb(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(idx)],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def bench_kv(kv_type, model_path, backend, gpu, ctx):
    print(f"\n=== KV type: {kv_type} ===", flush=True)
    before = gpu_used_mb(gpu) if backend != "cpu" else None
    srv = lc.LlamaServer(
        EXE[backend], model_path, backend=backend, gpu_index=gpu, ctx=ctx,
        extra_args=["-ctk", kv_type, "-ctv", kv_type],
        log_path=os.path.join(ROOT, "results", f"server-{kv_type}.log"))
    try:
        srv.start()
        time.sleep(1)
        after = gpu_used_mb(gpu) if backend != "cpu" else None
        vram = (after - before) if (before is not None and after is not None) else None
        try:
            lc.chat(srv.port, "Say hello.", max_tokens=16)
        except Exception:
            pass
        tps, outputs = [], {}
        for pr in PROMPTS:
            m = lc.measure(srv.port, pr["prompt"], max_tokens=MAX_TOKENS)
            if m["tps"]:
                tps.append(m["tps"])
            outputs[pr["id"]] = m["text"]
            print(f"  {pr['id']:<20} {round(m['tps'], 1) if m['tps'] else '—'} tok/s", flush=True)
        avg_tps = round(sum(tps) / len(tps), 2) if tps else None
        print(f"  vram: {vram} MiB  tok/s: {avg_tps}", flush=True)
        return {"label": kv_type, "vram_mb": vram, "tps": avg_tps, "outputs": outputs}
    finally:
        srv.stop()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda", choices=["cuda", "vulkan", "cpu"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    args = ap.parse_args()

    model_path = os.path.join(MODELS_DIR, f"{MODEL_BASE}-Q4_K_M.gguf")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}", flush=True)
        print(f"Set SPECBENCH_MODELS_DIR and/or SPECBENCH_MODEL_BASE.", flush=True)
        sys.exit(1)

    res = {
        "meta": {
            "model": MODEL_BASE,
            "quant": "Q4_K_M",
            "backend": args.backend,
            "gpu": args.gpu,
            "ctx": args.ctx,
            "max_tokens": MAX_TOKENS,
            "quality": "greedy output-similarity vs f16 KV cache (difflib ratio over suite)",
        },
        "configs": [],
    }
    for kv_type in KV_TYPES:
        r = bench_kv(kv_type, model_path, args.backend, args.gpu, args.ctx)
        if r:
            res["configs"].append(r)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
