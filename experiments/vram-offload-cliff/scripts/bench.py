"""VRAM offload cliff: what happens when your model doesn't fit on GPU.

Sweeps -ngl (GPU-offloaded layers) from 0 to 99 on a single fixed model+quant.
For each setting, measures generation tok/s (shared 12-prompt suite) and VRAM used
(nvidia-smi delta), revealing the cliff where partial CPU offload collapses throughput.

  python scripts/bench.py --backend cuda --gpu 0 --out results/results.json

Place your model in SPECBENCH_MODELS_DIR, named:
  <SPECBENCH_MODEL_BASE>-Q4_K_M.gguf
Default model: Qwen2.5-Coder-14B-Instruct (large enough that low -ngl spills to CPU).
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS  # noqa: E402
import llama_client as lc    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.environ.get("SPECBENCH_MODELS_DIR", os.path.join(ROOT, "models"))
EXE = {
    "cuda":   os.environ.get("SPECBENCH_CUDA_EXE",   os.path.join(ROOT, "runtimes", "cuda",   "llama-server.exe")),
    "vulkan": os.environ.get("SPECBENCH_VULKAN_EXE",  os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe")),
    "cpu":    os.environ.get("SPECBENCH_VULKAN_EXE",  os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe")),
}
MAX_TOKENS = 256
CTX = int(os.environ.get("SPECBENCH_CTX", "4096"))
MODEL_BASE = os.environ.get("SPECBENCH_MODEL_BASE", "Qwen2.5-Coder-14B-Instruct")

# Layers to offload to GPU: 0 = all CPU, 99 = all GPU.
NGL_SWEEP = [0, 8, 16, 24, 32, 40, 99]


def gpu_used_mb(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(idx)],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def wait_vram_free(gpu, max_used=1800, timeout=40):
    """Block until the prior server's VRAM is actually released (avoids a partial
    offload on the next launch from stale allocations)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        u = gpu_used_mb(gpu)
        if u is None or u <= max_used:
            return
        time.sleep(2)


def bench_ngl(ngl, path, backend, gpu):
    print(f"\n=== ngl={ngl} ({'all GPU' if ngl >= 99 else 'partial CPU offload' if ngl > 0 else 'all CPU'}) ===", flush=True)
    if backend != "cpu":
        wait_vram_free(gpu)
    before = gpu_used_mb(gpu) if backend != "cpu" else None
    log_path = os.path.join(ROOT, "results", f"server-ngl{ngl}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    srv = lc.LlamaServer(EXE[backend], path, backend=backend, gpu_index=gpu,
                         ctx=CTX, ngl=ngl, log_path=log_path)
    try:
        srv.start()
        time.sleep(1)
        after = gpu_used_mb(gpu) if backend != "cpu" else None
        vram = (after - before) if (before is not None and after is not None) else None
        records = lc.run_suite(srv.port, PROMPTS, max_tokens=MAX_TOKENS)
        tps_list = [r["tps"] for r in records if r.get("tps") is not None]
        avg_tps = round(sum(tps_list) / len(tps_list), 2) if tps_list else None
        print(f"  vram: {vram} MiB  tok/s: {avg_tps}", flush=True)
        return {"ngl": ngl, "tps": avg_tps, "vram_mb": vram}
    finally:
        srv.stop()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda", choices=["cuda", "vulkan", "cpu"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    args = ap.parse_args()

    path = os.path.join(MODELS_DIR, f"{MODEL_BASE}-Q4_K_M.gguf")
    if not os.path.exists(path):
        sys.exit(f"Model not found: {path}\nSet SPECBENCH_MODELS_DIR or SPECBENCH_MODEL_BASE.")

    res = {"meta": {"model": MODEL_BASE, "quant": "Q4_K_M", "backend": args.backend,
                    "gpu": args.gpu, "ctx": CTX, "max_tokens": MAX_TOKENS,
                    "ngl_sweep": NGL_SWEEP},
           "points": []}

    for ngl in NGL_SWEEP:
        try:
            r = bench_ngl(ngl, path, args.backend, args.gpu)
        except Exception as e:
            print(f"  NGL={ngl} FAILED: {e} — skipping", flush=True)
            r = None
        if r:
            res["points"].append(r)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
