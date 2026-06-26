"""Model-size sweep: speed vs correctness across Qwen2.5-Coder family sizes.

For each model size present in your models dir, this:
  - launches llama.cpp (via the shared harness) on your chosen device,
  - measures generation tok/s (shared 12-prompt suite) and VRAM used,
  - grades a small math/reasoning set (pass-rate) as a correctness signal.

  python scripts/bench.py --backend cuda --gpu 0 --out results/results.json

Download the models (all Qwen2.5-Coder-Instruct variants) into SPECBENCH_MODELS_DIR.
Missing model files are skipped automatically.

SECURITY NOTE: Model-generated code is NOT executed. Correctness is measured via
exact-answer matching on a graded math/reasoning set only.
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS                       # noqa: E402
import llama_client as lc                         # noqa: E402
from quality_set import GRADED, grade             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.environ.get("SPECBENCH_MODELS_DIR", os.path.join(ROOT, "models"))
EXE = {
    "cuda":   os.environ.get("SPECBENCH_CUDA_EXE",   os.path.join(ROOT, "runtimes", "cuda", "llama-server.exe")),
    "vulkan": os.environ.get("SPECBENCH_VULKAN_EXE", os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe")),
    "cpu":    os.environ.get("SPECBENCH_VULKAN_EXE", os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe")),
}
MAX_TOKENS = 256
CTX = 4096

# Ordered smallest -> largest parameter count.
MODELS = [
    {"label": "0.5B", "file": "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"},
    {"label": "1.5B", "file": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"},
    {"label": "3B",   "file": "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"},
    {"label": "7B",   "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_S.gguf"},
    {"label": "14B",  "file": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"},
]


def gpu_used_mb(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(idx)],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def bench_model(m, backend, gpu):
    path = os.path.join(MODELS_DIR, m["file"])
    if not os.path.exists(path):
        print(f"  [{m['label']}] missing ({path}) — skipping", flush=True)
        return None
    print(f"\n=== {m['label']} ({m['file']}) ===", flush=True)
    size_mb = round(os.path.getsize(path) / 1e6)
    before = gpu_used_mb(gpu) if backend != "cpu" else None
    srv = lc.LlamaServer(EXE[backend], path, backend=backend, gpu_index=gpu, ctx=CTX,
                         log_path=os.path.join(ROOT, "results", f"server-{m['label']}.log"))
    try:
        srv.start()
        time.sleep(1)
        after = gpu_used_mb(gpu) if backend != "cpu" else None
        vram = (after - before) if (before is not None and after is not None) else None
        try:
            lc.chat(srv.port, "Say hello.", max_tokens=16)
        except Exception:
            pass
        tps_vals = []
        for pr in PROMPTS:
            m_res = lc.measure(srv.port, pr["prompt"], max_tokens=MAX_TOKENS)
            if m_res["tps"]:
                tps_vals.append(m_res["tps"])
            print(f"  {pr['id']:<20} {round(m_res['tps'],1) if m_res['tps'] else '—'} tok/s", flush=True)
        passed = 0
        for g in GRADED:
            m_res = lc.measure(srv.port, g["prompt"], max_tokens=64)
            if grade(m_res["text"], g["answer"]):
                passed += 1
        avg_tps = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None
        print(f"  pass: {passed}/{len(GRADED)}  vram: {vram} MiB  tok/s: {avg_tps}", flush=True)
        return {"label": m["label"], "file": m["file"], "file_mb": size_mb, "vram_mb": vram,
                "tps": avg_tps, "pass": passed, "total": len(GRADED)}
    finally:
        srv.stop()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda", choices=["cuda", "vulkan", "cpu"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    args = ap.parse_args()

    res = {"meta": {"backend": args.backend, "gpu": args.gpu,
                    "max_tokens": MAX_TOKENS, "ctx": CTX,
                    "quality": "graded math/reasoning pass-rate (exact-answer matching)"},
           "models": []}
    for m in MODELS:
        try:
            r = bench_model(m, args.backend, args.gpu)
        except Exception as e:
            print(f"  MODEL FAILED ({m['label']}): {e} — skipping", flush=True)
            r = None
        if r:
            res["models"].append(r)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
        time.sleep(2)  # extra VRAM-release headroom between model launches
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
