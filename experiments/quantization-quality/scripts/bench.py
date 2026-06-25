"""Quantization sweep: speed vs quality vs size as you drop GGUF quant levels.

For each quant of the SAME model that you've placed in the models dir, this:
  - launches llama.cpp (via the shared harness) on your chosen device,
  - measures generation tok/s (shared 12-prompt suite) and VRAM used,
  - records greedy outputs (for output-drift vs the highest-precision quant),
  - grades a small math set (pass-rate) as a relatable quality signal.

  python scripts/bench.py --backend cuda --gpu 0 --out results/results.json

Download the quants yourself (e.g. bartowski Qwen2.5-Coder-7B GGUF: Q2_K, Q3_K_M,
Q4_K_M, Q5_K_M, Q6_K, Q8_0) into SPECBENCH_MODELS_DIR. Missing quants are skipped.
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
MODEL_BASE = os.environ.get("SPECBENCH_MODEL_BASE", "Qwen2.5-Coder-7B-Instruct")

# Ordered low-bit -> high-bit. `bits` is a rough bits-per-weight for the size axis.
QUANTS = [
    {"label": "Q2_K",   "bits": 2.6},
    {"label": "Q3_K_M", "bits": 3.4},
    {"label": "Q4_K_M", "bits": 4.5},
    {"label": "Q5_K_M", "bits": 5.5},
    {"label": "Q6_K",   "bits": 6.6},
    {"label": "Q8_0",   "bits": 8.5},
]


def gpu_used_mb(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(idx)],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def bench_quant(q, backend, gpu):
    path = os.path.join(MODELS_DIR, f"{MODEL_BASE}-{q['label']}.gguf")
    if not os.path.exists(path):
        print(f"  [{q['label']}] missing ({path}) — skipping", flush=True)
        return None
    print(f"\n=== {q['label']} ===", flush=True)
    size_mb = round(os.path.getsize(path) / 1e6)
    before = gpu_used_mb(gpu) if backend != "cpu" else None
    srv = lc.LlamaServer(EXE[backend], path, backend=backend, gpu_index=gpu, ctx=CTX,
                         log_path=os.path.join(ROOT, "results", f"server-{q['label']}.log"))
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
            print(f"  {pr['id']:<20} {round(m['tps'],1) if m['tps'] else '—'} tok/s", flush=True)
        passed = 0
        for g in GRADED:
            m = lc.measure(srv.port, g["prompt"], max_tokens=64)
            if grade(m["text"], g["answer"]):
                passed += 1
        print(f"  math: {passed}/{len(GRADED)}  vram: {vram} MiB  tok/s: {round(sum(tps)/len(tps),1) if tps else '—'}", flush=True)
        return {"label": q["label"], "bits": q["bits"], "file_mb": size_mb, "vram_mb": vram,
                "tps": round(sum(tps) / len(tps), 2) if tps else None,
                "math_pass": passed, "math_total": len(GRADED), "outputs": outputs}
    finally:
        srv.stop()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda", choices=["cuda", "vulkan", "cpu"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    args = ap.parse_args()

    res = {"meta": {"model": MODEL_BASE, "backend": args.backend, "gpu": args.gpu,
                    "max_tokens": MAX_TOKENS, "ctx": CTX,
                    "quality": "graded math pass-rate + greedy output-drift vs highest-precision quant"},
           "quants": []}
    for q in QUANTS:
        r = bench_quant(q, args.backend, args.gpu)
        if r:
            res["quants"].append(r)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
