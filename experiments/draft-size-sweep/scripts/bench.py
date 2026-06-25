"""Draft-model size sweep for speculative decoding on a single consumer GPU.

Fixed TARGET model; sweeps DRAFT models (0.5B, 1.5B, 3B) to find the tok/s sweet spot.
Measures baseline (no draft) tok/s first, then per-draft tok/s, speedup, and VRAM used.

Usage:
    python scripts/bench.py --out results/results.json

Env vars:
    SPECBENCH_MODELS_DIR   folder holding .gguf files       (default: <experiment>/models)
    SPECBENCH_TARGET       target model base name           (default: Qwen2.5-Coder-14B-Instruct)
    SPECBENCH_CUDA_EXE     path to CUDA llama-server.exe   (default: <experiment>/runtimes/cuda/llama-server.exe)
    SPECBENCH_VULKAN_EXE   path to Vulkan llama-server.exe (unused by default; available for reference)
"""
import argparse, json, os, re, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS  # noqa: E402
import llama_client as lc   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR  = os.environ.get("SPECBENCH_MODELS_DIR",  os.path.join(ROOT, "models"))
TARGET_BASE = os.environ.get("SPECBENCH_TARGET",       "Qwen2.5-Coder-14B-Instruct")
CUDA_EXE    = os.environ.get("SPECBENCH_CUDA_EXE",    os.path.join(ROOT, "runtimes", "cuda",   "llama-server.exe"))
VULKAN_EXE  = os.environ.get("SPECBENCH_VULKAN_EXE",  os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe"))

MAX_TOKENS = 256
CTX = 4096
PORT = 8099

# Draft models to sweep in order of increasing size.
DRAFT_MODELS = [
    {"label": "0.5B", "file": "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"},
    {"label": "1.5B", "file": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"},
    {"label": "3B",   "file": "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"},
]


def gpu_used_mb(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(idx)],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def avg_tps(records):
    vals = [r["tps"] for r in records if r.get("tps") is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def parse_acceptance_from_log(log_path):
    """Best-effort: scan server log for 'draft acceptance = X.XX' and return last value."""
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        matches = re.findall(r"draft acceptance\s*=\s*([\d.]+)", text, re.IGNORECASE)
        if matches:
            return float(matches[-1])
    except Exception:
        pass
    return None


def run_suite_on_server(exe, model_path, backend, gpu, extra_args, log_path):
    """Launch a LlamaServer, capture VRAM delta, run the shared suite, return (records, vram)."""
    before = gpu_used_mb(gpu)
    srv = lc.LlamaServer(
        exe, model_path, backend=backend, gpu_index=gpu,
        ctx=CTX, extra_args=extra_args, log_path=log_path
    )
    try:
        srv.start()
        time.sleep(1)
        after = gpu_used_mb(gpu)
        vram = (after - before) if (before is not None and after is not None) else None
        records = lc.run_suite(srv.port, PROMPTS, max_tokens=MAX_TOKENS)
        for r in records:
            tstr = f"{round(r['tps'], 1)} tok/s" if r.get("tps") else "—"
            print(f"  {r['id']:<20} {tstr}", flush=True)
        return records, vram
    finally:
        srv.stop()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    ap.add_argument("--gpu", type=int, default=0,
                    help="physical GPU index (0 = primary, default 0)")
    ap.add_argument("--gpu-label", default=None,
                    help="override the device name recorded in meta")
    args = ap.parse_args()

    gpu = args.gpu
    backend = "cuda"
    exe = CUDA_EXE
    target_path = os.path.join(MODELS_DIR, f"{TARGET_BASE}-Q4_K_M.gguf")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    gpu_name = args.gpu_label or {
        0: "RTX 5060 Ti (Blackwell, sm_120)",
        1: "GTX 1080 Ti (Pascal, sm_61)",
    }.get(gpu, f"GPU {gpu}")

    meta = {
        "target": TARGET_BASE,
        "target_file": f"{TARGET_BASE}-Q4_K_M.gguf",
        "backend": backend,
        "gpu": gpu_name,
        "gpu_index": gpu,
        "max_tokens": MAX_TOKENS,
        "ctx": CTX,
        "spec_type": "draft-simple",
        "spec_draft_n_max": 5,
    }

    print(f"TARGET  = {target_path}", flush=True)
    print(f"GPU     = {gpu_name}", flush=True)

    # --- BASELINE (no draft) ---
    print("\n=== BASELINE (no draft) ===", flush=True)
    if not os.path.exists(target_path):
        print(f"  ERROR: target model not found: {target_path}", flush=True)
        baseline_tps = None
        baseline_records = []
    else:
        baseline_log = os.path.join(ROOT, "results", "server-baseline.log")
        baseline_records, _ = run_suite_on_server(
            exe, target_path, backend, gpu, [], baseline_log)
        baseline_tps = avg_tps(baseline_records)
        print(f"  baseline avg tok/s: {baseline_tps}", flush=True)

    results = {
        "meta": meta,
        "baseline_tps": baseline_tps,
        "baseline_records": baseline_records,
        "drafts": [],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --- DRAFT SWEEP ---
    for d in DRAFT_MODELS:
        draft_path = os.path.join(MODELS_DIR, d["file"])
        if not os.path.exists(draft_path):
            print(f"\n  [{d['label']}] missing ({draft_path}) — skipping", flush=True)
            continue
        print(f"\n=== DRAFT {d['label']} ({d['file']}) ===", flush=True)
        file_mb = round(os.path.getsize(draft_path) / 1e6)
        log_path = os.path.join(ROOT, "results", f"server-{d['label']}.log")
        extra_args = [
            "-md", draft_path,
            "--spec-type", "draft-simple",
            "--spec-draft-n-max", "5",
            "-ngld", "999",
        ]
        try:
            records, vram = run_suite_on_server(
                exe, target_path, backend, gpu, extra_args, log_path)
            tps = avg_tps(records)
            speedup = round(tps / baseline_tps, 3) if (tps and baseline_tps) else None
            acceptance = parse_acceptance_from_log(log_path)
            print(
                f"  draft avg tok/s: {tps}  speedup: {speedup}  "
                f"vram: {vram} MiB  acceptance: {acceptance}",
                flush=True,
            )
            results["drafts"].append({
                "label": d["label"],
                "file": d["file"],
                "file_mb": file_mb,
                "tps": tps,
                "speedup": speedup,
                "vram_mb": vram,
                "acceptance": acceptance,
            })
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results["drafts"].append({
                "label": d["label"],
                "file": d["file"],
                "error": str(e),
            })

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
