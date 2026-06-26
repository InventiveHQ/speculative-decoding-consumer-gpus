"""MoE on CPU: does a 30B Mixture-of-Experts run at *active*-parameter speed?

For each model present, on CPU only (-ngl 0), this:
  - measures generation tok/s with the shared 12-prompt suite,
  - grades a small math/reasoning set (pass-rate) as a quality signal,
  - records active vs total parameter counts.

The thesis: single-user CPU decode is memory-bandwidth-bound, so tok/s tracks the
parameters *touched per token* (active), not the total on disk. A 30B-A3B MoE
(3.3B active) should land near a dense 3B for speed while answering like a far
larger model.

  python scripts/bench.py --out results/results.json --threads 6

Env:
  SPECBENCH_MODELS_DIR   one or more dirs holding the GGUFs (os.pathsep-separated).
                         Defaults also search D:\\models and ./models.
  SPECBENCH_CPU_EXE      llama-server build to use (must support qwen3moe/qwen3vl).
                         Falls back to the CUDA build (runs CPU via -ngl 0).

SECURITY NOTE: model-generated code is never executed. Quality is exact-answer
matching on a graded math/reasoning set only.
"""
import argparse, json, os, platform, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS                       # noqa: E402
import llama_client as lc                         # noqa: E402
import hwinfo                                      # noqa: E402
from quality_set import GRADED, grade             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAX_TOKENS = 256
CTX = 4096

# Candidate dirs to search for each GGUF (first hit wins).
MODEL_DIRS = [d for d in os.environ.get("SPECBENCH_MODELS_DIR", "").split(os.pathsep) if d] + [
    os.path.join(ROOT, "models"), r"D:\models",
    os.path.join(ROOT, "..", "..", "models"),
]
EXE = os.environ.get("SPECBENCH_CPU_EXE",
                     os.path.join(ROOT, "..", "..", "runtimes", "cuda", "llama-server.exe"))

# active_b / total_b in billions. dense models: active == total.
MODELS = [
    {"label": "1.5B dense",   "kind": "dense", "active_b": 1.5,  "total_b": 1.5,
     "file": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"},
    {"label": "3B dense",     "kind": "dense", "active_b": 3.1,  "total_b": 3.1,
     "file": "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"},
    {"label": "7B dense",     "kind": "dense", "active_b": 7.6,  "total_b": 7.6,
     "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"},
    {"label": "8B dense",     "kind": "dense", "active_b": 8.0,  "total_b": 8.0,
     "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"},
    {"label": "30B-A3B MoE",  "kind": "moe",   "active_b": 3.3,  "total_b": 30.5,
     "file": "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf"},
]


def find_model(fname):
    for d in MODEL_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def bench_model(m, threads, port=8137):
    path = find_model(m["file"])
    if not path:
        print(f"  [{m['label']}] missing ({m['file']}) — skipping", flush=True)
        return None
    print(f"\n=== {m['label']} ({m['file']}) ===", flush=True)
    size_mb = round(os.path.getsize(path) / 1e6)
    log = os.path.join(ROOT, "results", f"server-{m['label'].split()[0]}.log")
    srv = lc.LlamaServer(EXE, path, backend="cpu", port=port, ctx=CTX,
                         extra_args=["-t", str(threads), "-tb", str(threads)], log_path=log)
    try:
        srv.start()
        try:
            lc.chat(srv.port, "Say hello.", max_tokens=16)   # warmup
        except Exception:
            pass
        tps_vals = []
        for pr in PROMPTS:
            r = lc.measure(srv.port, pr["prompt"], max_tokens=MAX_TOKENS)
            if r["tps"]:
                tps_vals.append(r["tps"])
            print(f"  {pr['id']:<20} {round(r['tps'],1) if r['tps'] else '—'} tok/s", flush=True)
        passed = 0
        for g in GRADED:
            r = lc.measure(srv.port, g["prompt"], max_tokens=96)
            if grade(r["text"], g["answer"]):
                passed += 1
        avg = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None
        print(f"  -> {avg} tok/s   quality {passed}/{len(GRADED)}   ({size_mb} MB on disk)", flush=True)
        return {"label": m["label"], "kind": m["kind"], "file": m["file"],
                "active_b": m["active_b"], "total_b": m["total_b"], "file_mb": size_mb,
                "tps": avg, "pass": passed, "total": len(GRADED)}
    finally:
        srv.stop()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6, help="CPU threads (default 6 = physical cores)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    args = ap.parse_args()

    if not os.path.exists(EXE):
        sys.exit(f"llama-server not found: {EXE}\nSet SPECBENCH_CPU_EXE.")

    hw = hwinfo.collect()
    print(f"CPU:     {hw.get('cpu')}")
    print(f"Threads: {args.threads}   (physical/logical: "
          f"{hw.get('cpu_cores',{}).get('physical')}/{hw.get('cpu_cores',{}).get('logical')})")
    print(f"Exe:     {EXE}")

    res = {"meta": {"device": "cpu", "cpu": hw.get("cpu"), "threads": args.threads,
                    "ram": hw.get("ram"), "max_tokens": MAX_TOKENS, "ctx": CTX,
                    "quality": "graded math/reasoning pass-rate (exact-answer matching)",
                    "exe_version": _exe_version()},
           "models": []}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for m in MODELS:
        try:
            r = bench_model(m, args.threads)
        except Exception as e:
            print(f"  MODEL FAILED ({m['label']}): {e} — skipping", flush=True)
            r = None
        if r:
            res["models"].append(r)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
        time.sleep(2)
    print(f"\nWrote {args.out}", flush=True)


def _exe_version():
    try:
        out = subprocess.run([EXE, "--version"], capture_output=True, text=True, timeout=15)
        for line in (out.stderr + out.stdout).splitlines():
            if "version" in line.lower():
                return line.strip()
    except Exception:
        pass
    return None


if __name__ == "__main__":
    main()
