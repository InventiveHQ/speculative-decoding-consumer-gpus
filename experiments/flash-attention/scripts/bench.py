"""Flash-attention sweep: does -fa actually speed up generation on a consumer GPU?

Fixed model + quant (SPECBENCH_MODEL_BASE-Q4_K_M.gguf), backend cuda.
Sweeps context lengths [4096, 16384, 32768] x flash-attention [off, on].
Records generation tok/s (shared 12-prompt suite) and VRAM used (nvidia-smi delta).

NOTE: the llama.cpp flash-attn flag has been "-fa"/"--flash-attn", historically
boolean and in recent builds takes on/off/auto — adjust if your build differs.

  python scripts/bench.py --gpu 0 --out results/results.json

Place your model in SPECBENCH_MODELS_DIR as <MODEL_BASE>-Q4_K_M.gguf.
Set SPECBENCH_CUDA_EXE to your llama-server.exe path if not using the default.
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS                       # noqa: E402
import llama_client as lc                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.environ.get("SPECBENCH_MODELS_DIR", os.path.join(ROOT, "models"))
EXE = os.environ.get("SPECBENCH_CUDA_EXE",
                     os.path.join(ROOT, "runtimes", "cuda", "llama-server.exe"))
MODEL_BASE = os.environ.get("SPECBENCH_MODEL_BASE", "Qwen2.5-Coder-7B-Instruct")
MAX_TOKENS = 256
BACKEND = "cuda"

# Context lengths to sweep — short, medium, long.
CONTEXTS = [4096, 16384, 32768]


def gpu_used_mb(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(idx)],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def bench_one(model_path, ctx, fa_on, gpu):
    """Run one (ctx, fa) cell; return a result dict or None on failure."""
    fa_label = "on" if fa_on else "off"
    # NOTE: the llama.cpp flash-attn flag has been "-fa"/"--flash-attn", historically
    # boolean and in recent builds takes on/off/auto — adjust if your build differs.
    extra = ["-fa", "on"] if fa_on else []
    print(f"\n=== ctx={ctx}  fa={fa_label} ===", flush=True)
    before = gpu_used_mb(gpu)
    log = os.path.join(ROOT, "results", f"server-ctx{ctx}-fa{fa_label}.log")
    srv = lc.LlamaServer(EXE, model_path, backend=BACKEND, gpu_index=gpu,
                         ctx=ctx, extra_args=extra, log_path=log)
    try:
        srv.start()
        time.sleep(1)
        after = gpu_used_mb(gpu)
        vram = (after - before) if (before is not None and after is not None) else None
        tps_list = []
        for pr in PROMPTS:
            m = lc.measure(srv.port, pr["prompt"], max_tokens=MAX_TOKENS)
            if m["tps"]:
                tps_list.append(m["tps"])
            print(f"  {pr['id']:<22} {round(m['tps'], 1) if m['tps'] else '—'} tok/s",
                  flush=True)
        mean_tps = round(sum(tps_list) / len(tps_list), 2) if tps_list else None
        print(f"  vram: {vram} MiB  mean tok/s: {mean_tps}", flush=True)
        return {"ctx": ctx, "fa": fa_on, "tps": mean_tps, "vram_mb": vram}
    finally:
        srv.stop()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU index (CUDA_VISIBLE_DEVICES; default 0)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"),
                    help="Output path for results.json")
    args = ap.parse_args()

    model_file = f"{MODEL_BASE}-Q4_K_M.gguf"
    model_path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(model_path):
        sys.exit(
            f"Model not found: {model_path}\n"
            f"Set SPECBENCH_MODELS_DIR or place {model_file} there."
        )

    res = {
        "meta": {
            "model": MODEL_BASE,
            "quant": "Q4_K_M",
            "backend": BACKEND,
            "gpu": args.gpu,
            "max_tokens": MAX_TOKENS,
            "contexts": CONTEXTS,
        },
        "points": [],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for ctx in CONTEXTS:
        for fa_on in [False, True]:
            try:
                r = bench_one(model_path, ctx, fa_on, args.gpu)
            except Exception as e:
                print(f"  CELL FAILED (ctx={ctx} fa={fa_on}): {e} — skipping", flush=True)
                r = None
            if r:
                res["points"].append(r)
            json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
            time.sleep(2)  # extra VRAM-release headroom between launches

    print(f"\nWrote {args.out}", flush=True)
    print("Next: python scripts/aggregate.py && python scripts/inject.py")


if __name__ == "__main__":
    main()
