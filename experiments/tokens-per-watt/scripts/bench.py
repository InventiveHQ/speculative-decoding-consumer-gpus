"""Tokens-per-watt sweep: measure how capping GPU power affects generation speed.

For a fixed model (CUDA backend), this:
  - Queries the GPU's default, max, and min power limits via nvidia-smi,
  - Sweeps power limits at [100, 90, 80, 70, 60, 50]% of the default (clamped
    to >= min limit, duplicates removed),
  - For each cap: sets the limit (REQUIRES ADMIN — see note below), launches the
    model once, runs the shared 12-prompt suite measuring tok/s, and samples power
    draw during generation to compute tokens-per-watt,
  - In a finally block, ALWAYS restores the default power limit.

  python scripts/bench.py --gpu 0 --out results/results.json

NOTE: Setting GPU power limits requires administrator/elevated privileges.
  Windows: run from an Administrator command prompt (right-click -> Run as admin).
  Linux:   run with sudo, or grant nvidia-smi via sudoers.
If the power-limit command fails, the sweep prints a clear warning, records
limit_set=false for that step, and continues measuring at the current active limit.

IMPORTANT: This script modifies GPU hardware settings. The default power limit is
always restored in a finally block, even if the sweep is interrupted. If the script
is killed ungracefully (SIGKILL / task-killed), run manually:
  nvidia-smi -pl <default_watts> -i <gpu>
"""
import argparse, json, os, subprocess, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS          # noqa: E402
import llama_client as lc            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.environ.get("SPECBENCH_MODELS_DIR", os.path.join(ROOT, "models"))
EXE = os.environ.get("SPECBENCH_CUDA_EXE",
                     os.path.join(ROOT, "runtimes", "cuda", "llama-server.exe"))
MAX_TOKENS = 256
CTX = 4096
MODEL_BASE = os.environ.get("SPECBENCH_MODEL_BASE", "Qwen2.5-Coder-7B-Instruct")

PERCENT_STEPS = [100, 90, 80, 70, 60, 50]
PORT = 8099


# ---------------------------------------------------------------------------
# nvidia-smi helpers
# ---------------------------------------------------------------------------

def query_power_limits(gpu):
    """Return (default_limit_w, max_limit_w, min_limit_w) as floats, or raise."""
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=power.default_limit,power.max_limit,power.min_limit",
         "--format=csv,noheader,nounits", "-i", str(gpu)],
        text=True, stderr=subprocess.DEVNULL)
    parts = [p.strip() for p in out.strip().split(",")]
    return float(parts[0]), float(parts[1]), float(parts[2])


def set_power_limit(watts, gpu):
    """Set GPU power limit. Returns True on success, False on failure."""
    try:
        subprocess.check_call(
            ["nvidia-smi", "-pl", str(int(watts)), "-i", str(gpu)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        print(f"\n  WARNING: Could not set power limit to {int(watts)}W on GPU {gpu}.", flush=True)
        print(f"  nvidia-smi -pl requires administrator/elevated privileges.", flush=True)
        print(f"  On Windows: run from an Administrator command prompt.", flush=True)
        print(f"  On Linux: run with sudo.", flush=True)
        print(f"  Measurement continues at the current active power limit.", flush=True)
        return False


def gpu_used_mb(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(idx)],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-power-point benchmark
# ---------------------------------------------------------------------------

def bench_power_point(watt_limit, limit_set, model_path, gpu):
    """Launch the server, run the full suite, sample power. Returns result dict."""
    print(f"\n=== {watt_limit}W (limit {'set' if limit_set else 'UNSET — running at current limit'}) ===",
          flush=True)
    log = os.path.join(ROOT, "results", f"server-{watt_limit}w.log")
    srv = lc.LlamaServer(EXE, model_path, backend="cuda", gpu_index=gpu,
                         ctx=CTX, port=PORT, log_path=log)

    tps_all = []
    power_samples = []
    stop_sampling = threading.Event()

    def power_sampler():
        """Background thread: poll nvidia-smi for power draw every 0.5 s."""
        while not stop_sampling.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits", "-i", str(gpu)],
                    text=True, stderr=subprocess.DEVNULL)
                power_samples.append(float(out.strip()))
            except Exception:
                pass
            time.sleep(0.5)

    try:
        srv.start()
        time.sleep(1)

        # Warmup
        try:
            lc.chat(srv.port, "Say hello.", max_tokens=16)
        except Exception:
            pass

        # Start background power sampling
        sampler = threading.Thread(target=power_sampler, daemon=True)
        sampler.start()

        try:
            for pr in PROMPTS:
                m = lc.measure(srv.port, pr["prompt"], max_tokens=MAX_TOKENS)
                if m["tps"]:
                    tps_all.append(m["tps"])
                print(f"  {pr['id']:<22} {round(m['tps'], 1) if m['tps'] else '—'} tok/s",
                      flush=True)
        finally:
            stop_sampling.set()
            sampler.join(timeout=5)

    finally:
        srv.stop()
        time.sleep(2)

    avg_tps = round(sum(tps_all) / len(tps_all), 2) if tps_all else None
    avg_power = round(sum(power_samples) / len(power_samples), 1) if power_samples else None
    tps_per_watt = round(avg_tps / avg_power, 4) if (avg_tps and avg_power) else None
    print(f"  -> avg tok/s={avg_tps}  avg_power_w={avg_power}  tok/s/W={tps_per_watt}  "
          f"power samples={len(power_samples)}", flush=True)
    return {
        "watt_limit": watt_limit,
        "limit_set": limit_set,
        "tps": avg_tps,
        "avg_power_w": avg_power,
        "tps_per_watt": tps_per_watt,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sweep GPU power limits and measure tok/s + tokens-per-watt.")
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU index (default 0)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"),
                    help="Output path for results.json")
    args = ap.parse_args()

    model_path = os.path.join(MODELS_DIR, f"{MODEL_BASE}-Q4_K_M.gguf")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}", flush=True)
        print(f"Set SPECBENCH_MODELS_DIR and/or SPECBENCH_MODEL_BASE.", flush=True)
        sys.exit(1)

    print(f"Model: {model_path}", flush=True)
    print(f"GPU:   {args.gpu}", flush=True)

    # Query power limits
    default_w = max_w = min_w = None
    try:
        default_w, max_w, min_w = query_power_limits(args.gpu)
        print(f"Power limits — default: {default_w}W  max: {max_w}W  min: {min_w}W", flush=True)
    except Exception as e:
        print(f"Could not query power limits via nvidia-smi: {e}", flush=True)
        print(f"Proceeding with a single run at the current power limit.", flush=True)

    # Build sweep steps (deduplicated, high->low)
    steps = []  # list of (watt_limit, pct)
    if default_w is not None:
        seen = set()
        for pct in PERCENT_STEPS:
            w = round(default_w * pct / 100)
            if min_w is not None:
                w = max(w, int(min_w))
            if w not in seen:
                seen.add(w)
                steps.append((w, pct))
        steps.sort(key=lambda x: x[0], reverse=True)   # 100% -> 50%
    else:
        steps = [(0, 100)]   # 0 = "unknown"; will still run once

    res = {
        "meta": {
            "model": MODEL_BASE,
            "quant": "Q4_K_M",
            "gpu": args.gpu,
            "default_limit_w": default_w,
            "max_limit_w": max_w,
            "min_limit_w": min_w,
            "max_tokens": MAX_TOKENS,
            "ctx": CTX,
            "backend": "cuda",
        },
        "points": [],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    try:
        for watt_limit, pct in steps:
            limit_set = False
            if default_w is not None:
                desc = f"{pct}% of {default_w}W default"
                print(f"\nSetting power limit to {watt_limit}W ({desc})...", flush=True)
                limit_set = set_power_limit(watt_limit, args.gpu)
            else:
                print(f"\nRunning at current power limit (could not query limits).", flush=True)

            point = bench_power_point(watt_limit, limit_set, model_path, args.gpu)
            res["points"].append(point)
            json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
            print(f"Checkpoint saved to {args.out}", flush=True)
    finally:
        # ALWAYS restore the default power limit
        if default_w is not None:
            print(f"\nRestoring default power limit ({default_w}W)...", flush=True)
            ok = set_power_limit(default_w, args.gpu)
            if ok:
                print("Power limit restored to default.", flush=True)
            else:
                print(f"WARNING: Could not restore power limit to {default_w}W.", flush=True)
                print(f"  Run manually: nvidia-smi -pl {int(default_w)} -i {args.gpu}", flush=True)

    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
