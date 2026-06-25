"""CPU thread-scaling sweep: find the optimal thread count for CPU-only LLM inference.

The rule of thumb says "more threads = faster" but after the physical core count
hyperthreading kicks in — adding threads past that point can actually hurt.

  python scripts/bench.py

Env:
  SPECBENCH_MODELS_DIR   directory containing GGUF models (required)
  SPECBENCH_VULKAN_EXE   path to llama-server executable (vulkan build supports CPU too)
  SPECBENCH_MODEL_BASE   model base name, default Qwen2.5-Coder-3B-Instruct
                         expects <base>-Q4_K_M.gguf inside MODELS_DIR
"""
import argparse, json, os, platform, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS   # noqa: E402
import llama_client as lc     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.environ.get("SPECBENCH_MODELS_DIR", os.path.join(ROOT, "models"))
EXE = os.environ.get("SPECBENCH_VULKAN_EXE",
                     os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe"))
MODEL_BASE = os.environ.get("SPECBENCH_MODEL_BASE", "Qwen2.5-Coder-3B-Instruct")
MAX_TOKENS = 256
CTX = 4096

THREAD_COUNTS = [1, 2, 4, 6, 8, 12, 16]


def cpu_name():
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                           "(Get-CimInstance Win32_Processor).Name"],
                                          text=True, stderr=subprocess.DEVNULL)
            if out.strip():
                return out.strip().splitlines()[0].strip()
        elif platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def physical_cores():
    """Return physical (non-hyperthreaded) core count, or None if not detectable."""
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor |"
                 " Measure-Object -Property NumberOfCores -Sum).Sum"],
                text=True, stderr=subprocess.DEVNULL)
            val = int(out.strip().splitlines()[-1].strip())
            return val if val > 0 else None
        elif platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("cpu cores"):
                    return int(line.split(":", 1)[1].strip())
        elif platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"],
                                          text=True, stderr=subprocess.DEVNULL)
            return int(out.strip())
    except Exception:
        pass
    return None


def bench_threads(model_path, n, port=8099):
    print(f"\n=== threads={n} ===", flush=True)
    log = os.path.join(ROOT, "results", f"server-t{n}.log")
    srv = lc.LlamaServer(
        EXE, model_path, backend="cpu",
        extra_args=["-t", str(n), "-tb", str(n)],
        port=port, ctx=CTX, log_path=log,
    )
    try:
        srv.start()
        records = lc.run_suite(srv.port, PROMPTS, max_tokens=MAX_TOKENS)
        tps_vals = [r["tps"] for r in records if r.get("tps")]
        mean_tps = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None
        print(f"  mean tok/s: {mean_tps}", flush=True)
        return mean_tps
    finally:
        srv.stop()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    args = ap.parse_args()

    model_path = os.path.join(MODELS_DIR, f"{MODEL_BASE}-Q4_K_M.gguf")
    if not os.path.exists(model_path):
        sys.exit(
            f"Model not found: {model_path}\n"
            f"Set SPECBENCH_MODELS_DIR and/or SPECBENCH_MODEL_BASE."
        )

    logical = os.cpu_count() or 1
    phys = physical_cores()
    cname = cpu_name()
    sweep = [n for n in THREAD_COUNTS if n <= logical]
    if not sweep:
        sweep = [1]

    print(f"CPU:           {cname}")
    print(f"Logical cores: {logical}  Physical cores: {phys}")
    print(f"Model:         {model_path}")
    print(f"Thread sweep:  {sweep}")

    res = {
        "meta": {
            "model": MODEL_BASE,
            "cpu_name": cname,
            "logical_cores": logical,
            "physical_cores": phys,
            "max_tokens": MAX_TOKENS,
            "ctx": CTX,
        },
        "points": [],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for n in sweep:
        tps = bench_threads(model_path, n)
        if tps is not None:
            res["points"].append({"threads": n, "tps": tps})
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)

    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
