"""Bundle your draft-size sweep results + hardware into a community submission.

    python scripts/make_submission.py --name "rtx4090"

Writes results/community/<name>.json (your results.json under runs.draftsweep + hardware).
Open a PR adding that file.
"""
import argparse, json, os, platform, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


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


def nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
        return [{"name": p[0].strip(), "memory": p[1].strip(), "driver": p[2].strip(),
                 "compute_cap": p[3].strip()}
                for p in (l.split(",") for l in out.strip().splitlines()) if len(p) >= 4]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    res = os.path.join(R, "results.json")
    if not os.path.exists(res):
        sys.exit("No results/results.json — run the benchmark first.")
    sub = {"name": args.name, "notes": args.notes,
           "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                        "cpu": cpu_name(), "gpus": nvidia_smi()},
           "runs": {"draftsweep": json.load(open(res, encoding="utf-8-sig"))}}
    out = os.path.join(R, "community", f"{args.name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(sub, open(out, "w", encoding="utf-8"), indent=2)
    print(f"Wrote {out} — open a PR adding it. Thanks!")


if __name__ == "__main__":
    main()
