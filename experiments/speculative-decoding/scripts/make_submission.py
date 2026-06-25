"""Bundle your benchmark results + hardware info into one community submission file.

Run this AFTER you've produced one or more results/*.json files (see README).
It captures your GPU/CPU details and copies in your results, ready to PR into
results/community/.

    python scripts/make_submission.py --name "rtx4090-ryzen7950x"

Produces results/community/<name>.json. Open a PR adding that file. Thanks!
"""
import argparse, json, os, platform, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results")

# results files we know about -> a friendly key in the submission
KNOWN = {
    "results.json": "gpu0",
    "results_1080ti.json": "gpu1",
    "results_14b.json": "gpu_14b",
    "results_cpu.json": "cpu",
}


def cpu_name():
    """Best-effort friendly CPU model name across OSes."""
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                text=True, stderr=subprocess.DEVNULL)
            if out.strip():
                return out.strip().splitlines()[0].strip()
        elif platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({"name": parts[0], "memory": parts[1],
                             "driver": parts[2], "compute_cap": parts[3]})
        return gpus
    except Exception:
        return []


def git_state(path):
    """Is this results file locally produced/edited, or the repo's committed sample?

    Returns "changed" (modified or untracked — i.e. yours), "clean" (identical to
    what's committed — almost certainly the repo's sample data, not yours), or
    None if git is unavailable / the file isn't in a git repo.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", os.path.basename(path)],
            cwd=os.path.dirname(path), text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return "changed" if out.strip() else "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short label for your submission, e.g. rtx4090-ryzen7950x")
    ap.add_argument("--notes", default="", help="optional free-text notes (quant, llama.cpp version, etc.)")
    ap.add_argument("--include", default=None,
                    help="comma-separated result files to bundle, e.g. 'results.json,results_cpu.json'. "
                         "By default only files you changed locally are included, so you don't accidentally "
                         "submit the repo's committed sample data.")
    args = ap.parse_args()

    explicit = {s.strip() for s in args.include.split(",")} if args.include else None

    runs = {}
    skipped, no_git = [], False
    for fname, key in KNOWN.items():
        p = os.path.join(R, fname)
        if not os.path.exists(p):
            continue
        if explicit is not None:
            if fname not in explicit:
                continue
        else:
            state = git_state(p)
            if state == "clean":
                # Identical to what's committed: this is the repo's sample data, not the
                # contributor's run. Skip it (the original bug bundled it as if it were theirs).
                skipped.append(fname)
                continue
            if state is None:
                no_git = True
        runs[key] = json.load(open(p, encoding="utf-8-sig"))

    if skipped:
        print("Skipped (unchanged from the repo's committed results — looks like sample data, not yours):")
        for f in skipped:
            print(f"  - {f}   (run it, or pass --include {f} if you really did produce it)")
    if no_git:
        print("WARNING: git not available, so I can't tell your results from the repo's sample data — "
              "included every results/*.json present. Double-check the bundled runs, or use --include.")
    if not runs:
        sys.exit("No new results to bundle. Run the benchmark first (see README), "
                 "or pass --include <files> to force-include specific result files.")
    print("Including: " + ", ".join(sorted(runs)))

    submission = {
        "name": args.name,
        "notes": args.notes,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": cpu_name(),
            "gpus": nvidia_smi(),
        },
        "runs": runs,
    }
    out = os.path.join(R, "community", f"{args.name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(submission, open(out, "w", encoding="utf-8"), indent=2)
    print(f"Wrote {out}")
    print("Open a pull request adding that file to results/community/. Thank you!")


if __name__ == "__main__":
    main()
