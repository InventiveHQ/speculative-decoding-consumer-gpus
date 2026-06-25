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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short label for your submission, e.g. rtx4090-ryzen7950x")
    ap.add_argument("--notes", default="", help="optional free-text notes (quant, llama.cpp version, etc.)")
    args = ap.parse_args()

    runs = {}
    for fname, key in KNOWN.items():
        p = os.path.join(R, fname)
        if os.path.exists(p):
            runs[key] = json.load(open(p, encoding="utf-8-sig"))
    if not runs:
        sys.exit("No results/*.json found. Run the benchmark first (see README).")

    submission = {
        "name": args.name,
        "notes": args.notes,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
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
