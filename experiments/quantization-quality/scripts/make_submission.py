"""Bundle your results + auto-detected hardware into a community submission.

    python scripts/make_submission.py --name "rtx5090-ryzen7950x"

Writes results/community/<name>.json: your results.json under runs.sweep, plus a
full hardware snapshot (CPU, RAM size/speed, GPU + PCIe link, motherboard, storage)
captured automatically by harness/hwinfo.py. Open a PR adding that file. Thanks!
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "..", "..", "harness"))
import hwinfo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short label, e.g. rtx5090-ryzen7950x")
    ap.add_argument("--notes", default="", help="optional notes (quant, llama.cpp version, etc.)")
    args = ap.parse_args()
    res = os.path.join(R, "results.json")
    if not os.path.exists(res):
        sys.exit("No results/results.json - run the benchmark first (see README).")
    sub = {"name": args.name, "notes": args.notes,
           "hardware": hwinfo.collect(),
           "runs": {"sweep": json.load(open(res, encoding="utf-8-sig"))}}
    out = os.path.join(R, "community", f"{args.name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(sub, open(out, "w", encoding="utf-8"), indent=2)
    print(f"Wrote {out}")
    print("Captured rig: " + hwinfo.summary(sub["hardware"]))
    print("Open a PR adding that file to results/community/. Thank you!")


if __name__ == "__main__":
    main()
