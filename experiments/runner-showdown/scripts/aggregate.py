"""Aggregate runner-showdown results across devices into chart-ready data.json.

Reads every results/results-*.json (one per device), builds per-device runner
rankings + a runner x device matrix. Tolerates missing data (pending state).
"""
import json, os, glob, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
CAT_LABELS = {"code": "Code", "math": "Math", "reasoning": "Reasoning",
              "summarization": "Summarization", "chat": "Chat"}
CAT_ORDER = ["code", "math", "reasoning", "summarization", "chat"]


def loadj(p):
    try:
        return json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        return None


def mean_tps(recs):
    v = [r["tps"] for r in recs if r.get("tps")]
    return statistics.mean(v) if v else None


def cat_tps(recs):
    out = {}
    for c in CAT_ORDER:
        v = [r["tps"] for r in recs if r.get("tps") and r.get("category") == c]
        if v:
            out[c] = round(statistics.mean(v), 1)
    return out


def runner_rows(data):
    rows = []
    for run in (data or {}).get("runners", []):
        if run.get("status") != "ok":
            continue
        o = mean_tps(run.get("records", []))
        if not o:
            continue
        rows.append({"name": run["name"], "model": run.get("model"),
                     "overall": round(o, 1), "cats": cat_tps(run["records"])})
    rows.sort(key=lambda r: r["overall"], reverse=True)
    if rows:
        top = rows[0]["overall"]
        for r in rows:
            r["rel"] = round(r["overall"] / top, 2)
            r["overhead_pct"] = round((1 - r["overall"] / top) * 100, 1)
    return rows


def device_blocks():
    """One block per results-<device>.json (plus legacy results.json)."""
    blocks = []
    files = sorted(glob.glob(os.path.join(R, "results-*.json")))
    if os.path.exists(os.path.join(R, "results.json")):
        files.append(os.path.join(R, "results.json"))
    for f in files:
        d = loadj(f)
        if not d:
            continue
        rows = runner_rows(d)
        if not rows:
            continue
        dev = (d.get("meta", {}) or {}).get("device") or \
            os.path.basename(f).replace("results-", "").replace(".json", "")
        blocks.append({"device": dev, "fastest": rows[0]["name"], "runners": rows})
    return blocks


def build_matrix(blocks):
    runners, devices = [], [b["device"] for b in blocks]
    for b in blocks:
        for r in b["runners"]:
            if r["name"] not in runners:
                runners.append(r["name"])
    cells = {rn: {} for rn in runners}
    for b in blocks:
        for r in b["runners"]:
            cells[r["name"]][b["device"]] = r["overall"]
    return {"runners": runners, "devices": devices, "cells": cells}


def community():
    out = []
    for f in sorted(glob.glob(os.path.join(R, "community", "*.json"))):
        d = loadj(f)
        if not d:
            continue
        runs = d.get("runs", {})
        # a submission may bundle several device files under runs.*
        best = None
        for v in (runs.values() if isinstance(runs, dict) else []):
            for r in runner_rows(v):
                if best is None or r["overall"] > best["overall"]:
                    best = {"name": r["name"], "overall": r["overall"]}
        if not best:
            continue
        hw = d.get("hardware", {})
        out.append({"contributor": d.get("name"), "gpu": (hw.get("gpus") or [{}])[0].get("name", ""),
                    "fastest": best["name"], "fastest_tps": best["overall"]})
    return out


def main():
    blocks = device_blocks()
    out = {
        "experiment": "runner-showdown",
        "metric": "wall-clock tokens/sec",
        "has_data": bool(blocks),
        "devices": blocks,
        "matrix": build_matrix(blocks) if blocks else {"runners": [], "devices": [], "cells": {}},
        "cat_keys": CAT_ORDER,
        "cat_labels": CAT_LABELS,
        "community": community(),
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)
    if blocks:
        for b in blocks:
            print(f"[{b['device']}] fastest={b['fastest']}  " +
                  "  ".join(f"{r['name']}={r['overall']}({r['rel']}x)" for r in b["runners"]))
    else:
        print("No results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
