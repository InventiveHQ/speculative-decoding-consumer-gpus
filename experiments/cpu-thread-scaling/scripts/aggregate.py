"""Aggregate the CPU thread-scaling sweep into chart-ready data.json (tolerates no data).

Reads results/results.json produced by bench.py.
Writes results/data.json consumed by inject.py -> site/index.html.
Community JSON files in results/community/*.json are summarised into the
community list so the page can render multi-CPU comparisons.
"""
import json, os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def loadj(p):
    try:
        return json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        return None


def community():
    rows = []
    comm_dir = os.path.join(R, "community")
    if not os.path.isdir(comm_dir):
        return rows
    for f in glob.glob(os.path.join(comm_dir, "*.json")):
        d = loadj(f)
        if not d:
            continue
        threads_data = (d.get("runs") or {}).get("threads") or {}
        points = threads_data.get("points", [])
        valid = [p for p in points if p.get("tps") is not None]
        if not valid:
            continue
        best = max(valid, key=lambda p: p["tps"])
        rows.append({
            "name": d.get("name"),
            "cpu": (d.get("hardware") or {}).get("cpu"),
            "points": sorted(valid, key=lambda p: p["threads"]),
            "peak_tps": best["tps"],
            "peak_threads": best["threads"],
        })
    return rows


def main():
    data = loadj(os.path.join(R, "results.json"))
    meta = (data or {}).get("meta", {})
    raw_points = (data or {}).get("points", [])
    points = sorted(
        [p for p in raw_points if p.get("tps") is not None],
        key=lambda p: p["threads"],
    )
    out = {
        "experiment": "cpu-thread-scaling",
        "has_data": bool(points),
        "cpu_name": meta.get("cpu_name"),
        "logical_cores": meta.get("logical_cores"),
        "physical_cores": meta.get("physical_cores"),
        "model": meta.get("model"),
        "points": points,
        "community": community(),
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)
    if points:
        best = max(points, key=lambda p: p["tps"])
        print(f"{'threads':>8}{'tok/s':>10}")
        for p in points:
            marker = " <- peak" if p["threads"] == best["threads"] else ""
            print(f"{p['threads']:>8}{p['tps']:>10.2f}{marker}")
        print(f"\nPeak: {best['tps']:.2f} tok/s at {best['threads']} threads")
    else:
        print("No results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
