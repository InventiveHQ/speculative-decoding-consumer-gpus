"""Aggregate the context-length sweep into chart-ready data.json (tolerates no data).

Reads results/results.json, sorts points by ctx, and writes results/data.json.
Running before the benchmark produces a valid 'pending' file (has_data: false).
Community submissions in results/community/*.json are folded in too.
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
    """Scan results/community/*.json and return a list of submission summaries."""
    comm_dir = os.path.join(R, "community")
    entries = []
    for f in sorted(glob.glob(os.path.join(comm_dir, "*.json"))):
        d = loadj(f)
        if not d:
            continue
        ctx_run = (d.get("runs") or {}).get("context") or {}
        pts = ctx_run.get("points") or []
        if not isinstance(pts, list):
            continue
        entries.append({
            "name": d.get("name"),
            "hardware": d.get("hardware", {}),
            "notes": d.get("notes", ""),
            "meta": ctx_run.get("meta", {}),
            "points": sorted(pts, key=lambda p: p.get("ctx", 0)),
        })
    return entries


def main():
    data = loadj(os.path.join(R, "results.json"))
    points = sorted((data or {}).get("points", []), key=lambda p: p.get("ctx", 0))
    meta = (data or {}).get("meta") or {}
    comm = community()

    out = {
        "experiment": "context-length-tax",
        "has_data": bool(points),
        "meta": meta if points else None,
        "points": points,
        "community": comm,
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)

    if points:
        print(f"{'ctx':>8}{'tok/s':>10}{'VRAM MB':>10}{'TTFT ms':>10}")
        for p in points:
            tps = str(p.get("tps", "—"))
            vram = str(p.get("vram_mb", "—"))
            ttft = str(p.get("ttft_ms", "—"))
            print(f"{p['ctx']:>8}{tps:>10}{vram:>10}{ttft:>10}")
    else:
        print("No context sweep results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
