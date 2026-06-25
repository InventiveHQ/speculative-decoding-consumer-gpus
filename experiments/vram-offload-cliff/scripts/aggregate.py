"""Aggregate the VRAM offload cliff sweep into chart-ready data.json (tolerates no data).

Reads results/results.json and emits results/data.json with points sorted by ngl.
Runs in pending state (has_data: false) if no results exist yet.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def loadj(p):
    try:
        return json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        return None


def summarize(data):
    pts = [p for p in (data or {}).get("points", []) if p.get("ngl") is not None]
    if not pts:
        return []
    return sorted(pts, key=lambda p: p["ngl"])


def community():
    return []


def main():
    data = loadj(os.path.join(R, "results.json"))
    points = summarize(data)
    out = {
        "experiment": "vram-offload-cliff",
        "has_data": bool(points),
        "model": (data or {}).get("meta", {}).get("model"),
        "backend": (data or {}).get("meta", {}).get("backend"),
        "points": points,
        "community": community(),
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)
    if points:
        print(f"{'ngl':>5}{'tok/s':>10}{'VRAM MiB':>12}")
        for p in points:
            print(f"{p['ngl']:>5}{str(p.get('tps', '—')):>10}{str(p.get('vram_mb', '—')):>12}")
    else:
        print("No sweep results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
