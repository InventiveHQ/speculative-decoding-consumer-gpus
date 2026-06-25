"""Aggregate the flash-attention sweep into chart-ready data.json (tolerates no data).

Reads results/results.json and writes results/data.json with:
  has_data   — false when no results yet (pending state)
  points     — list of {ctx, fa, tps, vram_mb}
  contexts   — sorted unique context lengths
  community  — list of community submissions bundled from results/community/*.json
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
    """Collect community submissions from results/community/*.json."""
    rows = []
    pat = os.path.join(R, "community", "*.json")
    for f in sorted(glob.glob(pat)):
        d = loadj(f)
        if not d or "runs" not in d:
            continue
        rows.append({
            "name": d.get("name"),
            "hardware": d.get("hardware", {}),
            "points": (d["runs"].get("flashattn") or {}).get("points", []),
        })
    return rows


def main():
    data = loadj(os.path.join(R, "results.json"))
    points = (data or {}).get("points", [])
    contexts = sorted({p["ctx"] for p in points}) if points else []
    out = {
        "experiment": "flash-attention",
        "has_data": bool(points),
        "model": (data or {}).get("meta", {}).get("model"),
        "backend": (data or {}).get("meta", {}).get("backend"),
        "points": points,
        "contexts": contexts,
        "community": community(),
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)
    if points:
        print(f"{'ctx':>8}  {'fa':<5}  {'tok/s':>8}  {'vram_mb':>8}")
        for p in points:
            fa_str = "on" if p.get("fa") else "off"
            print(f"{p['ctx']:>8}  {fa_str:<5}  {str(p.get('tps', '—')):>8}  "
                  f"{str(p.get('vram_mb', '—')):>8}")
    else:
        print("No flash-attention results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
