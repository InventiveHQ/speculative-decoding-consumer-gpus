"""Aggregate power-sweep results into chart-ready data.json (tolerates no data).

Reads results/results.json produced by bench.py, sorts points by watt_limit,
and merges any community submissions from results/community/*.json.
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
    pattern = os.path.join(R, "community", "*.json")
    for path in sorted(glob.glob(pattern)):
        d = loadj(path)
        if d:
            rows.append({
                "name": d.get("name"),
                "notes": d.get("notes", ""),
                "hardware": d.get("hardware", {}),
            })
    return rows


def main():
    data = loadj(os.path.join(R, "results.json"))
    points = []
    if data and data.get("points"):
        for pt in data["points"]:
            if pt.get("watt_limit") is not None:
                points.append({
                    "watt_limit": pt["watt_limit"],
                    "tps": pt.get("tps"),
                    "avg_power_w": pt.get("avg_power_w"),
                    "tps_per_watt": pt.get("tps_per_watt"),
                    "limit_set": pt.get("limit_set"),
                })
        points.sort(key=lambda p: p["watt_limit"])   # ascending for charts

    out = {
        "experiment": "tokens-per-watt",
        "has_data": bool(points),
        "meta": (data or {}).get("meta", {}),
        "points": points,
        "community": community(),
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)

    if points:
        print(f"{'watts':>8}{'tok/s':>8}{'avg_W':>8}{'tok/s/W':>10}{'limit_set':>12}")
        for p in points:
            print(f"{str(p['watt_limit']):>8}{str(p['tps']):>8}{str(p['avg_power_w']):>8}"
                  f"{str(p['tps_per_watt']):>10}{str(p.get('limit_set', '?')):>12}")
    else:
        print("No power-sweep results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
