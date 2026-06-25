"""Aggregate the model-size sweep into chart-ready data.json (tolerates no data).

Correctness = graded math/reasoning pass-rate across the shared GRADED set.
The 'knee' is where pass-rate climbs sharply as model size increases — that's the
smallest model that reliably solves problems.
"""
import json, os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def loadj(p):
    try:
        return json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        return None


def summarize(data):
    models = [m for m in (data or {}).get("models", []) if m.get("label")]
    if not models:
        return []
    rows = []
    for m in models:
        rows.append({
            "label": m["label"],
            "file_mb": m.get("file_mb"),
            "vram_mb": m.get("vram_mb"),
            "tps": m.get("tps"),
            "pass": m.get("pass"),
            "total": m.get("total"),
            "pass_pct": round(100 * m["pass"] / m["total"], 0) if m.get("total") else None,
        })
    return rows


def community():
    pat = os.path.join(R, "community", "*.json")
    rows = []
    for p in glob.glob(pat):
        d = loadj(p)
        if d:
            rows.append({
                "name": d.get("name"),
                "notes": d.get("notes", ""),
                "hardware": d.get("hardware", {}).get("gpus", []),
                "models": (d.get("runs") or {}).get("models", {}).get("models", []),
            })
    return rows


def main():
    data = loadj(os.path.join(R, "results.json"))
    rows = summarize(data)
    out = {
        "experiment": "smallest-model-that-can-code",
        "has_data": bool(rows),
        "backend": (data or {}).get("meta", {}).get("backend"),
        "models": rows,
        "community": community(),
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)
    if rows:
        print(f"{'model':<8}{'tok/s':>8}{'VRAM':>8}{'size':>8}{'pass':>8}{'pass%':>8}")
        for r in rows:
            print(f"{r['label']:<8}{str(r['tps']):>8}{str(r['vram_mb']):>8}{str(r['file_mb']):>8}{str(r['pass']):>8}{str(r['pass_pct']):>8}")
    else:
        print("No model results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
