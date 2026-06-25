"""Aggregate draft-size sweep into chart-ready data.json (tolerates missing data)."""
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
    for f in sorted(glob.glob(os.path.join(R, "community", "*.json"))):
        d = loadj(f)
        if d:
            rows.append({
                "name": d.get("name"),
                "notes": d.get("notes"),
                "hardware": d.get("hardware"),
            })
    return rows


def main():
    data = loadj(os.path.join(R, "results.json"))
    meta = (data or {}).get("meta", {})
    raw_drafts = (data or {}).get("drafts", [])

    drafts = []
    for d in raw_drafts:
        if d.get("error"):
            continue
        drafts.append({
            "label": d.get("label"),
            "file_mb": d.get("file_mb"),
            "tps": d.get("tps"),
            "speedup": d.get("speedup"),
            "vram_mb": d.get("vram_mb"),
            "acceptance": d.get("acceptance"),
        })

    baseline_tps = (data or {}).get("baseline_tps")
    has_data = bool(drafts and baseline_tps is not None)

    out = {
        "experiment": "draft-size-sweep",
        "has_data": has_data,
        "target": meta.get("target"),
        "backend": meta.get("backend"),
        "gpu": meta.get("gpu"),
        "baseline_tps": baseline_tps,
        "drafts": drafts,
        "community": community(),
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)

    if has_data:
        print(f"{'draft':<8}{'tok/s':>8}{'speedup':>10}{'VRAM MiB':>10}{'accept%':>10}")
        print(f"{'baseline':<8}{str(baseline_tps):>8}{'1.000x':>10}{'—':>10}{'—':>10}")
        for d in drafts:
            acc = f"{round(d['acceptance'] * 100, 1)}%" if d.get("acceptance") is not None else "—"
            print(f"{d['label']:<8}{str(d['tps']):>8}{str(d['speedup'])+'x':>10}{str(d['vram_mb']):>10}{acc:>10}")
    else:
        print("No sweep results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
