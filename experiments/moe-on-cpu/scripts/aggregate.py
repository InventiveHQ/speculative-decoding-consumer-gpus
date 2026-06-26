"""Aggregate results.json (+ community/*.json) into site/data.json for the page."""
import glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def load(p):
    try:
        return json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        return None


def fit_inverse(dense):
    """Fit tps ~= k / active over dense points; return k (or None)."""
    pts = [(d["active_b"], d["tps"]) for d in dense if d.get("tps") and d.get("active_b")]
    if not pts:
        return None
    return sum(a * t for a, t in pts) / len(pts)


def community():
    rows = []
    for p in sorted(glob.glob(os.path.join(R, "community", "*.json"))):
        d = load(p)
        if d and "runs" in d:
            rows.append({"name": d.get("name"), "notes": d.get("notes", ""),
                         "hardware": d.get("hardware", {}),
                         "models": d["runs"].get("moe", {}).get("models", [])})
    return rows


def main():
    data = load(os.path.join(R, "results.json")) or {"meta": {}, "models": []}
    models = sorted(data.get("models", []), key=lambda m: m["active_b"])
    dense = [m for m in models if m["kind"] == "dense"]
    moe = next((m for m in models if m["kind"] == "moe"), None)

    headline = {}
    if moe and dense:
        nearest = min(dense, key=lambda d: abs(d["active_b"] - moe["active_b"]))
        k = fit_inverse(dense)
        headline = {
            "moe_label": moe["label"], "moe_tps": moe["tps"],
            "moe_active_b": moe["active_b"], "moe_total_b": moe["total_b"],
            "moe_pass": moe["pass"], "moe_total": moe["total"],
            "nearest_label": nearest["label"], "nearest_tps": nearest["tps"],
            "nearest_active_b": nearest["active_b"],
            "nearest_pass": nearest["pass"],
            "speed_vs_nearest": round(moe["tps"] / nearest["tps"], 2) if nearest["tps"] else None,
            "total_x_nearest": round(moe["total_b"] / nearest["active_b"], 1),
            # what a *dense* model of the MoE's total size would do at active-param speed:
            "hyp_dense_total_tps": round(k / moe["total_b"], 2) if k else None,
            "speedup_vs_hyp_dense": round((moe["tps"] / (k / moe["total_b"])), 1) if k and moe["tps"] else None,
        }

    out = {
        "meta": data.get("meta", {}),
        "models": models,
        "headline": headline,
        "community": community(),
    }
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)
    if headline:
        print(f"MoE {headline['moe_tps']} tok/s vs {headline['nearest_label']} "
              f"{headline['nearest_tps']} ({headline['speed_vs_nearest']}x), "
              f"{headline['total_x_nearest']}x the params; a dense {moe['total_b']}B would do "
              f"~{headline['hyp_dense_total_tps']} tok/s "
              f"(MoE is {headline['speedup_vs_hyp_dense']}x faster than that).")
    print("Wrote", os.path.join(R, "data.json"))


if __name__ == "__main__":
    main()
