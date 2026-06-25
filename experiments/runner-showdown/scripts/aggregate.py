"""Aggregate runner-showdown results into chart-ready data.json (tolerates no data)."""
import json, os, glob, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
CAT_LABELS = {"code": "Code", "math": "Math", "reasoning": "Reasoning",
              "summarization": "Summarization", "chat": "Chat"}
CAT_ORDER = ["code", "math", "reasoning", "summarization", "chat"]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p, encoding="utf-8-sig")) if os.path.exists(p) else None


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
        recs = run.get("records", [])
        o = mean_tps(recs)
        if not o:
            continue
        rows.append({"name": run["name"], "model": run.get("model"),
                     "overall": round(o, 1), "cats": cat_tps(recs)})
    rows.sort(key=lambda r: r["overall"], reverse=True)
    if rows:
        top = rows[0]["overall"]
        for r in rows:
            r["rel"] = round(r["overall"] / top, 2)
            r["overhead_pct"] = round((1 - r["overall"] / top) * 100, 1)
    return rows


def community():
    out = []
    for f in sorted(glob.glob(os.path.join(R, "community", "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8-sig"))
        except Exception:
            continue
        rows = runner_rows(d.get("runs", {}).get("showdown") if isinstance(d.get("runs"), dict) else d)
        if not rows:
            continue
        hw = d.get("hardware", {})
        gpu = (hw.get("gpus") or [{}])[0].get("name", "")
        out.append({"contributor": d.get("name"), "gpu": gpu,
                    "fastest": rows[0]["name"], "fastest_tps": rows[0]["overall"],
                    "ranking": [{"name": r["name"], "overall": r["overall"], "rel": r["rel"]} for r in rows]})
    return out


def main():
    data = load("results.json")
    rows = runner_rows(data) if data else []
    out = {
        "experiment": "runner-showdown",
        "metric": "wall-clock tokens/sec",
        "has_data": bool(rows),
        "fastest": rows[0]["name"] if rows else None,
        "runners": rows,
        "categories": [CAT_LABELS[c] for c in CAT_ORDER],
        "cat_keys": CAT_ORDER,
        "cat_labels": CAT_LABELS,
        "community": community(),
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)
    if rows:
        print("Runners (wall-clock tok/s):")
        for r in rows:
            print(f"  {r['name']:<12} {r['overall']:>7}  rel {r['rel']}  (-{r['overhead_pct']}%)")
    else:
        print("No runner results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
