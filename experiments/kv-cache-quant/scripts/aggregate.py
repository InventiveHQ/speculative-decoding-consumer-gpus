"""Aggregate the KV-cache sweep into chart-ready data.json (tolerates no data).

Quality = greedy output-similarity vs the f16 KV-cache config (difflib ratio
over the shared prompt suite). Shows whether q8_0/q4_0 KV caches preserve
output quality as they shrink VRAM usage.
"""
import json, os, glob, statistics, difflib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def loadj(p):
    try:
        return json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        return None


def similarity(a, b):
    if not a or not b:
        return None
    return difflib.SequenceMatcher(None, a, b).ratio()


def summarize(data):
    configs = [c for c in (data or {}).get("configs", []) if c.get("label")]
    if not configs:
        return [], None
    # f16 is the reference; fall back to first config if f16 absent
    ref = next((c for c in configs if c["label"] == "f16"), configs[0])
    ref_label = ref["label"]
    rows = []
    for c in configs:
        outs, refs = c.get("outputs", {}), ref.get("outputs", {})
        sims = [similarity(outs.get(k), refs.get(k)) for k in refs]
        sims = [s for s in sims if s is not None]
        rows.append({
            "label": c["label"],
            "vram_mb": c.get("vram_mb"),
            "tps": c.get("tps"),
            "similarity": round(100 * statistics.mean(sims), 1) if sims else None,
            "is_ref": c["label"] == ref_label,
        })
    return rows, ref_label


def community():
    rows = []
    for p in glob.glob(os.path.join(R, "community", "*.json")):
        d = loadj(p)
        if d and "name" in d:
            rows.append({"name": d["name"], "notes": d.get("notes", ""),
                         "hardware": d.get("hardware", {})})
    return rows


def main():
    data = loadj(os.path.join(R, "results.json"))
    rows, ref_label = summarize(data)
    out = {
        "experiment": "kv-cache-quant",
        "has_data": bool(rows),
        "model": (data or {}).get("meta", {}).get("model"),
        "backend": (data or {}).get("meta", {}).get("backend"),
        "ctx": (data or {}).get("meta", {}).get("ctx"),
        "reference": ref_label,
        "configs": rows,
        "community": community(),
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)
    if rows:
        print(f"{'cache type':<12}{'tok/s':>8}{'VRAM MB':>10}{'sim%':>8}")
        for r in rows:
            print(f"{r['label']:<12}{str(r['tps']):>8}{str(r['vram_mb']):>10}{str(r['similarity']):>8}")
    else:
        print("No KV-cache results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
