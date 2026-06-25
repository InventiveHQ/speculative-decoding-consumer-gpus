"""Aggregate the quantization sweep into chart-ready data.json (tolerates no data).

Quality = graded math pass-rate + greedy output-similarity vs the highest-precision
quant present (difflib ratio over the shared suite). The 'cliff' is wherever those
drop sharply as bits fall.
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
    quants = [q for q in (data or {}).get("quants", []) if q.get("label")]
    if not quants:
        return []
    quants.sort(key=lambda q: q.get("bits", 0))            # low-bit -> high-bit
    ref = quants[-1]                                        # highest precision present
    rows = []
    for q in quants:
        outs, refs = q.get("outputs", {}), ref.get("outputs", {})
        sims = [similarity(outs.get(k), refs.get(k)) for k in refs]
        sims = [s for s in sims if s is not None]
        rows.append({
            "label": q["label"], "bits": q.get("bits"),
            "file_mb": q.get("file_mb"), "vram_mb": q.get("vram_mb"),
            "tps": q.get("tps"),
            "math_pass": q.get("math_pass"), "math_total": q.get("math_total"),
            "math_pct": round(100 * q["math_pass"] / q["math_total"], 0) if q.get("math_total") else None,
            "similarity": round(100 * statistics.mean(sims), 1) if sims else None,
            "is_ref": q["label"] == ref["label"],
        })
    return rows


def main():
    data = loadj(os.path.join(R, "results.json"))
    rows = summarize(data)
    out = {
        "experiment": "quantization-quality",
        "has_data": bool(rows),
        "model": (data or {}).get("meta", {}).get("model"),
        "backend": (data or {}).get("meta", {}).get("backend"),
        "reference": rows[-1]["label"] if rows else None,
        "quants": rows,
        "community": [],
    }
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "data.json"), "w", encoding="utf-8"), indent=2)
    if rows:
        print(f"{'quant':<8}{'tok/s':>8}{'VRAM':>8}{'size':>8}{'math%':>8}{'sim%':>8}")
        for r in rows:
            print(f"{r['label']:<8}{str(r['tps']):>8}{str(r['vram_mb']):>8}{str(r['file_mb']):>8}{str(r['math_pct']):>8}{str(r['similarity']):>8}")
    else:
        print("No quant results yet — data.json written in 'pending' state.")


if __name__ == "__main__":
    main()
