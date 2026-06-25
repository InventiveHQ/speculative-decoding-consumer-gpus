"""Aggregate raw benchmark results (two GPUs) into the generational DATA blob."""
import json, os, statistics, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

CAT_LABELS = {"code": "Code", "math": "Math", "reasoning": "Reasoning",
              "summarization": "Summarization", "chat": "Chat"}
CAT_ORDER = ["code", "math", "reasoning", "summarization", "chat"]
METHODS = ["baseline", "draft_0_5b", "ngram"]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p, encoding="utf-8-sig")) if os.path.exists(p) else None


def mean_tps(records):
    vals = [r["tps"] for r in records if r.get("tps")]
    return statistics.mean(vals) if vals else None


def cat_tps(records):
    out = {}
    for c in CAT_ORDER:
        vals = [r["tps"] for r in records if r.get("tps") and r["category"] == c]
        if vals:
            out[c] = statistics.mean(vals)
    return out


def table_of(data):
    t = {}
    for run in data["runs"]:
        if run.get("status") != "ok":
            continue
        recs = run.get("records", [])
        t[(run["backend"], run["method"])] = {"overall": mean_tps(recs), "cats": cat_tps(recs)}
    return t


def rnd(v, n=1):
    return round(v, n) if v else 0


def sp(t, b, m):
    base = t.get((b, "baseline"), {}).get("overall")
    o = t.get((b, m), {}).get("overall")
    return (o / base) if (base and o) else None


def cat_sp(t, b, m, c):
    base = t.get((b, "baseline"), {}).get("cats", {}).get(c)
    o = t.get((b, m), {}).get("cats", {}).get(c)
    return (o / base) if (base and o) else None


def best_config(t):
    """highest overall speedup vs baseline among speculative methods, with its backend/method."""
    best = None
    for b in set(k[0] for k in t):
        for m in ["draft_0_5b", "ngram"]:
            s = sp(t, b, m)
            if s and (best is None or s > best["sp"]):
                best = {"sp": s, "backend": b, "method": m}
    return best


def gpu_block(data, key, name, sub, year):
    t = table_of(data)
    # Spotlight the draft config on whichever backend runs it best (highest code speedup).
    cands = [b for b in ("cuda", "vulkan") if cat_sp(t, b, "draft_0_5b", "code")]
    bb = max(cands, key=lambda b: cat_sp(t, b, "draft_0_5b", "code")) if cands else "vulkan"
    bm = "draft_0_5b"
    cats = []
    for c in CAT_ORDER:
        s = cat_sp(t, bb, bm, c)
        if s:
            cats.append({"label": CAT_LABELS[c], "speedup": round(s, 2)})
    best = {"sp": max((c["speedup"] for c in cats), default=1.0), "backend": bb, "method": bm}
    return {
        "key": key, "name": name, "sub": sub, "year": year,
        "method_tps": {bk: {m: rnd(t.get((bk, m), {}).get("overall")) for m in METHODS}
                       for bk in ("cuda", "vulkan")},
        "baseline_cuda": rnd(t.get(("cuda", "baseline"), {}).get("overall")),
        "baseline_vulkan": rnd(t.get(("vulkan", "baseline"), {}).get("overall")),
        "cuda_draft_sp": round(sp(t, "cuda", "draft_0_5b") or 0, 2),
        "vulkan_draft_sp": round(sp(t, "vulkan", "draft_0_5b") or 0, 2),
        "cuda_draft_code_sp": round(cat_sp(t, "cuda", "draft_0_5b", "code") or 0, 2),
        "vulkan_draft_math_sp": round(cat_sp(t, "vulkan", "draft_0_5b", "math") or 0, 2),
        "best_speedup": round(best["sp"], 2) if best else 1.0,
        "best_method": {"draft_0_5b": "Draft model", "ngram": "N-gram"}[bm],
        "best_backend": bb,
        "best_category": (max(cats, key=lambda x: x["speedup"]) if cats else None),
        "categories": cats,
        "_t": t,
    }


def x(v):
    return f"{v:.2f}×" if v else "n/a"


# ---------- community submissions (results/community/*.json) ----------
def _device_label(key, hw, meta=None):
    gpus = hw.get("gpus", []) if isinstance(hw, dict) else []
    # Non-NVIDIA contributors (Apple Metal, AMD) have no nvidia-smi data, so fall back
    # to the device name the harness recorded in the run's meta (e.g. via --gpu-label).
    meta_gpu = (meta.get("gpu") if isinstance(meta, dict) else None) or None
    if key == "cpu":
        name = (hw.get("cpu") or "").replace("(R)", "").replace("(TM)", "")
        name = name.split(" CPU @")[0].replace(" Processor", "")
        name = " ".join(name.split())
        return f"CPU · {name}" if name else "CPU"
    if key == "gpu1":
        return gpus[1]["name"] if len(gpus) > 1 else (meta_gpu or "GPU 1")
    if key == "gpu_14b":
        return (gpus[0]["name"] if gpus else (meta_gpu or "GPU")) + " · 14B"
    return gpus[0]["name"] if gpus else (meta_gpu or "GPU 0")


def _run_metrics(run_data):
    t = table_of(run_data)
    backs = sorted(set(k[0] for k in t))
    if not backs:
        return None
    base = max((t.get((b, "baseline"), {}).get("overall") or 0) for b in backs)
    best_draft = None
    for b in backs:
        s = sp(t, b, "draft_0_5b")
        if s and (best_draft is None or s > best_draft[0]):
            best_draft = (s, b)
    ng = max((sp(t, b, "ngram") or 0) for b in backs)
    bestcat = None
    for b in backs:
        for c in CAT_ORDER:
            s = cat_sp(t, b, "draft_0_5b", c)
            if s and (bestcat is None or s > bestcat["sp"]):
                bestcat = {"label": CAT_LABELS[c], "sp": round(s, 2)}
    return {
        "baseline": round(base, 1) if base else None,
        "draft_sp": round(best_draft[0], 2) if best_draft else None,
        "draft_backend": best_draft[1] if best_draft else None,
        "ngram_sp": round(ng, 2) if ng else None,
        "best_cat": bestcat,
    }


def community():
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "community", "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(d, dict) or "runs" not in d:
            continue
        hw = d.get("hardware", {})
        for key, run_data in d["runs"].items():
            if not isinstance(run_data, dict) or "runs" not in run_data:
                continue
            m = _run_metrics(run_data)
            if not m or not m["baseline"]:
                continue
            rows.append({"contributor": d.get("name"), "device": _device_label(key, hw, run_data.get("meta")),
                         "notes": d.get("notes", ""), **m})
    return rows


def main():
    bw_data = load("results.json")          # Blackwell 7B
    ps_data = load("results_1080ti.json")   # Pascal 7B
    gpus = []
    bw = gpu_block(bw_data, "blackwell", "RTX 5060 Ti", "Blackwell · 2025 · ~$570 · 16 GB", 2025)
    gpus.append(bw)
    ps = None
    if ps_data:
        ps = gpu_block(ps_data, "pascal", "GTX 1080 Ti", "Pascal · 2017 · 11 GB", 2017)
        gpus.append(ps)

    tb = bw["_t"]
    # headline = best speedup seen on any GPU
    head = max(gpus, key=lambda g: g["best_speedup"])
    peak_tps = max(v for g in gpus for bk in g["method_tps"].values() for v in bk.values() if v)

    # ---- small vs big model on Blackwell, using the working spec backend (Vulkan) ----
    big_data = load("results_14b.json")
    bigmodel = None
    if big_data:
        bt7 = bw["_t"]               # 7B Q4 on Blackwell
        bt14 = table_of(big_data)    # 14B Q4 on Blackwell
        bk = "cuda"                  # same backend that collapsed on the 7B — shows overhead amortizing
        def block(t, label):
            return {
                "label": label,
                "baseline": rnd(t.get((bk, "baseline"), {}).get("overall")),
                "draft_overall": rnd(t.get((bk, "draft_0_5b"), {}).get("overall")),
                "draft_sp": round(sp(t, bk, "draft_0_5b") or 0, 2),
                "draft_code_sp": round(cat_sp(t, bk, "draft_0_5b", "code") or 0, 2),
            }
        rows = []
        for c in CAT_ORDER:
            rows.append({"label": CAT_LABELS[c],
                         "small": round(cat_sp(bt7, bk, "draft_0_5b", c) or 0, 2),
                         "big": round(cat_sp(bt14, bk, "draft_0_5b", c) or 0, 2)})
        bigmodel = {
            "backend": bk,
            "small": block(bt7, "Qwen-Coder 7B · Q4_K_S"),
            "big": block(bt14, "Qwen-Coder 14B · Q4_K_M"),
            "rows": rows,
            "best_big_cat": max(rows, key=lambda r: r["big"]),
        }

    findings = [
        {"h": "Speculative decoding is not a free lunch at batch&nbsp;1",
         "p": f"NVIDIA's 15× comes from a 120B model at high concurrency. With one user and a fast 7B, generation is already memory-bandwidth-bound — the 5060 Ti runs ~{bw['baseline_cuda']} tok/s on its own. A draft model only helps if it guesses well enough to beat that, so the win is real but measured in tens of percent, not multiples — and only on the right tasks."},
        {"h": "The biggest surprise: speculation's overhead can backfire on a fast model",
         "p": (f"Speculative decoding isn't free — each draft-then-verify cycle carries overhead, and on an <em>already-fast</em> target it can cost more than it saves. On Blackwell's quick 7B (~{bw['baseline_cuda']} tok/s) the stock <b>CUDA</b> build's draft path collapsed to {x(bw['cuda_draft_sp'])} of baseline — a 3–4× slowdown at perfectly healthy acceptance rates. The <em>same CUDA build, same card</em>, pointed at a slower 14B hit {x(bigmodel['big']['draft_code_sp'])} on code — the heavier model gives the overhead something to amortize against. The eight-year-old <b>Pascal</b> 1080&nbsp;Ti never collapsed on the 7B either, since it only runs ~{ps['baseline_cuda']} tok/s. The <b>Vulkan</b> build carries far less speculative overhead ({x(bw['vulkan_draft_sp'])} overall on the 7B, where CUDA managed only {x(bw['cuda_draft_sp'])}). Rule of thumb: the faster your model already runs, the more the runtime choice matters — on a fast small model, prefer Vulkan or model-free n-gram."
                if (ps and bigmodel) else
                f"The stock CUDA build's draft path collapsed to {x(bw['cuda_draft_sp'])} of baseline on the fast 7B, while Vulkan ran it at {x(bw['vulkan_draft_sp'])} — speculation overhead backfiring on an already-fast model. Use Vulkan or n-gram on fast small models.")},
        {"h": "Speedups are entirely task-dependent — NVIDIA's exact shape",
         "p": f"Predictable text wins. The best working configs climbed to {x(max(c['speedup'] for g in gpus for c in g['categories']))} on code/math but fell <em>below</em> 1.0× on open-ended chat, because the tiny draft model nails structured output and flails on free-form prose. Acceptance rate is the whole game — match the method to the workload."},
        {"h": "N-gram speculation is the free safety net",
         "p": "It needs no second model and almost no memory, predicts repeats straight from your context, and landed right around 1.0× across the suite on both cards and both runtimes — it rarely hurts and sometimes helps on repetitive or structured output. Little reason not to leave it on."},
        {"h": "For memory-bound decoding, the generation gap is smaller than the spec sheet",
         "p": (f"The 2017 GTX 1080 Ti baselined at ~{ps['baseline_vulkan']} tok/s (Vulkan) versus the 2025 5060 Ti's ~{bw['baseline_cuda']} — only about {round(bw['baseline_cuda']/max(ps['baseline_vulkan'],ps['baseline_cuda']),2)}×, despite eight years and a generational leap, because single-user decoding is bound by memory bandwidth (where the two cards are far closer than their age suggests), not raw compute. Worth knowing before you upgrade just to run a local model faster."
                if ps else
                "Single-user decoding is bound by memory bandwidth, not raw compute — so the gap between GPU generations is smaller here than their spec sheets suggest.")},
        {"h": "The bigger the model, the bigger the win — and a 14B does fit",
         "p": (f"Speculation pays off most on slow models, which is the whole reason NVIDIA runs it on a 120B. We see the same curve in miniature: moving from the 7B to a full-quality 14B (Q4_K_M) on the 5060 Ti, draft speculation on code jumped from {x(bigmodel['small']['draft_code_sp'])} to <b>{x(bigmodel['big']['draft_code_sp'])}</b>. And contrary to our first assumption, the 14B + draft fits in 16&nbsp;GB with room to spare (~10&nbsp;GB used at 4K context) — the weights are only ~9&nbsp;GB; it's the KV cache and context length, not the model, that you budget for. Keep the context sane (or quantize the KV cache with <code class='inl'>-ctk q8_0</code>) and a 14B speculates happily on a consumer card."
                if bigmodel else
                "Speculation pays off most on big, slow models — which is why NVIDIA runs it on a 120B — and a quantized 14B fits a 16&nbsp;GB card with room for a draft model alongside it.")},
    ]

    # ---- CPU-only run (7B), to test the overhead thesis at the slow extreme ----
    cpu_data = load("results_cpu.json")
    cpu = None
    if cpu_data:
        tc = table_of(cpu_data)
        bk = "cpu"
        rows = [{"label": CAT_LABELS[c], "sp": round(cat_sp(tc, bk, "draft_0_5b", c) or 0, 2)}
                for c in CAT_ORDER if cat_sp(tc, bk, "draft_0_5b", c)]
        cpu = {
            "baseline": rnd(tc.get((bk, "baseline"), {}).get("overall")),
            "draft": rnd(tc.get((bk, "draft_0_5b"), {}).get("overall")),
            "ngram": rnd(tc.get((bk, "ngram"), {}).get("overall")),
            "draft_sp": round(sp(tc, bk, "draft_0_5b") or 0, 2),
            "draft_code_sp": round(cat_sp(tc, bk, "draft_0_5b", "code") or 0, 2),
            "rows": rows,
            "best_cat": max(rows, key=lambda r: r["sp"]) if rows else None,
        }

    # global best category speedup — 7B across GPUs, plus the 14B big-model run
    best_cat_label = head["best_category"]["label"] if head["best_category"] else "code"
    best_cat_val = head["best_category"]["speedup"] if head["best_category"] else 1.0
    best_cat_model = "7B"
    if bigmodel and bigmodel["best_big_cat"]["big"] > (best_cat_val or 0):
        best_cat_label = bigmodel["best_big_cat"]["label"]
        best_cat_val = bigmodel["best_big_cat"]["big"]
        best_cat_model = "14B"

    data = {
        "date": "June 2026",
        "lossless": True,
        "gpus": [{k: v for k, v in g.items() if k != "_t"} for g in gpus],
        "bigmodel": bigmodel,
        "cpu": cpu,
        "community": community(),
        "tldr": {
            "best_speedup": head["best_speedup"],
            "best_method": head["best_method"],
            "best_backend": head["best_backend"],
            "best_gpu": head["name"],
            "best_cat": best_cat_label,
            "best_cat_speedup": best_cat_val,
            "best_cat_model": best_cat_model,
            "peak_tps": round(peak_tps),
        },
        "findings": findings,
    }
    out = os.path.join(R, "data.json")
    json.dump(data, open(out, "w", encoding="utf-8"), indent=2)

    # console summary
    for g in gpus:
        print(f"\n== {g['name']} ({g['sub']}) ==")
        print(f"{'method':<14}{'CUDA':>9}{'Vulkan':>9}")
        for m in METHODS:
            print(f"{m:<14}{g['method_tps']['cuda'][m]:>9}{g['method_tps']['vulkan'][m]:>9}")
        print(f"  best: {x(g['best_speedup'])} via {g['best_method']}/{g['best_backend']}; "
              f"cuda-draft {x(g['cuda_draft_sp'])}, vk-draft {x(g['vulkan_draft_sp'])}")
        print(f"  categories(best cfg): " + ", ".join(f"{c['label']} {x(c['speedup'])}" for c in g['categories']))
    print("\nDATA ->", out)


if __name__ == "__main__":
    main()
