"""Runner showdown: Ollama vs llama.cpp vs LM Studio — across devices.

They're all llama.cpp underneath, so this measures the *wrapper tax* (bundled
version + default flags + HTTP/templating overhead), on the SAME model + quant,
across each device you can target: 5060 Ti, 1080 Ti, CPU, ...

bench.py probes whichever runners are currently reachable and tags the run with a
device label YOU set (after pinning each runner to that device — see README). Run
once per device:

  python scripts/bench.py --device 5060ti
  python scripts/bench.py --device 1080ti
  python scripts/bench.py --device cpu

Throughput is WALL-CLOCK tokens/sec (completion_tokens / elapsed) — the fair metric
across runners that don't expose llama.cpp-style timings.
"""
import argparse, json, os, sys, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS                      # noqa: E402
import llama_client as lc                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAX_TOKENS = 256

# Same model across all three. `model` is the id each runner expects; llama.cpp
# usually ignores it. Override via --config runners.json if your ports/ids differ.
RUNNERS = [
    {"name": "llama.cpp", "host": "127.0.0.1", "port": 8080,  "model": "local-model"},
    {"name": "ollama",    "host": "127.0.0.1", "port": 11434, "model": "qwen2.5-coder:7b"},
    {"name": "lm studio", "host": "127.0.0.1", "port": 1234,  "model": "qwen2.5-coder-7b-instruct"},
]


def reachable(host, port):
    for path in ("/v1/models", "/health"):
        try:
            with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            continue
    return False


def bench_runner(r):
    print(f"\n=== {r['name']}  ({r['host']}:{r['port']}, model={r['model']}) ===", flush=True)
    if not reachable(r["host"], r["port"]):
        print("  NOT REACHABLE on this device — skipping.", flush=True)
        return {"name": r["name"], "endpoint": f"{r['host']}:{r['port']}", "model": r["model"],
                "status": "unreachable"}
    try:
        lc.chat(r["port"], "Say hello.", host=r["host"], max_tokens=16, model=r["model"])
    except Exception as e:
        print(f"  warmup failed: {e}", flush=True)
        return {"name": r["name"], "endpoint": f"{r['host']}:{r['port']}", "model": r["model"],
                "status": "error", "error": str(e)}
    records = []
    for pr in PROMPTS:
        try:
            m = lc.measure(r["port"], pr["prompt"], host=r["host"], model=r["model"], max_tokens=MAX_TOKENS)
            rec = {"id": pr["id"], "category": pr.get("category"), "tps": m["tps_wallclock"],
                   "reported_tps": round(m["tps"], 2) if m["tps"] else None, "predicted_n": m["predicted_n"]}
            print(f"  {pr['id']:<20} {rec['tps']:>7} tok/s (wall)" if rec["tps"] else f"  {pr['id']}: no tokens", flush=True)
        except Exception as e:
            rec = {"id": pr["id"], "category": pr.get("category"), "error": str(e)}
            print(f"  {pr['id']}: ERR {e}", flush=True)
        records.append(rec)
    return {"name": r["name"], "endpoint": f"{r['host']}:{r['port']}", "model": r["model"],
            "status": "ok", "records": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="default",
                    help="label for the device all runners are pinned to (e.g. 5060ti, 1080ti, cpu)")
    ap.add_argument("--out", default=None, help="default: results/results-<device>.json")
    ap.add_argument("--config", help="JSON file overriding the RUNNERS list")
    ap.add_argument("--only", help="comma-separated runner names to include")
    args = ap.parse_args()

    runners = json.load(open(args.config, encoding="utf-8")) if args.config else RUNNERS
    if args.only:
        want = {s.strip().lower() for s in args.only.split(",")}
        runners = [r for r in runners if r["name"].lower() in want]
    out = args.out or os.path.join(ROOT, "results", f"results-{args.device}.json")

    res = {"meta": {"device": args.device, "max_tokens": MAX_TOKENS,
                    "metric": "wall-clock tokens/sec",
                    "note": "same model+quant across runners; greedy, 256 tokens, shared 12-prompt suite"},
           "runners": []}
    print(f"DEVICE = {args.device}")
    for r in runners:
        res["runners"].append(bench_runner(r))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
