"""Runner showdown: Ollama vs llama.cpp vs LM Studio — same model, who's fastest?

Measures the throughput each wrapper delivers (and the overhead it adds) on the
*same* model + quant, using the shared harness and the canonical 12-prompt suite.

Assumes each runner is already serving an OpenAI-compatible endpoint with the SAME
model loaded (that's the fair-fight requirement). Start them first, e.g.:

  llama.cpp : llama-server -m Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf -ngl 999 --port 8080 --jinja
  ollama    : ollama serve   (then: ollama run qwen2.5-coder:7b once to pull)
  lm studio : load the same model, start its local server (port 1234)

Then:  python scripts/bench.py --out results/results.json

Throughput is reported as WALL-CLOCK tokens/sec (completion_tokens / elapsed) so
the comparison is fair across runners that don't expose llama.cpp-style timings.
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness"))
from prompts import PROMPTS                      # noqa: E402
import llama_client as lc                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAX_TOKENS = 256

# Same model across all three. `model` is the id each runner expects; llama.cpp
# usually ignores it (serves whatever is loaded). Override endpoints/models via a
# JSON config (--config) if your ports/model ids differ.
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
        print("  NOT REACHABLE — start this runner with the same model, or skip.", flush=True)
        return {"name": r["name"], "endpoint": f"{r['host']}:{r['port']}", "model": r["model"],
                "status": "unreachable"}
    # warmup
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
            rec = {"id": pr["id"], "category": pr.get("category"),
                   "tps": m["tps_wallclock"],                 # the fair, comparable metric
                   "reported_tps": round(m["tps"], 2) if m["tps"] else None,
                   "predicted_n": m["predicted_n"]}
            print(f"  {pr['id']:<20} {rec['tps']:>7} tok/s (wall)" if rec["tps"] else f"  {pr['id']}: no tokens", flush=True)
        except Exception as e:
            rec = {"id": pr["id"], "category": pr.get("category"), "error": str(e)}
            print(f"  {pr['id']}: ERR {e}", flush=True)
        records.append(rec)
    return {"name": r["name"], "endpoint": f"{r['host']}:{r['port']}", "model": r["model"],
            "status": "ok", "records": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    ap.add_argument("--config", help="JSON file overriding the RUNNERS list")
    ap.add_argument("--only", help="comma-separated runner names to include")
    args = ap.parse_args()

    runners = RUNNERS
    if args.config:
        runners = json.load(open(args.config, encoding="utf-8"))
    if args.only:
        want = {s.strip().lower() for s in args.only.split(",")}
        runners = [r for r in runners if r["name"].lower() in want]

    out = {"meta": {"max_tokens": MAX_TOKENS, "metric": "wall-clock tokens/sec",
                    "note": "same model+quant across runners; greedy, 256 tokens, shared 12-prompt suite"},
           "runners": []}
    for r in runners:
        out["runners"].append(bench_runner(r))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
