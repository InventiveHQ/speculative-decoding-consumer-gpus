"""Speculative-decoding benchmark for a single consumer Blackwell GPU (RTX 5060 Ti).

For every (backend x method) it launches llama-server pinned to GPU 0, replays a
fixed prompt set (see prompts.py), and records the server-reported generation
rate (tokens/sec), output token count, and an output hash so we can prove the
speculative runs produce byte-identical text to the baseline (greedy).

Usage:
    python bench.py --out ..\results\results.json
"""
import argparse, hashlib, json, os, socket, subprocess, sys, time, urllib.request, urllib.error
from prompts import PROMPTS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# --- Paths are configurable so anyone can reproduce. Override via env vars or CLI. ---
#   SPECBENCH_MODELS_DIR  : folder holding the .gguf files       (default ./models)
#   SPECBENCH_CUDA_EXE    : path to the CUDA llama-server.exe     (default ./runtimes/cuda/llama-server.exe)
#   SPECBENCH_VULKAN_EXE  : path to the Vulkan llama-server.exe   (default ./runtimes/vulkan/llama-server.exe)
MODELS_DIR = os.environ.get("SPECBENCH_MODELS_DIR", os.path.join(ROOT, "models"))

# Named target presets -> filename inside MODELS_DIR (download these yourself; see README).
TARGETS = {
    "qwen7b":  "Qwen2.5-Coder-7B-Instruct-Q4_K_S.gguf",
    "qwen14b": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
    "qwen3b":  "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
}
DRAFT_FILE = "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"

TARGET = os.path.join(MODELS_DIR, TARGETS["qwen7b"])
DRAFT  = os.path.join(MODELS_DIR, DRAFT_FILE)

CUDA_EXE   = os.environ.get("SPECBENCH_CUDA_EXE",   os.path.join(ROOT, "runtimes", "cuda", "llama-server.exe"))
VULKAN_EXE = os.environ.get("SPECBENCH_VULKAN_EXE", os.path.join(ROOT, "runtimes", "vulkan", "llama-server.exe"))
# Apple Silicon / Metal: a stock llama-server (e.g. `brew install llama.cpp`) offloads
# to the Apple GPU with -ngl 999. Defaults to llama-server on PATH; override if needed.
METAL_EXE  = os.environ.get("SPECBENCH_METAL_EXE", "llama-server")

PORT = 8099
HOST = "127.0.0.1"
MAX_TOKENS = 256
CTX = 4096

def make_backends(gpu):
    g = str(gpu)
    return [
        {"name": "cuda",   "exe": CUDA_EXE,   "env": {"CUDA_VISIBLE_DEVICES": g}},
        {"name": "vulkan", "exe": VULKAN_EXE, "env": {"GGML_VK_VISIBLE_DEVICES": g}},
        {"name": "metal",  "exe": METAL_EXE,  "env": {}},
    ]

BACKENDS = make_backends(0)
CPU_MODE = False

# spec-draft-n-max 5 = draft 5 tokens per step (NVIDIA-style block drafting).
METHODS = [
    {"name": "baseline",   "label": "No speculation",        "args": []},
    {"name": "draft_0_5b", "label": "Draft model (0.5B)",    "args": ["-md", DRAFT, "--spec-type", "draft-simple", "--spec-draft-n-max", "5", "-ngld", "999"]},
    {"name": "ngram",      "label": "N-gram (model-free)",   "args": ["--spec-type", "ngram-cache", "--spec-draft-n-max", "5"]},
]


def wait_health(timeout=180):
    t0 = time.time()
    url = f"http://{HOST}:{PORT}/health"
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def port_free():
    s = socket.socket()
    try:
        s.bind((HOST, PORT)); return True
    except OSError:
        return False
    finally:
        s.close()


def chat(prompt, max_tokens=MAX_TOKENS):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1234,
        "cache_prompt": False,
        "timings_per_token": False,
    }).encode()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def launch(backend, method):
    if CPU_MODE:
        args = [backend["exe"], "--host", HOST, "--port", str(PORT),
                "--model", TARGET, "-ngl", "0", "--device", "none",
                "-c", str(CTX), "--jinja", "--no-webui"]
    else:
        args = [backend["exe"], "--host", HOST, "--port", str(PORT),
                "--model", TARGET, "-ngl", "999", "-c", str(CTX),
                "--split-mode", "none", "-mg", "0", "--jinja", "--no-webui"]
    args += method["args"]
    env = dict(os.environ)
    env.update(backend["env"])
    log = open(os.path.join(ROOT, "results", f"server-{backend['name']}-{method['name']}.log"), "w", encoding="utf-8", errors="ignore")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    return p, log


def run_config(backend, method):
    print(f"\n=== {backend['name']} / {method['name']} ===", flush=True)
    if not port_free():
        print("  port busy, waiting...", flush=True); time.sleep(5)
    p, log = launch(backend, method)
    try:
        if not wait_health():
            print("  SERVER FAILED TO START", flush=True)
            return {"status": "server_failed"}
        # warmup
        try:
            chat("Say hello.", max_tokens=16)
        except Exception as e:
            print(f"  warmup err: {e}", flush=True)
        records = []
        for pr in PROMPTS:
            try:
                resp = chat(pr["prompt"])
            except Exception as e:
                print(f"  {pr['id']}: ERR {e}", flush=True)
                records.append({"id": pr["id"], "category": pr["category"], "error": str(e)})
                continue
            timings = resp.get("timings", {}) or {}
            usage = resp.get("usage", {}) or {}
            text = resp["choices"][0]["message"]["content"]
            tps = timings.get("predicted_per_second")
            rec = {
                "id": pr["id"], "category": pr["category"],
                "tps": tps,
                "predicted_n": timings.get("predicted_n", usage.get("completion_tokens")),
                "predicted_ms": timings.get("predicted_ms"),
                "draft_n": timings.get("draft_n"),
                "draft_n_accepted": timings.get("draft_n_accepted"),
                "out_hash": hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12],
                "out_len_chars": len(text),
            }
            records.append(rec)
            print(f"  {pr['id']:<20} {tps:>7.1f} tok/s  ({rec['predicted_n']} tok)" if tps else f"  {pr['id']}: no timings", flush=True)
        return {"status": "ok", "records": records}
    finally:
        p.terminate()
        try:
            p.wait(timeout=20)
        except Exception:
            p.kill()
        log.close()
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    ap.add_argument("--backends", default="cuda,vulkan")
    ap.add_argument("--target", default="qwen7b", choices=list(TARGETS))
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0, help="physical GPU index (0=5060 Ti, 1=1080 Ti)")
    ap.add_argument("--cpu", action="store_true", help="run on CPU only (-ngl 0 --device none)")
    ap.add_argument("--gpu-label", default=None, help="override the device name recorded in meta (e.g. 'Apple M3 Max (Metal)')")
    args = ap.parse_args()
    global CTX, BACKENDS, CPU_MODE
    if args.ctx:
        CTX = args.ctx
    if args.cpu:
        CPU_MODE = True
        BACKENDS = [{"name": "cpu", "exe": VULKAN_EXE, "env": {}}]
        args.backends = "cpu"
        print("CPU MODE (-ngl 0 --device none)")
    else:
        BACKENDS = make_backends(args.gpu)
        print(f"GPU index = {args.gpu}")
    want = set(args.backends.split(","))
    global TARGET
    TARGET = os.path.join(MODELS_DIR, TARGETS[args.target])
    print(f"TARGET = {TARGET}")

    gpu_name = args.gpu_label or ("CPU" if args.cpu else {0: "RTX 5060 Ti (Blackwell, sm_120)", 1: "GTX 1080 Ti (Pascal, sm_61)"}.get(args.gpu, f"GPU {args.gpu}"))
    results = {"meta": {"target": TARGET, "draft": DRAFT, "max_tokens": MAX_TOKENS,
                        "ctx": CTX, "gpu": gpu_name, "gpu_index": args.gpu}, "runs": []}
    for backend in BACKENDS:
        if backend["name"] not in want:
            continue
        for method in METHODS:
            out = run_config(backend, method)
            results["runs"].append({
                "backend": backend["name"], "method": method["name"],
                "label": method["label"], **out})
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
