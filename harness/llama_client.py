"""Reusable benchmarking core for the local-llm-benchmarks series.

Extracted from the speculative-decoding experiment's proven bench.py. Two layers:

  - LlamaServer: launch/health/stop a llama.cpp `llama-server` with given exe, model,
    flags, env, and device pinning. Use as a context manager.
  - chat() / measure() / run_suite(): talk to ANY OpenAI-compatible endpoint
    (llama.cpp, Ollama, LM Studio, vLLM...) and measure generation throughput.
    measure() reports both the server's own `timings.predicted_per_second` (when
    present, e.g. llama.cpp) and a wall-clock tok/s (works for every runner).

A new experiment is then ~20 lines: define the configs to sweep, start a server
(or point at a running one), call run_suite(), dump JSON. See harness/README.md.
"""
import json, socket, subprocess, time, urllib.request, urllib.error

DEFAULT_HOST = "127.0.0.1"


def wait_health(port, host=DEFAULT_HOST, timeout=180, proc=None):
    """Poll /health until ok. Returns True/False. Stops early if proc exits."""
    t0 = time.time()
    url = f"http://{host}:{port}/health"
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def port_free(port, host=DEFAULT_HOST):
    s = socket.socket()
    try:
        s.bind((host, port)); return True
    except OSError:
        return False
    finally:
        s.close()


def chat(port, prompt, host=DEFAULT_HOST, max_tokens=256, temperature=0.0,
         top_k=1, seed=1234, model=None, timeout=300, extra=None):
    """POST /v1/chat/completions. Returns the parsed JSON response."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "top_k": top_k,
        "seed": seed, "cache_prompt": True,
    }
    if model:
        body["model"] = model
    if extra:
        body.update(extra)
    data = json.dumps(body).encode()
    # Connection: close avoids keep-alive resets that flaked on long CPU generations.
    headers = {"Content-Type": "application/json", "Connection": "close"}
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"http://{host}:{port}/v1/chat/completions", data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last = e
            time.sleep(1.5)
    raise last


def measure(port, prompt, **kw):
    """Run one prompt; return {tps, tps_wallclock, predicted_n, draft_*}.

    `tps` prefers the server's own generation rate (llama.cpp timings); falls back
    to wall-clock. `tps_wallclock` is always end-to-end completion_tokens/elapsed.
    """
    t0 = time.time()
    resp = chat(port, prompt, **kw)
    elapsed = time.time() - t0
    timings = resp.get("timings", {}) or {}
    usage = resp.get("usage", {}) or {}
    n = timings.get("predicted_n") or usage.get("completion_tokens") or 0
    server_tps = timings.get("predicted_per_second")
    wall_tps = (n / elapsed) if elapsed > 0 and n else None
    text = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return {
        "tps": server_tps if server_tps else wall_tps,
        "tps_wallclock": round(wall_tps, 2) if wall_tps else None,
        "predicted_n": n,
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "text": text,
    }


def run_suite(port, prompts, max_tokens=256, warmup=True, on_result=None, **kw):
    """Replay a prompt suite (list of {id,category,prompt}); return record list."""
    if warmup:
        try:
            chat(port, "Say hello.", max_tokens=16, **kw)
        except Exception:
            pass
    records = []
    for pr in prompts:
        try:
            m = measure(port, pr["prompt"], max_tokens=max_tokens, **kw)
            rec = {"id": pr["id"], "category": pr.get("category"),
                   "tps": round(m["tps"], 2) if m["tps"] else None,
                   "predicted_n": m["predicted_n"],
                   "draft_n": m["draft_n"], "draft_n_accepted": m["draft_n_accepted"]}
        except Exception as e:
            rec = {"id": pr["id"], "category": pr.get("category"), "error": str(e)}
        records.append(rec)
        if on_result:
            on_result(rec)
    return records


class LlamaServer:
    """Context manager that launches a llama.cpp llama-server pinned to one device.

    backend: 'cuda' | 'vulkan' | 'cpu'. For cpu, layers stay on CPU (-ngl 0
    --device none). For gpu backends, all layers go on `gpu_index` (the right env
    var is set per backend so a single physical GPU is isolated).
    """
    def __init__(self, exe, model, backend="cuda", gpu_index=0, extra_args=None,
                 port=8099, host=DEFAULT_HOST, ctx=4096, ngl=999, jinja=True,
                 log_path=None, env=None):
        self.exe, self.model, self.backend = exe, model, backend
        self.gpu_index, self.extra_args = gpu_index, list(extra_args or [])
        self.port, self.host, self.ctx, self.ngl = port, host, ctx, ngl
        self.jinja, self.log_path, self.env_extra = jinja, log_path, dict(env or {})
        self.proc = None

    def _args(self):
        a = [self.exe, "--host", self.host, "--port", str(self.port),
             "--model", self.model, "-c", str(self.ctx), "--no-webui"]
        if self.backend == "cpu":
            a += ["-ngl", "0", "--device", "none"]
        else:
            a += ["-ngl", str(self.ngl), "--split-mode", "none", "-mg", "0"]
        if self.jinja:
            a += ["--jinja"]
        return a + self.extra_args

    def _env(self):
        import os
        e = dict(os.environ)
        if self.backend == "cuda":
            e["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        elif self.backend == "vulkan":
            e["GGML_VK_VISIBLE_DEVICES"] = str(self.gpu_index)
        e.update(self.env_extra)
        return e

    def start(self):
        out = open(self.log_path, "w", encoding="utf-8", errors="ignore") if self.log_path else subprocess.DEVNULL
        self._log = out
        self.proc = subprocess.Popen(self._args(), stdout=out,
                                     stderr=subprocess.STDOUT, env=self._env())
        if not wait_health(self.port, self.host, proc=self.proc):
            self.stop()
            raise RuntimeError("llama-server failed to start (see log)")
        return self

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
        if getattr(self, "_log", None) not in (None, subprocess.DEVNULL):
            try:
                self._log.close()
            except Exception:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        time.sleep(2)
        return False
