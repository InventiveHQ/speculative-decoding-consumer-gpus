# harness/ — shared benchmarking core

Write the engine once; each experiment is a thin folder under `experiments/`.

- **`prompts.py`** — the canonical 12-prompt suite (code / math / reasoning /
  summarization / chat). Import it so every experiment is comparable.
- **`llama_client.py`** — `LlamaServer` (launch/health/stop a `llama-server`
  pinned to one device) + `chat()` / `measure()` / `run_suite()` that work against
  any OpenAI-compatible endpoint (llama.cpp, Ollama, LM Studio, vLLM). `measure()`
  reports the server's own `predicted_per_second` when available *and* a wall-clock
  tok/s, so cross-runner comparisons are fair.

## A new experiment in ~20 lines

```python
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "harness"))
from prompts import PROMPTS
from llama_client import LlamaServer, run_suite

MODEL = os.environ["MODEL"]; EXE = os.environ["LLAMA_EXE"]
runs = []
for ctx in (2048, 4096, 8192, 16384):          # <- the variable you're sweeping
    with LlamaServer(EXE, MODEL, backend="cuda", ctx=ctx,
                     log_path=f"results/server-{ctx}.log") as s:
        recs = run_suite(s.port, PROMPTS, on_result=lambda r: print(ctx, r.get("tps")))
    runs.append({"label": f"ctx{ctx}", "records": recs})
json.dump({"runs": runs}, open("results/results.json", "w"), indent=2)
```

Then write an `aggregate.py` (turn `runs` into chart-ready `data.json`) and an
`inject.py` (bake it into a `site/index.html`). The speculative-decoding experiment
is the worked reference for all three steps.

> Experiment #1 (`experiments/speculative-decoding`) predates this harness and ships
> its own self-contained `scripts/` (the frozen code behind its published results).
> New experiments should build on `harness/`; #1 can be ported here on its next
> re-run.
