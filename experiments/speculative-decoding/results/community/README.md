# Community results

Each file here is one contributor's run on their own hardware, produced by
`python scripts/make_submission.py --name <label>`.

See the repo [CONTRIBUTING.md](../../../../CONTRIBUTING.md) for how to add yours.

Schema (per file):

```jsonc
{
  "name": "rtx4090-ryzen7950x",
  "notes": "llama.cpp b9999, Q4_K_M",
  "hardware": {
    "platform": "...", "python": "...", "cpu": "...",
    "gpus": [{ "name": "...", "memory": "...", "driver": "...", "compute_cap": "..." }]
  },
  "runs": {
    "gpu0":   { "...": "raw results.json" },
    "cpu":    { "...": "raw results_cpu.json" }
  }
}
```

All submissions are licensed CC BY 4.0.
