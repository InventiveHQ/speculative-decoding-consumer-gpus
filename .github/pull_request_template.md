<!-- Submitting benchmark results? Most of this is auto-filled by make_submission.py. -->

## What is this PR?

- [ ] **Results submission** (adds a file under `results/community/`)
- [ ] Code / docs change
- [ ] Other

## For results submissions

- **Hardware:** <!-- e.g. RTX 4090 + Ryzen 9 7950X — full specs (CPU/RAM/GPU+PCIe/board/storage) are auto-captured in the submission's "hardware" block by make_submission.py -->
- **llama.cpp version:** <!-- e.g. b9789 -->
- **Backends run:** <!-- cuda / vulkan / cpu -->
- **Models / quant:** <!-- e.g. Qwen2.5-Coder-7B Q4_K_S + 0.5B Q8_0 draft -->

Checklist:
- [ ] I ran `python scripts/make_submission.py --name <label>` and added the
      resulting `results/community/<label>.json`
- [ ] I used the default methodology (greedy, 256 tokens, the bundled 12-prompt
      suite) — or noted any deviations in the submission's `notes`
- [ ] I agree to license this data under CC BY 4.0

Anything surprising in your numbers? Tell us here 👇
