# Machine / backend notes

Record here: torch versions + install commands per machine (laptop CPU wheel,
XTX ROCm wheel), and every "works on one backend, breaks on another" incident
with its workaround. Rules of the road live in CLAUDE.md (SDPA-only attention,
no CUDA-only deps, keep --no-compile working, never assume a local GPU).

## Machines

| Machine | Backend | Install command | torch |
|---|---|---|---|
| Laptop (Windows 11) | CPU | `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` | 2.13.0+cpu |
| Modal (T4/L4/A10G/A100) | CUDA | baked into the Modal image (`pip_install("torch")`) | 2.13.0 (cu13) |
| XTX box (RX 7900 XTX, gfx1100) | ROCm | set up previously for another project | **to record — run the checklist below** |

## Incidents

- 2026-08-28 — Windows: the system `pip` launcher is broken (points at a
  removed `C:\Python314\python.exe`). Install CLI tools with
  `uv tool install <pkg>` instead; `modal` 1.5.4 is installed that way.
- 2026-08-28 — `uv sync` uninstalls torch, because torch is deliberately absent
  from every `pyproject.toml` and `uv sync` prunes anything undeclared. Reinstall
  the per-machine wheel after a sync (or use `uv sync --inexact`). CI does the
  same thing in order: `uv sync`, then the CPU wheel.

## The XTX checklist

Everything below runs on the AMD box and nowhere else. Nothing here is blocking Phase
3 code work; the parity leg is blocking any `axiom-runtime-*` tag.

```bash
# 0. environment — torch presents ROCm as `cuda`, so device="cuda:0" is correct
rocminfo | grep gfx                      # expect gfx1100
uv sync                                  # note: this prunes torch (see incidents)
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.4   # check the current tag
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"

# 1. the suite, on this backend
uv run ruff check .
uv run pytest -q                         # test_parity.py runs on CPU here; it must pass first

# 2. the ROCm parity leg (P4-02) — the actual gate item
uv run python scripts/rocm_check.py --json research/rocm/parity.json
```

`rocm_check.py` prints a markdown row for the table below. **Pass condition is
`token_identical: True`** on every model: the cached generation loop must produce the
same bars on ROCm that it does on CPU and CUDA. If it does not, record it in the
incidents table with the diff magnitude — do not work around it silently, and do not
tag a runtime release until it is understood.

### Parity + speed of the KV cache (P4-04)

64 samples, 24 steps, context = `max_context - horizon`, real weights.

| model | backend | cached | uncached | speedup | token-identical |
|---|---|---|---|---|---|
| `axiom-zero-base` | NVIDIA L4 (torch 2.13.0+cu130) | 2.82s | 25.78s | 9.1x | True |
| `axiom-zero-small` | NVIDIA L4 (torch 2.13.0+cu130) | 1.42s | 1.66s | 1.2x (16 samples) | True |
| `axiom-zero-small` | Windows CPU (torch 2.13.0+cpu) | 0.27s | 0.68s | 2.5x (2 samples, 4 steps) | True |
| `axiom-zero-small` | ROCm gfx1100 | — | — | — | *to run* |
| `axiom-zero-base` | ROCm gfx1100 | — | — | — | *to run* |

### Optional while you are on that box

- **Second-machine eval leg.** `uv run axiom-eval run --config configs/eval/default.yaml
  --device cuda:0 --timeframes 1h` cross-checks the Modal numbers for free. Model MC
  sampling will not match bitwise across devices — compare distributions, not bits.
- **Fine-tune iteration (P3-03/04).** `axiom-zero-small` is 24.7M parameters; it fits
  this card with room to spare, and it is the model the zero-shot grid says to start
  from (`docs/results/p2-zero-shot.md`).
- **`torch.compile`.** Not wired up yet (P4-07). When it is, ROCm breakages get logged
  here, not worked around.
