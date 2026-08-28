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
| XTX box | ROCm | set up previously for another project | to record from that machine |

## Incidents

- 2026-08-28 — Windows: the system `pip` launcher is broken (points at a
  removed `C:\Python314\python.exe`). Install CLI tools with
  `uv tool install <pkg>` instead; `modal` 1.5.4 is installed that way.
- 2026-08-28 — `uv sync` uninstalls torch, because torch is deliberately absent
  from every `pyproject.toml` and `uv sync` prunes anything undeclared. Reinstall
  the per-machine wheel after a sync (or use `uv sync --inexact`). CI does the
  same thing in order: `uv sync`, then the CPU wheel.
