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
| XTX box (RX 7900 XTX, gfx1100) | ROCm 7.14 via **WSL2** | see the XTX checklist below — `--index-url https://download.pytorch.org/whl/rocm7.2` | 2.13.0+rocm7.2 |

## Incidents

- 2026-08-28 — Windows: the system `pip` launcher is broken (points at a
  removed `C:\Python314\python.exe`). Install CLI tools with
  `uv tool install <pkg>` instead; `modal` 1.5.4 is installed that way.
- 2026-08-28 — `uv sync` uninstalls torch, because torch is deliberately absent
  from every `pyproject.toml` and `uv sync` prunes anything undeclared. Reinstall
  the per-machine wheel after a sync (or use `uv sync --inexact`). CI does the
  same thing in order: `uv sync`, then the CPU wheel.

- 2026-08-29 — **There is no ROCm torch wheel for Windows.** The XTX box runs
  Windows, and the checklist below used to be written as if it ran Linux.
  `uv pip install torch --index-url .../rocm6.4` fails on `win_amd64`: the index
  only publishes `manylinux_2_28_x86_64`. `rocminfo` likewise does not exist on the
  Windows side. The ROCm leg runs in **WSL2 (Ubuntu 24.04)**, which already has
  ROCm 7.14 with gfx1100 kernels installed. Checklist rewritten accordingly.
- 2026-08-29 — The failed ROCm install left the box with **no torch at all**,
  because `uv sync` had already pruned the CPU wheel (previous incident) and the
  ROCm install then errored out. `pytest` collapsed with `ModuleNotFoundError:
  No module named 'torch'` on three modules. Fix is the CPU wheel reinstall; the
  two incidents compound, so re-run the suite after any torch install, not just
  after `uv sync`.
- 2026-08-29 — **The `rocm6.4` index tag was stale.** It tops out at torch 2.9.1,
  while CPU and Modal CUDA are both on 2.13.0. `rocm7.2` publishes
  `torch-2.13.0+rocm7.2` for cp312, so all three legs compare the same torch
  version. Check the tag each time rather than pasting the one in this file.
- 2026-08-29 — **torch's bundled HSA runtime does not work under WSL.** The wheel
  ships its own `torch/lib/libhsa-runtime64.so`, which looks for
  `/sys/class/kfd/kfd/topology/nodes` — a native-Linux path that WSL does not have
  (WSL routes the GPU through `/dev/dxg`). Symptom: `torch.cuda.is_available()` is
  `False` and `_cuda_init()` raises `No CUDA GPUs are available`, while `rocminfo`
  happily reports gfx1100. Fix without touching the wheel:

      export LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1

  AMD's own instructions say to delete the bundled library and copy the system one
  over it; the `LD_PRELOAD` above is the reversible version of the same thing, and
  it survives a torch reinstall. Every ROCm command below needs it.
- 2026-08-29 — `rocm-smi` does not work in WSL (`Driver not initialized (amdgpu not
  found in modules)`) because there is no `amdgpu` kernel module there. This is
  expected and not a fault. Use `rocminfo` to confirm the card, and
  `/opt/rocm-wsl/bin/amd-smi` for telemetry.
- 2026-08-29 — **`parity_and_speed` has no warmup run, so the first model measured
  pays one-time kernel-load cost and reports a nonsense speedup.** Measured on the
  XTX: run `small` first and it reports 3.11s cached / 1.58s uncached (0.5x) — its
  "cached" number larger than `base`'s, which cannot be true. Run `base` first and
  `small` reports 0.53s / 3.97s (7.5x). Parity itself is unaffected
  (`token_identical: True`, `max_abs_diff: 0.0` in both orderings) — only the timings
  move. The recorded L4 `small` row (1.2x) is suspect for the same reason. Fix is a
  discarded warmup generate before timing; not done yet, because it invalidates the
  committed CUDA reference numbers until they are re-run.

## The XTX checklist

Everything below runs on the AMD box and nowhere else. Nothing here is blocking Phase
3 code work; the parity leg is blocking any `axiom-runtime-*` tag.

**The XTX box runs Windows, and ROCm torch is Linux-only — so the ROCm leg runs inside
WSL2, not in the Windows shell.** The Windows side of this box keeps its CPU wheel and
is where `ruff`/`pytest` normally run; the WSL side exists purely for the GPU legs. Two
environments, one checkout on `/mnt/d` — so the WSL env lives outside the repo
(`~/.venvs/axiom-rocm`) and never collides with the Windows `.venv`.

```powershell
# 0. Windows side — keep the CPU env working (uv sync prunes torch; see incidents)
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv run ruff check .
uv run pytest -q
```

```bash
# 1. WSL side — enter it, and confirm the card. `rocm-smi` does NOT work here; rocminfo does.
wsl -d Ubuntu-24.04
rocminfo | grep gfx1100                  # expect gfx1100 / Radeon RX 7900 XTX
cd /mnt/d/Development/hobby/axiom

# 2. build the Linux env out-of-tree, without disturbing the Windows .venv or uv.lock
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/axiom-rocm UV_LINK_MODE=copy
uv sync --frozen --python 3.12
# `uv pip install` ignores UV_PROJECT_ENVIRONMENT — point it at the interpreter:
# check the index tag each time — rocm6.4 is stale (tops out at torch 2.9.1)
uv pip install --python $HOME/.venvs/axiom-rocm/bin/python torch --index-url https://download.pytorch.org/whl/rocm7.2

# 3. the WSL shim — torch's bundled HSA runtime wants /sys/class/kfd, which WSL lacks.
#    Without this, torch.cuda.is_available() is False on a perfectly working card.
export LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1
$HOME/.venvs/axiom-rocm/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"

# 4. the suite, on this backend
$HOME/.venvs/axiom-rocm/bin/python -m pytest -q

# 5. the ROCm parity leg (P4-02) — the actual gate item
$HOME/.venvs/axiom-rocm/bin/python scripts/rocm_check.py --json research/rocm/parity.json
```

Timings from step 5 are only trustworthy for the *second* model onward — the harness has
no warmup run, so whichever model goes first absorbs kernel-load cost (see incidents).
`token_identical` is unaffected by this and is the thing being gated on.

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
| `axiom-zero-base` | ROCm gfx1100, WSL2 (torch 2.13.0+rocm7.2) | 1.95s | 13.23s | 6.8x | True |
| `axiom-zero-small` | ROCm gfx1100, WSL2 (torch 2.13.0+rocm7.2) | 0.53s | 3.97s | 7.5x | True |

ROCm leg run 2026-08-29 on the XTX under WSL2, 64 samples / 24 steps / 488 context, real
weights, torch 2.13.0+rocm7.2 — the same torch version as the CPU and L4 legs.
**`token_identical: True` and `max_abs_diff: 0.0` on both models**, in both orderings, so
the cached generation loop produces the same bars on ROCm that it does on CPU and CUDA.
P3-00a's pass condition is met.

The two rows above come from the run with `base` first, so `small`'s numbers are the
post-warmup ones; the `small`-first ordering reported 3.11s / 1.58s / 0.5x for the same
model, which is the warmup artifact and not a real result. `base` moved 9.7x → 6.8x
between the two orderings, so treat one significant figure as the real precision here
until the harness discards a warmup pass.

### Optional while you are on that box

- **Second-machine eval leg.** `uv run axiom-eval run --config configs/eval/default.yaml
  --device cuda:0 --timeframes 1h` cross-checks the Modal numbers for free. Model MC
  sampling will not match bitwise across devices — compare distributions, not bits.
- **Fine-tune iteration (P3-03/04).** `axiom-zero-small` is 24.7M parameters; it fits
  this card with room to spare, and it is the model the zero-shot grid says to start
  from (`docs/results/p2-zero-shot.md`).
- **`torch.compile`.** Not wired up yet (P4-07). When it is, ROCm breakages get logged
  here, not worked around.
