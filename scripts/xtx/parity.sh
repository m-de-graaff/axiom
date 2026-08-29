#!/usr/bin/env bash
# ROCm parity + speed leg (P4-02/B-09) — run inside WSL2 on the XTX.
set -euo pipefail
export LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1
cd /mnt/d/Development/hobby/axiom
mkdir -p research/rocm
exec "$HOME/.venvs/axiom-rocm/bin/python" -u scripts/rocm_check.py --json research/rocm/parity.json
