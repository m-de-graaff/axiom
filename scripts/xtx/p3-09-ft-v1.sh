#!/usr/bin/env bash
# Fine-tune run 2 (v1) on val — run inside WSL2 on the XTX.
set -euo pipefail
export LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1
cd /mnt/d/Development/hobby/axiom
set -a; . ./.env; set +a
mkdir -p research/p3-09
exec "$HOME/.venvs/axiom-rocm/bin/python" -u -m axiom_eval.cli run \
  --config configs/eval/val.yaml --timeframes 1h \
  --models axiom-ft-25m-crypto1-512-v1 \
  --device cuda:0 > research/p3-09/ft-v1-val.log 2>&1
