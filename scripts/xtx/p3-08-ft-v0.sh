#!/usr/bin/env bash
# First fine-tune vs zero-shot on val (P3-08 first look) — run inside WSL2 on the XTX.
set -euo pipefail
export LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1
cd /mnt/d/Development/hobby/axiom
set -a; . ./.env; set +a
mkdir -p research/p3-08
exec "$HOME/.venvs/axiom-rocm/bin/python" -u -m axiom_eval.cli run \
  --config configs/eval/val.yaml --timeframes 1h \
  --models axiom-ft-25m-crypto1-512-v0 \
  --device cuda:0 > research/p3-08/ft-v0-val.log 2>&1
