#!/usr/bin/env bash
set -euo pipefail
export LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1
cd /mnt/d/Development/hobby/axiom
set -a; . ./.env; set +a
mkdir -p research/p3-00d
exec "$HOME/.venvs/axiom-rocm/bin/python" -u -m axiom_eval.cli run \
  --config configs/eval/val.yaml --timeframes 1h \
  --models axiom-zero-small persistence ewma lightgbm \
  --device cuda:0 > research/p3-00d/eval.log 2>&1
