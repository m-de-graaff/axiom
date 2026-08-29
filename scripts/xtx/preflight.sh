#!/usr/bin/env bash
set -euo pipefail
export LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1
cd /mnt/d/Development/hobby/axiom
set -a; . ./.env; set +a
"$HOME/.venvs/axiom-rocm/bin/python" - <<'PY'
import os
import torch, wandb
print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("wandb", wandb.__version__)
k = os.environ.get("WANDB_API_KEY", "")
print("key_len", len(k), "clean", k == k.strip())
PY
