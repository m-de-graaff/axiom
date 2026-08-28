#!/usr/bin/env bash
# =============================================================================
# AXIOM scaffold — empty folder -> full monorepo skeleton.
#
# Works on ANY machine, including a CPU-only laptop: local GPU is optional,
# all GPU tasks route to Modal (see infra/modal_app/smoke.py).
#
# Usage:
#   1) Put scaffold.sh, CLAUDE.md, TODO.md, README.md (and optionally
#      AXIOM_BUILD_ORDER.md) into an EMPTY folder.
#   2) bash scaffold.sh
#   3) Follow the printed next steps (uv sync, torch-per-machine, git subtree).
#
# The script is idempotent-ish but designed for a one-shot run in an empty dir.
# Set FORCE=1 to bypass the empty-directory check (at your own risk).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- safety check
extra="$(ls -A 2>/dev/null | grep -Ev '^(scaffold\.sh|CLAUDE\.md|TODO\.md|README\.md|AXIOM_BUILD_ORDER\.md|\.git)$' || true)"
if [ -n "${extra}" ] && [ -z "${FORCE:-}" ]; then
  echo "ERROR: directory not empty (found: $(echo "${extra}" | tr '\n' ' '))."
  echo "Run in an empty folder containing only the starter files, or FORCE=1 bash scaffold.sh"
  exit 1
fi

echo "==> Scaffolding Axiom in $(pwd)"

# ---------------------------------------------------------------- git + dirs
[ -d .git ] || git init -q

mkdir -p configs/data configs/finetune configs/eval configs/pretrain \
         packages infra/modal_app services/signal_api apps \
         scripts tests db docs research/day1 reports vendor
touch apps/.gitkeep reports/.gitkeep vendor/.gitkeep

[ -f AXIOM_BUILD_ORDER.md ] && mv AXIOM_BUILD_ORDER.md docs/AXIOM_BUILD_ORDER.md \
  && echo "    moved AXIOM_BUILD_ORDER.md -> docs/"

# ---------------------------------------------------------------- root pyproject
cat > pyproject.toml <<'EOF_PYROOT'
[project]
name = "axiom"
version = "0.0.1"
description = "Axiom: finance-native TSFM (Kronos fork) + signals + trading loop"
requires-python = ">=3.11"
# NOTE: torch is intentionally NOT listed — install per machine (CPU wheel on
# the laptop, ROCm wheel on the XTX box, CUDA inside the Modal image).
# See CLAUDE.md.
dependencies = [
  "axiom-model",
  "axiom-data",
  "axiom-eval",
  "axiom-signals",
  "axiom-trader",
]

[dependency-groups]
dev = ["ruff>=0.6", "pytest>=8"]

[tool.uv]
package = false

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
axiom-model = { workspace = true }
axiom-data = { workspace = true }
axiom-eval = { workspace = true }
axiom-signals = { workspace = true }
axiom-trader = { workspace = true }

[tool.ruff]
line-length = 100
extend-exclude = ["vendor", "apps", "data", "research"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
EOF_PYROOT

# ---------------------------------------------------------------- packages
write_pkg() {
  local name="$1"; shift
  local dir="packages/axiom_${name}"
  mkdir -p "${dir}/axiom_${name}"
  local deps=""
  for d in "$@"; do deps="${deps}\"${d}\", "; done
  deps="${deps%, }"
  cat > "${dir}/pyproject.toml" <<PKGTOML
[project]
name = "axiom-${name}"
version = "0.0.1"
description = "Axiom ${name} package"
requires-python = ">=3.11"
dependencies = [${deps}]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
PKGTOML
  cat > "${dir}/axiom_${name}/__init__.py" <<PKGINIT
"""axiom_${name} — see docs/AXIOM_BUILD_ORDER.md and CLAUDE.md."""
PKGINIT
}

write_pkg model   numpy einops huggingface_hub safetensors pyyaml pandas
write_pkg data    pandas pyarrow duckdb httpx pyyaml numpy
write_pkg eval    pandas numpy scipy pyyaml
write_pkg signals numpy pandas pyyaml
write_pkg trader  pydantic pyyaml

# ---------------------------------------------------------------- .gitignore
cat > .gitignore <<'EOF_GITIGNORE'
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.env
data/
reports/
wandb/
checkpoints/
*.ckpt
node_modules/
.next/
dist/
.DS_Store
EOF_GITIGNORE

# ---------------------------------------------------------------- LICENSE + NOTICE
YEAR="$(date +%Y)"
cat > LICENSE <<EOF_LICENSE
MIT License

Copyright (c) ${YEAR} Axiom author

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF_LICENSE

cat > NOTICE <<'EOF_NOTICE'
Axiom includes and derives from Kronos:
  https://github.com/shiyu-coder/Kronos
  Copyright (c) the Kronos authors (Yu Shi et al.)
  Licensed under the MIT License.

Citation:
  Shi et al., "Kronos: A Foundation Model for the Language of Financial
  Markets", arXiv:2508.02739 (accepted at AAAI 2026).

Pretrained weights loaded from Hugging Face "NeoQuasar" repositories retain
their original licenses — check each model card before redistributing
derived weights.
EOF_NOTICE

# ---------------------------------------------------------------- .env.example
cat > .env.example <<'EOF_ENVEX'
# Copy to .env (gitignored). Cloud equivalents live in Modal Secrets.
POSTGRES_URL=postgresql://user:pass@host:5432/axiom
WANDB_API_KEY=
TELEGRAM_WEBHOOK_URL=
# --- trading (Phase 8+) ---
EXCHANGE=bybit_testnet
BYBIT_TESTNET_KEY=
BYBIT_TESTNET_SECRET=
AXIOM_LIVE=0          # hard gate: live orders require AXIOM_LIVE=1 AND risk-engine approval
AXIOM_MODEL=axiom-zero-base
EOF_ENVEX

# ---------------------------------------------------------------- CI
mkdir -p .github/workflows
cat > .github/workflows/ci.yml <<'EOF_CIYML'
name: ci
on: [push, pull_request]
jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.11"
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest -q
EOF_CIYML

# ---------------------------------------------------------------- DB schema
cat > db/schema.sql <<'EOF_SCHEMA'
-- Axiom Postgres schema (see build order §8.2).
-- With TimescaleDB, optionally: SELECT create_hypertable('candles','ts');

CREATE TABLE IF NOT EXISTS candles (
  symbol text NOT NULL,
  tf     text NOT NULL,
  ts     timestamptz NOT NULL,
  o double precision, h double precision, l double precision,
  c double precision, v double precision,
  PRIMARY KEY (symbol, tf, ts)
);

CREATE TABLE IF NOT EXISTS runs (
  run_id   uuid PRIMARY KEY,
  kind     text NOT NULL,            -- 'infer_cron' | 'train' | 'eval' | ...
  git_sha  text,
  config   jsonb,
  started  timestamptz NOT NULL DEFAULT now(),
  finished timestamptz,
  status   text NOT NULL DEFAULT 'running',
  error    text
);

CREATE TABLE IF NOT EXISTS forecasts (
  run_id   uuid NOT NULL REFERENCES runs(run_id),
  symbol   text NOT NULL,
  tf       text NOT NULL,
  made_at  timestamptz NOT NULL,     -- close time of last observed bar
  horizon  int  NOT NULL,            -- bars ahead
  quantiles jsonb NOT NULL,          -- {"q10":..,"q50":..,"q90":..} per step or terminal
  mc_summary jsonb,                  -- samples, T, top_p, dispersion stats
  model    text NOT NULL,
  PRIMARY KEY (run_id, symbol, tf, horizon)
);
CREATE INDEX IF NOT EXISTS forecasts_lookup ON forecasts (symbol, tf, made_at);

CREATE TABLE IF NOT EXISTS signals (
  symbol   text NOT NULL,
  tf       text NOT NULL,
  made_at  timestamptz NOT NULL,
  horizon  int  NOT NULL,
  p_up     real NOT NULL,
  exp_ret  real NOT NULL,
  conf     real NOT NULL,
  stance   text NOT NULL,            -- 'BULL' | 'BEAR' | 'NEUTRAL'
  model    text NOT NULL,
  PRIMARY KEY (symbol, tf, made_at, horizon)
);
CREATE INDEX IF NOT EXISTS signals_recent ON signals (made_at DESC);

CREATE TABLE IF NOT EXISTS model_health (
  day        date NOT NULL,
  model      text NOT NULL,
  rankic     real,
  hit_rate   real,
  coverage80 real,
  notes      text,
  PRIMARY KEY (day, model)
);

-- Phase 8 adds: orders, fills, positions, pnl_daily.
EOF_SCHEMA

# ---------------------------------------------------------------- configs
cat > configs/universe_v1.yaml <<'EOF_UNIVERSE'
# Axiom universe v1 — Binance USDT spot pairs (+ matching USD-M perps for funding/OI).
# TODO (P1-01): verify liquidity, add listing dates, expand toward ~50, then FREEZE
# before Phase 2. Frozen universes only — no in-flight edits (survivorship hygiene).
version: 1
venue: binance
quote: USDT
perps: true            # also fetch USD-M futures klines + fundingRate for these bases
symbols:
  - BTCUSDT
  - ETHUSDT
  - SOLUSDT
  - BNBUSDT
  - XRPUSDT
  - DOGEUSDT
  - ADAUSDT
  - AVAXUSDT
  - LINKUSDT
  - DOTUSDT
  - LTCUSDT
  - BCHUSDT
  - NEARUSDT
  - UNIUSDT
  - ATOMUSDT
  - APTUSDT
  - ARBUSDT
  - OPUSDT
  - FILUSDT
  - INJUSDT
  - SUIUSDT
  - AAVEUSDT
  - ETCUSDT
  - TRXUSDT
  # TODO: extend to ~50 with listing dates: {symbol: PEPEUSDT, listed: "2023-05"}
EOF_UNIVERSE

cat > configs/data/crypto_v1.yaml <<'EOF_DATACFG'
# Dataset build config — consumed by axiom_data (P1-10).
universe: configs/universe_v1.yaml
source_tf: 1m
timeframes: [15m, 1h, 4h]
resample: right_closed_right_labeled     # single tested implementation, see CLAUDE.md
context_bars: 512                        # raised to 2048 at M2
horizons: [6, 12, 24]
normalization: upstream_v1               # MUST match training; spec: docs/normalization.md
embargo_bars: 512
splits:
  train: { start: "2018-01-01", end: "2023-12-31" }
  val:   { start: "2024-01-15", end: "2024-06-30" }
  test:  { start: "2024-07-15", end: "2026-06-30" }   # READ-ONLY — harness access only
EOF_DATACFG

cat > configs/finetune/crypto_v0.yaml <<'EOF_FTCFG'
# First fine-tune run (P3). Start from upstream finetune_csv defaults; after run 1,
# change exactly ONE thing per run. TODO fields are filled from vendor/kronos configs.
run_name: axiom-ft-102m-crypto1-512-v0
seed: 1337
precision: bf16
init:
  tokenizer: NeoQuasar/Kronos-Tokenizer-base
  predictor: NeoQuasar/Kronos-base
data: configs/data/crypto_v1.yaml
stage_a_tokenizer:
  enabled: true
  epochs: TODO        # copy upstream default
  lr: TODO
  batch_size: TODO
stage_b_predictor:
  enabled: true
  epochs: TODO
  lr: TODO
  batch_size: TODO
log:
  wandb_project: axiom
EOF_FTCFG

cat > configs/eval/default.yaml <<'EOF_EVALCFG'
# Frozen eval harness config (P2). Changing this file invalidates comparability —
# bump a version suffix instead of editing in place once M1 work starts.
seed: 1337
data: configs/data/crypto_v1.yaml
split: test
models: [axiom-zero-mini, axiom-zero-small, axiom-zero-base]
baselines:
  persistence: true
  ewma: true
  lightgbm: true
  chronos_bolt: false
mc:
  samples: 64
  temperature: 1.0
  top_p: 0.9
costs:                       # round-trip cost = 2 * (taker_fee + slippage)
  taker_fee_bps: 10
  slippage_bps: 7
metrics: [rankic, dir_acc_cost, mae_logret, rmse_logret, coverage_10_90, pit]
slices: [year, vol_tercile]
report_dir: reports/
EOF_EVALCFG

# ---------------------------------------------------------------- tests
cat > tests/test_smoke.py <<'EOF_TSMOKE'
"""Workspace smoke test: all packages import."""


def test_workspace_imports():
    import axiom_data  # noqa: F401
    import axiom_eval  # noqa: F401
    import axiom_model  # noqa: F401
    import axiom_signals  # noqa: F401
    import axiom_trader  # noqa: F401
EOF_TSMOKE

cat > tests/test_parity.py <<'EOF_TPARITY'
"""Parity harness placeholder (P4-02).

Will assert, for axiom_model generation:
  1) greedy decoding: token-identical before/after any optimization,
     on CPU (tiny config, CI), Modal CUDA, and — before any
     axiom-runtime-* release tag — local ROCm (RX 7900 XTX);
  2) sampled MC: return-distribution moments (mean/std/q10/q50/q90 over
     ~1k paths) within tight tolerance.
NEVER weaken tolerances to pass (CLAUDE.md).
"""

import pytest


@pytest.mark.skip(reason="P4-02: implement real parity harness")
def test_parity_placeholder():
    raise AssertionError("unreachable")
EOF_TPARITY

# ---------------------------------------------------------------- scripts
cat > scripts/download_binance.py <<'EOF_DLBIN'
"""Bulk downloader for data.binance.vision (P1-02/03).

Plan:
  - spot monthly 1m klines:   data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM}.zip
  - USD-M futures klines:     data/futures/um/monthly/klines/{SYMBOL}/1m/...
  - USD-M funding rates:      data/futures/um/monthly/fundingRate/{SYMBOL}/...
  - verify .CHECKSUM files; resume-safe; async (httpx); write raw zips to
    data/raw/, extract+convert to Parquet via axiom_data.
Fallback (P1-04): run this on a Modal function / non-EU VPS if unreachable from NL.
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/universe_v1.yaml")
    p.add_argument("--out", default="data/raw")
    p.parse_args()
    raise SystemExit("TODO P1-02: implement (see docstring + build order §3.1)")


if __name__ == "__main__":
    main()
EOF_DLBIN

# ---------------------------------------------------------------- Modal apps
cat > infra/modal_app/smoke.py <<'EOF_SMOKEPY'
"""GPU smoke test on Modal — no local GPU needed (P0-07).

Loads Kronos-small on a T4, runs a short forecast on synthetic OHLCV.
Prereq: vendor/kronos present (P0-04 subtree). Cost: pennies (fits free credits).

usage:  modal run infra/modal_app/smoke.py
"""

import modal

app = modal.App("axiom-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "pandas", "numpy", "einops",
        "huggingface_hub", "safetensors", "tqdm",
    )
    .add_local_dir("vendor/kronos", "/root/kronos")
)


@app.function(image=image, gpu="T4", timeout=900)
def forecast_smoke():
    import sys
    import time

    sys.path.insert(0, "/root/kronos")
    import numpy as np
    import pandas as pd
    import torch

    print("device:", torch.cuda.get_device_name(0))

    from model import Kronos, KronosPredictor, KronosTokenizer  # upstream API

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    n_params = sum(p.numel() for p in mdl.parameters()) / 1e6
    print(f"loaded Kronos-small: {n_params:.1f}M params")
    pred = KronosPredictor(mdl, tok, device="cuda:0", max_context=512)

    # Synthetic random-walk OHLCV — no external data needed for a smoke test.
    n, h = 400, 24
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "close": close,
            "volume": rng.uniform(1e3, 1e4, n),
        }
    )
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.abs(rng.normal(0, 0.002, n)))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.abs(rng.normal(0, 0.002, n)))
    df["amount"] = df["close"] * df["volume"]
    ts = pd.date_range("2026-01-01", periods=n + h, freq="1h")

    t0 = time.time()
    try:
        out = pred.predict(
            df=df[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=pd.Series(ts[:n]),
            y_timestamp=pd.Series(ts[n:]),
            pred_len=h,
            T=1.0,
            top_p=0.9,
            sample_count=3,
        )
        print(f"forecast OK in {time.time() - t0:.1f}s")
        print(out.head())
        print("SMOKE PASSED — GPU + weights + generation all working.")
    except TypeError as e:
        # Upstream signatures can drift between versions. If this fires, align
        # the call with vendor/kronos/examples/prediction_example.py.
        print("predict() signature mismatch — adapt to upstream example:", e)
        raise
EOF_SMOKEPY

cat > infra/modal_app/train.py <<'EOF_TRAINPY'
"""Modal training app (P3-05). Skeleton — verify decorator/arg names against
current Modal docs before first use.

usage:
  modal run infra/modal_app/train.py::train --config-yaml "$(cat configs/finetune/crypto_v0.yaml)"
"""

import modal

app = modal.App("axiom-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",  # CUDA build inside Modal — local installs stay manual (CPU/ROCm)
        "pandas", "pyarrow", "numpy", "einops",
        "huggingface_hub", "safetensors", "wandb", "pyyaml",
    )
    .add_local_dir("packages", "/root/packages")
)
data_vol = modal.Volume.from_name("axiom-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("axiom-ckpts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=12 * 60 * 60,
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(config_yaml: str):
    import sys

    sys.path.insert(0, "/root/packages/axiom_model")
    # TODO P3-05: from axiom_model.train.finetune import run
    # run(config_yaml, data_root="/data", ckpt_root="/ckpts")
    raise NotImplementedError("P3-05: wire axiom_model.train.finetune.run")
EOF_TRAINPY

cat > infra/modal_app/infer_cron.py <<'EOF_CRONPY'
"""Hourly signal cron on Modal L4 (P6-04). Skeleton — see build order §8.3.

Rules: staleness guard before inference; idempotent upserts on
(symbol, tf, made_at); alert webhook on failure; every run row in `runs`.
"""

import modal

app = modal.App("axiom-signals")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "pandas", "numpy", "ccxt", "psycopg[binary]", "pyyaml")
    .add_local_dir("packages", "/root/packages")
)
ckpt_vol = modal.Volume.from_name("axiom-ckpts", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    schedule=modal.Cron("2 * * * *"),  # hh:02, just after 1h candle close
    volumes={"/ckpts": ckpt_vol},
    secrets=[modal.Secret.from_name("postgres"), modal.Secret.from_name("telegram")],
)
def hourly_signals():
    # TODO P6-03: bars = pull_latest_bars_ccxt(universe, tf="1h", lookback=CTX)
    # TODO P6-03: guard_stale(bars, max_age_bars=2)
    # TODO P6-04: fc = predictor.predict_mc(bars, samples=64, pred_len=24)
    # TODO P6-04: upsert(signals_from_paths(fc, costs)); upsert_fan(fc); log_health()
    raise NotImplementedError("P6-04: wire signal pipeline")
EOF_CRONPY

# ---------------------------------------------------------------- API stub
cat > services/signal_api/main.py <<'EOF_APIPY'
"""Read-only signal API (P6-06). Deploy via Modal ASGI or a small VPS.
Only this API touches the DB from the outside world; dashboard + AI chat
consume it. Token auth required before exposure."""

from fastapi import FastAPI

app = FastAPI(title="axiom-signal-api")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


# TODO P6-06: /signals?tf=1h · /forecast/{symbol} · /health/model · /universe
EOF_APIPY

# ---------------------------------------------------------------- docs seeds
cat > docs/rocm-notes.md <<'EOF_ROCMD'
# Machine / backend notes

Record here: torch versions + install commands per machine (laptop CPU wheel,
XTX ROCm wheel), and every "works on one backend, breaks on another" incident
with its workaround. Rules of the road live in CLAUDE.md (SDPA-only attention,
no CUDA-only deps, keep --no-compile working, never assume a local GPU).
EOF_ROCMD

cat > docs/normalization.md <<'EOF_NORMD'
# Normalization spec (P1-09)

Document the upstream Kronos per-window normalization scheme here after reading
vendor/kronos source, then implement it ONCE in axiom_data.normalization.
Training, eval, and inference must all import that single module.
Mismatch here is the project's #1 known silent failure mode.
EOF_NORMD

# ---------------------------------------------------------------- README fallback
if [ ! -f README.md ]; then
  printf '# Axiom\n\nSee docs/AXIOM_BUILD_ORDER.md, TODO.md, CLAUDE.md.\n' > README.md
  echo "    (wrote minimal README.md — full version wasn't found next to scaffold.sh)"
fi

# ---------------------------------------------------------------- done
cat <<'EOF_NEXT'

==> Axiom scaffold complete. (Local GPU NOT required — Modal covers all GPU work.)

Next steps (also tracked as P0-xx in TODO.md):
  1. uv sync && source .venv/bin/activate
  2. install torch for THIS machine (never via pyproject):
       laptop (CPU):  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
       XTX box:       uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.4   (check current tag)
  3. git subtree add --prefix vendor/kronos https://github.com/shiyu-coder/Kronos master --squash
  4. uv run ruff check . && uv run pytest -q
  5. git add -A && git commit -m "chore(P0-03): scaffold axiom monorepo"
  6. pip install modal && modal setup
  7. GPU smoke test with NO local GPU:   modal run infra/modal_app/smoke.py    (P0-07)

Docs: docs/AXIOM_BUILD_ORDER.md · tasks: TODO.md · rules: CLAUDE.md
EOF_NEXT