# Axiom

Guidance for Claude Code (and any AI assistant) working in this repository. Read this fully before changing anything. The authoritative plan is `docs/AXIOM_BUILD_ORDER.md`; the live task list is `TODO.md`. When this file and ad-hoc instructions conflict, ask.

## What this project is

Axiom is a fork/extension of **Kronos** (Shi et al. 2025, arXiv:2508.02739, MIT) — a decoder-only foundation model for financial K-lines — plus a signal service (Monte-Carlo forecasts → bull/bear probabilities), a Next.js dashboard, and a paper-first trading loop. Solo developer, Netherlands. The project optimizes for three things, in order: **honest evaluation, reproducibility, not losing money to bugs.** Speed of shipping comes fourth.

## Golden rules (non-negotiable)

1. **Eval-first.** No model or preprocessing change merges without before/after numbers from the frozen eval harness (`axiom-eval`). Include the W&B run ID in the PR/commit message. "The chart looks better" is not evidence.
2. **Parity is sacred.** Any change to `axiom_model` internals (generation loop, attention, caching, quantization, compile flags) requires `tests/test_parity.py` green on the tiny-config **CPU** variant (runs in CI) **and on Modal CUDA** before merging. The **ROCm leg** (RX 7900 XTX) is additionally required before tagging any `axiom-runtime-*` release. Greedy decoding must be token-identical; sampled MC distributions must match within the configured tolerance. Never weaken tolerances to make a test pass.
3. **Test years are read-only.** Splits live in `configs/data/*.yaml`. Never fit, tune, threshold-pick, or "just peek" on the test split. The embargo gap is enforced in the dataset builder — do not bypass it. If you need more validation data, say so; do not touch test.
4. **Costs everywhere.** Any accuracy, signal, or PnL claim must be net of the fee+slippage thresholds in `configs/eval/*.yaml`. A directional hit that doesn't clear round-trip costs is not a hit.
5. **Configs are law.** Every run must be reproducible from a committed YAML in `configs/` + git SHA + dataset hash. No magic numbers in code; thresholds, universes, horizons, and risk limits are config. If a value isn't in config, put it there before using it.

## Hardware & backend constraints

- **A local GPU is OPTIONAL.** Modal-only development is fully supported: CPU handles all data/eval/harness/dev work, a 102M model runs CPU inference for smoke tests, and every GPU task has a Modal path in `infra/modal_app/` (GPU smoke with zero local hardware: `modal run infra/modal_app/smoke.py`). Never make local-GPU presence an assumption in code — detect capabilities and degrade gracefully.
- **Portability rules apply on every machine (also CPU CI):**
  - **NEVER** add hard dependencies on `flash-attn`, `xformers`, `bitsandbytes`, or any CUDA-only wheel. If a CUDA-only optimization is valuable, gate it behind a runtime capability check and an optional extra.
  - All attention goes through `torch.nn.functional.scaled_dot_product_attention` (portable CPU/ROCm/CUDA).
  - `torch` is intentionally **not** in any `pyproject.toml` — it is installed per machine (CPU wheel on the laptop, ROCm wheel on the XTX box, CUDA inside the Modal image). Do not "fix" this by adding torch to dependencies.
- **When the AMD box (RX 7900 XTX, ROCm) is in the loop:** PyTorch presents it as `cuda`, so `device="cuda"` is correct everywhere. Keep the `--no-compile` escape hatch working; `torch.compile` issues on ROCm get logged in `docs/rocm-notes.md`, not worked around silently. The XTX exists for iteration speed and the ROCm parity leg — it is an accelerator, not a dependency.
- **Modal (NVIDIA) is canonical.** Full-data training runs and all scheduled/production inference run on Modal. GPU policy: T4 for smoke tests, L4 for the hourly signal cron, A10G/L4 for subset fine-tunes when no local GPU is available, A100-80GB for full fine-tunes, H100 only for context-extension/pretraining milestones. Don't request bigger GPUs than the policy without a reason in the PR.

## Commands

```bash
# environment
uv sync                                  # installs workspace packages + dev group
source .venv/bin/activate

# torch — per machine, never via pyproject:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu       # CPU-only laptop
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.4   # AMD XTX box (check pytorch.org for current tag)

# quality gates (must be green before any commit)
uv run ruff check .
uv run pytest -q

# GPU smoke test with NO local GPU (after vendor/kronos subtree exists):
modal run infra/modal_app/smoke.py

# evaluation & benchmarks
uv run axiom-eval run --config configs/eval/default.yaml            # -> reports/{run_id}/
uv run axiom-eval run --config configs/eval/default.yaml     --models persistence ewma --timeframes 1h --max-anchors 4      # laptop smoke, no GPU
modal run infra/modal_app/eval.py                                  # same run on an L4 (P2-13)
uv run axiom-bench infer --model axiom-ft-crypto-v0 --symbols 50    # once implemented (P4)

# Modal
modal run infra/modal_app/train.py::train --config-yaml "$(cat configs/finetune/crypto_v0.yaml)"
modal deploy infra/modal_app/infer_cron.py
modal volume put axiom-data data/parquet /parquet

# data (P1)
uv run axiom-data download --config configs/universe_v1.yaml   # binance.vision zips
uv run axiom-data ingest   --config configs/data/crypto_v1.yaml # -> parquet + resample
uv run axiom-data qa       --config configs/data/crypto_v1.yaml
uv run axiom-data build    --config configs/data/crypto_v1.yaml # prints dataset hash
modal run infra/modal_app/download.py                           # same, straight to the volume
```

## Repo map

```
configs/            experiment YAMLs — the single source of truth for parameters
packages/
  axiom_model/      vendored+refactored Kronos core: tokenizer, transformer, predictor, registry, train/
  axiom_data/       download, parquet store, resampling, QA, dataset builder, normalization
  axiom_eval/       metrics (RankIC, calibration…), baselines, walk-forward, reports
  axiom_signals/    MC paths → p_up / exp_ret / quantile band / stance
  axiom_trader/     (Phase 8) risk engine + broker adapters (Sim → testnet → live)
infra/modal_app/    Modal smoke / train / hourly-inference / API apps
services/signal_api FastAPI read-only API over Postgres
apps/dashboard/     Next.js 15 + lightweight-charts + Vercel AI SDK chat
db/schema.sql       Postgres DDL
vendor/kronos/      upstream subtree (read-only reference — never edit in place)
docs/               AXIOM_BUILD_ORDER.md, normalization.md, rocm-notes.md, decisions
tests/              parity + unit tests; CI runs these on CPU with a tiny config
data/               local only, gitignored
```

## Conventions

- **Model naming:** `axiom-{stage}-{params}-{data}-{ctx}-v{N}` (e.g. `axiom-ft-102m-crypto1-512-v0`). `packages/axiom_model/axiom_model/registry.py` is the only place that maps names → weights (HF repo or Modal volume path). Code never hardcodes checkpoint paths.
- **Weight compatibility:** `Axiom*` classes must keep loading upstream `NeoQuasar/Kronos-*` checkpoints (`axiom-zero-*` registry entries). Refactors that break this need a migration note.
- **One change per training run.** Each run = one YAML in `configs/finetune/` + one W&B run logging: config, git SHA, dataset hash, metrics. Runs not reproducible this way didn't happen.
- **Commits reference TODO IDs:** `feat(P4-04): implement per-layer KV cache`. Trunk-based; `main` always deployable; `research/*` branches may be messy.
- **Normalization lives in exactly one module** (`axiom_data.normalization`), documented in `docs/normalization.md`, used identically by training, eval, and inference. Never re-implement it locally "for convenience" — normalization drift is the project's #1 known failure mode.
- **Resampling is right-closed, right-labeled**, implemented once in `axiom_data`, covered by unit tests. Off-by-one bars = lookahead leakage.

## Secrets & safety

- Local secrets in `.env` (gitignored, template in `.env.example`); cloud secrets as Modal Secrets: `wandb`, `postgres`, `telegram`, `exchange-*`. Never print, log, or commit secret values; never echo them in CI.
- Exchange API keys: **withdrawals disabled, IP-restricted**, separate keys for testnet vs live. Live keys never appear in CI, notebooks, or Modal images — only Modal Secrets.
- **Live trading is flag-gated:** no code path may place a real-money order unless `AXIOM_LIVE=1` *and* the risk engine approves. Until Phase 9 is formally opened in `TODO.md`, only SimBroker and testnet adapters may be wired to strategies.
- Risk-engine defaults (position caps, daily loss limit, circuit breakers) live in config; changing them requires an explicit human-approved commit — never adjust them as a side effect of another change.

## Things you must never do

- Touch, tune on, or report cherry-picked results from the test years.
- Weaken/skip parity tests, QA checks, or the staleness guard to make CI or a demo pass.
- Add CUDA-only hard dependencies, or add `torch` to `pyproject.toml`.
- Assume a local GPU exists; code paths must run (however slowly) on CPU or route to Modal.
- Commit anything in `data/`, checkpoints, or `.env`; commit generated reports outside `reports/`.
- Edit `vendor/kronos/` in place (port code into `packages/axiom_model` instead), or remove the Kronos attribution in `NOTICE`.
- Call live exchange endpoints (non-testnet) from any code, test, or notebook before Phase 9.
- Invent thresholds, fees, universes, or hyperparameters inline — config first.
- Present forecast output as financial advice in UI copy; the dashboard describes model output, nothing more.

## Current status

- **Phase: 2 — Eval harness: built, gate not yet met.** `axiom-eval run` scores models and the
  baseline panel (persistence, EWMA, LightGBM) on a shared, epoch-aligned anchor grid and writes
  `reports/{run_id}/` (HTML + metrics JSON + panel parquet + both configs), optionally to W&B.
  Metrics: RankIC + t-stat, cost-aware directional accuracy, MAE/RMSE on log returns, 10–90
  coverage + PIT, sliced by year and vol tercile, plus the long/flat cost tripwire. Leakage rules
  are asserts. Spec and the three deliberate deviations from the build order (no vectorbt, no
  LightGBM walk-forward refit, Chronos-Bolt skipped) are in `docs/eval.md`. Smoke-tested end to end
  on the real corpus at 1h. **Still open for the gate:** the full zero-shot scoring run over
  {mini, small, base} × {15m, 1h, 4h} (GPU) and the Modal L4 cross-machine leg (P2-13,
  `infra/modal_app/eval.py`).
- **Phase: 1 — Data foundation: complete.** 50-symbol Binance USDT universe frozen in
  `configs/universe_v1.yaml`, selected on train-period median daily volume (not a live
  snapshot) with a continuity screen; 1m spot history downloaded and CHECKSUM-verified,
  ingested to partitioned Parquet and resampled to 15m/1h/4h; QA clean; `axiom-data build`
  emits dataset hash `dc6d1a9d976d5efdcd98ba57df234be5a8ab75e79700efc10771fd4a9c1747aa`
  reproducibly on the laptop (twice) **and** on Modal from the volume copy of the corpus,
  with identical window counts; corpus on the `axiom-data` volume. Phase 1 gate met. (Update this line as gates close; details in `TODO.md`.)
- Phase 0 complete. P0-01 (ROCm/XTX box) was set up previously for another project; its
  torch ROCm wheel version still needs recording in `docs/rocm-notes.md` from that machine.
- Known limitation carried into Phase 2: the universe is a survivor set (candidates are
  pairs Binance lists today). Delisted-mid-history symbols are absent; `XMRUSDT` and
  `WAVESUSDT` were delisted during the test window and contribute no test windows.
  Reports must say so.
- Next: close the **P2 gate** (run the harness on Modal for the zero-shot grid + the
  cross-machine comparison), then **Phase 3 — zero-shot baseline & first fine-tune**.

## When unsure

Prefer asking over inventing. Cite the relevant section of `docs/AXIOM_BUILD_ORDER.md` when proposing deviations. Small honest steps beat large clever ones — this codebase will eventually route real money, and it is built like it.