# AXIOM — Build Order & Engineering Plan

**From an empty folder → an improved Kronos fork ("Axiom") → signal service → dashboard → paper trading → live.**

- **Author context:** solo developer, Netherlands. Local GPU: **AMD RX 7900 XTX (24 GB, RDNA3/gfx1100)**. Cloud: **Modal.com** (NVIDIA serverless).
- **Upstream:** [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) (MIT), models on HF under `NeoQuasar` (mini 4.1M / small 24.7M / base 102.3M, all 512-bar context; large 499M is closed).
- **Prime directive:** *Model first, product second* — but **eval harness before model changes**. You cannot claim "better" or "faster" without a fixed measuring stick.

---

## 0. Operating Principles (read once, obey always)

1. **Eval-first.** No model change ships without before/after numbers from the frozen eval harness (Phase 2). "It looks better on the chart" is not evidence.
2. **Faster = provably identical.** Every speed optimization must pass a numerical-parity test against the reference implementation (same seed → same tokens, or statistically indistinguishable MC distributions).
3. **Time is the only honest axis.** All splits are chronological with an embargo gap. Never shuffle. Never tune on the test years.
4. **Everything is a config + a run ID.** Every training run, eval, and forecast batch gets an ID, a git SHA, and a config file committed to the repo. Future-you will thank present-you.
5. **Local for iteration, Modal for truth.** The 7900 XTX is your fast-feedback dev box. Canonical training runs and all scheduled production inference happen on Modal (NVIDIA), so results are reproducible and CUDA-only tooling is available.
6. **Gates, not vibes.** Each milestone below has explicit acceptance criteria. Don't start Phase N+1's *spend* (GPU-hours, paid data, live money) before Phase N's gate is green.

---

## 1. Hardware & Environment Strategy

### 1.1 The RX 7900 XTX reality check (ROCm)

The 7900 XTX is a genuinely useful ML card *if* you set it up correctly. Key facts:

| Topic | Status on RDNA3 (gfx1100) | What you do |
|---|---|---|
| PyTorch | ✅ Official ROCm wheels; the device presents as `cuda` in PyTorch, so `.cuda()`, `device="cuda"` in Kronos code works **unmodified** | Install the ROCm build of PyTorch (below) |
| OS | ✅ Linux native (Ubuntu 22.04/24.04) is the reliable path. WSL2 is officially supported by AMD for the RX 7900 series | Prefer native Ubuntu or dual-boot; WSL2 acceptable |
| `HSA_OVERRIDE_GFX_VERSION` | ❌ Not needed — gfx1100 is officially supported | Don't cargo-cult it |
| bf16 | ✅ Works | Use bf16 for train + inference |
| SDPA (`torch.nn.functional.scaled_dot_product_attention`) | ✅ Works (flash/mem-efficient backends via AOTriton on recent ROCm + PyTorch) | Make SDPA the default attention path in Axiom |
| `flash-attn` (Dao) pip wheel | ❌ CUDA-only; ROCm fork targets MI-series (CDNA), not RDNA3 | **Do not** hard-depend on `flash-attn`; import it conditionally, CUDA-only |
| xformers | ❌ effectively CUDA-only | Avoid |
| bitsandbytes | ⚠️ ROCm support is partial/fragile | Avoid; a 102M model doesn't need it |
| Triton / `torch.compile` (inductor) | ✅ Works on ROCm | Use it; expect occasional rough edges |
| 24 GB VRAM | ✅ Enormous for a 102M-param model | Full fine-tunes locally, big batches, long contexts |

**Consequence:** the XTX comfortably handles *all* of: data prep, eval harness runs, zero-shot inference, small/medium fine-tunes of Axiom (102M), and inference-optimization work. What it can't do: CUDA-only kernels, and multi-GPU scale — that's Modal's job.

**ROCm + PyTorch install (Ubuntu 24.04, native):**

```bash
# 1) AMD driver + ROCm (check https://rocm.docs.amd.com for the current installer version)
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_latest.deb   # name varies per release
sudo apt install ./amdgpu-install_latest.deb
sudo amdgpu-install --usecase=graphics,rocm
sudo usermod -aG render,video $USER   # then log out & back in
rocminfo | grep gfx                    # expect: gfx1100

# 2) Python env (uv) + ROCm PyTorch wheel
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 && source .venv/bin/activate
# Check https://pytorch.org for the current ROCm index tag (e.g. rocm6.4) — it changes.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

# 3) Sanity
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True + 'AMD Radeon RX 7900 XTX'
```

Optional quality-of-life: `export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (helps fragmentation on long runs). If you go the WSL2 route instead, follow AMD's "ROCm on WSL" guide for the 7900 series and rerun the sanity check.

### 1.2 Modal: what it's for and what it costs

Modal gives you serverless NVIDIA GPUs with cron scheduling, persistent volumes, and secrets — perfect for (a) canonical training runs, (b) the hourly production inference job, (c) an API endpoint. Starter plan includes ~$30/month of free credits; per-second billing after that. Approximate on-demand rates (verify on modal.com/pricing — they move): T4 ≈ $0.59/h, L4 ≈ $0.80/h, A10G ≈ $1.10/h, L40S ≈ $1.95/h, A100-40GB ≈ $2.10/h, A100-80GB ≈ $2.50–3.00/h, H100 ≈ $4–4.55/h. Multi-GPU via `gpu="H100:8"`.

**GPU selection policy for Axiom:**

| Job | GPU | Why |
|---|---|---|
| Hourly signal cron (50 symbols × 64 MC paths) | L4 or A10G | Cheap, plenty for 102M inference; job runs minutes |
| Fine-tune Axiom-102M (full data) | A100-80GB | Big batches, fast epochs; ~4–12 h per run |
| Context-extension continued pretraining (2048 ctx) | H100 (×1–2) | Attention cost grows; still only ~50–150 GPU-h |
| From-scratch pretrain 300–500M (Milestone M5, gated) | H100:8 | ~1.5–4k H100-hours ⇒ low-five-figures $ — only after the gate |

### 1.3 "What runs where" (pin this)

| Task | 7900 XTX (local) | Modal |
|---|---|---|
| Data download + Parquet building | ✅ (CPU-bound anyway) | volume sync only |
| Eval harness + baselines | ✅ primary | ✅ for canonical published numbers |
| Zero-shot Kronos inference | ✅ | ✅ |
| Fine-tune experiments (subset data) | ✅ overnight runs | — |
| Canonical fine-tune (full data) | — | ✅ A100 |
| Inference-speed work (KV cache, compile, batching) | ✅ develop here | ✅ verify on CUDA too (both backends must pass parity) |
| Context extension / pretraining | — | ✅ H100 |
| Hourly production signals | — | ✅ cron |
| Dashboard dev | ✅ (Node) | — |

> **Laptop / no-GPU mode:** the local GPU is optional — every "7900 XTX" cell above degrades gracefully. CPU covers all data, eval, and dev work (a 102M model even runs CPU inference for smoke tests), and Modal absorbs every GPU task: `modal run infra/modal_app/smoke.py` replaces the local smoke test (P0-07), subset fine-tunes go to A10G/L4 instead of overnight XTX runs, and parity for merges is CPU + Modal CUDA, with the ROCm leg required only before tagging `axiom-runtime-*` releases. The XTX buys iteration speed and the extra parity backend, never unblocks the critical path.

---

## 2. Phase 0 — Bootstrap the Repo (Day 1–2)

**Goal:** empty folder → running Kronos forecast on your XTX, inside a clean monorepo named `axiom`, with Modal saying hello on an NVIDIA GPU.

### 2.1 Repo layout (uv workspace monorepo)

```text
axiom/
├── pyproject.toml              # uv workspace root; ruff + pytest config
├── README.md
├── LICENSE                     # MIT (yours)
├── NOTICE                      # attribution: Kronos, Shi et al. 2025, MIT — keep forever
├── .gitignore                  # data/, .venv/, wandb/, *.ckpt ...
├── configs/                    # every experiment = one YAML here, committed
│   ├── eval/            ├── finetune/            └── pretrain/
├── packages/
│   ├── axiom_model/            # vendored + refactored Kronos core
│   │   └── axiom_model/{tokenizer.py, transformer.py, predictor.py, registry.py}
│   ├── axiom_data/             # ingestion, parquet store, dataset builder, QA
│   ├── axiom_eval/             # metrics, walk-forward, baselines, report gen
│   ├── axiom_signals/          # MC paths → bull/bear signals
│   └── axiom_trader/           # (Phase 8) risk engine + broker adapters
├── infra/
│   └── modal_app/              # train.py, infer_cron.py, api.py, images.py
├── services/
│   └── signal_api/             # FastAPI app (served via Modal ASGI or VPS)
├── apps/
│   └── dashboard/              # (Phase 7) Next.js 15 + AI SDK
├── scripts/                    # download_binance.py, build_dataset.py, ...
├── research/                   # notebooks; nothing here is load-bearing
└── data/                       # local only, gitignored: raw/ parquet/ ckpts/
```

### 2.2 Steps

```bash
mkdir axiom && cd axiom && git init
uv init --name axiom && uv venv --python 3.11
# packages as workspace members: uv init packages/axiom_model ... (repeat per package)
```

1. **Vendor Kronos.** Two options: `git subtree add --prefix vendor/kronos https://github.com/shiyu-coder/Kronos master --squash` (keeps upstream pullable), or simply copy `model/`, `finetune/`, `finetune_csv/`, `examples/` into `packages/axiom_model/` and `research/upstream/`. Either way: copy Kronos's `LICENSE` into `NOTICE` with the paper citation (Shi et al., arXiv:2508.02739). MIT makes the fork clean; renaming is allowed, attribution is required.
2. **Rename with a compatibility bridge.** Create `axiom_model` classes as thin subclasses first — `class AxiomTokenizer(KronosTokenizer)`, `class Axiom(Kronos)`, `class AxiomPredictor(KronosPredictor)` — so `Axiom.from_pretrained("NeoQuasar/Kronos-base")` **still loads upstream weights**. Add `registry.py` mapping friendly IDs → sources: `axiom-zero-base` → `NeoQuasar/Kronos-base`, later `axiom-ft-crypto-v0` → your checkpoint path/HF repo. Refactor internals gradually behind this stable API.
3. **Smoke test locally (XTX):** run the equivalent of `examples/prediction_example.py` through `AxiomPredictor` on bundled sample data. Acceptance: a forecast DataFrame + plot, no CUDA-only imports triggered. (Tip: upstream also ships a `webui/` — handy for eyeballing forecasts before your dashboard exists.)
4. **Modal hello-GPU:** `pip install modal && modal setup`, then a trivial `@app.function(gpu="T4")` that prints `torch.cuda.get_device_name()`. Create volumes now: `axiom-data`, `axiom-ckpts`. Create secrets: `wandb`, `postgres`, (later) exchange keys.
5. **CI skeleton:** GitHub Actions running `ruff check`, `pytest` (CPU-only unit tests), on every push. Add a `tests/test_parity.py` placeholder — it becomes sacred in Phase 4.
6. **Experiment tracking:** pick one and commit to it — Weights & Biases free tier (recommended) or local MLflow. Every run logs: config, git SHA, dataset hash, metrics.

**Phase 0 gate ✅:** forecast runs locally on ROCm; Modal runs a GPU function; CI is green; `NOTICE` exists.

---

## 3. Phase 1 — Data Foundation (Days 3–7)

**Goal:** a versioned, queryable crypto OHLCV corpus large enough for fine-tuning *and* honest evaluation. Crypto first (free, deep, 24/7); equities later via EODHD when/if you expand.

### 3.1 Bulk history: Binance Vision

`data.binance.vision` hosts free bulk zips: spot + USD-M futures klines (1m and up), and for futures also **fundingRate** and metrics. Plan:

- Universe: top ~50 USDT pairs by liquidity (spot) + their USD-M perps. Pin the list in `configs/universe_v1.yaml` (with listing dates — survivorship awareness).
- Download **1m klines, full history**, monthly zips + checksums, via `scripts/download_binance.py` (async httpx, resume-safe, verify CHECKSUM files).
- **Note:** Binance the *exchange* has exited EU service, but the public data site is a dataset host. If it's ever unreachable from NL, run the downloader on a small non-EU VPS or a Modal function and write straight to the `axiom-data` volume; fallbacks: Bybit/OKX public data dumps, CryptoDataDownload CSVs.

### 3.2 Storage layout

- **Parquet, partitioned:** `data/parquet/{venue}/{symbol}/{tf}/year=YYYY/month=MM.parquet`, columns `ts, open, high, low, close, volume, amount` (Kronos/Axiom expects `amount`; for crypto use quote-asset volume).
- Resample 1m → 5m/15m/1h/4h with strict right-closed, right-labeled bars (one function, unit-tested — off-by-one resampling is a classic leakage source).
- Query layer: **DuckDB** over the Parquet tree locally; sync the tree to the Modal `axiom-data` volume (`modal volume put`).
- Live/latest bars (needed from Phase 6): ccxt REST pulls into Postgres. Not needed for model work yet.

### 3.3 Dataset builder (`axiom_data.datasets`)

- Sliding windows: `(context ≤ 512 bars, horizon H)` samples with per-window normalization **exactly matching** upstream's scheme (read their preprocessing before writing yours — normalization mismatches are the #1 silent killer, see Pitfalls).
- **Chronological splits with embargo:** e.g. train ≤ 2023-12, val 2024-H1, test 2024-H2 → 2026-06, with a gap ≥ 1 max-context between splits. Test years are touched by the eval harness only.
- Data QA report (auto-generated): gaps, duplicate ts, zero-volume bars, extreme bar sanity (h ≥ max(o,c), l ≤ min(o,c)), per-symbol coverage. Fail the build on violations.

**Phase 1 gate ✅:** `axiom-data build --config configs/data/crypto_v1.yaml` reproducibly emits train/val/test with a printed dataset hash; QA report clean; corpus synced to Modal volume.

---

## 4. Phase 2 — Eval Harness *Before* Touching the Model (Days 5–10, overlaps Phase 1)

**Goal:** a frozen measuring stick. This is the highest-ROI code in the whole project.

### 4.1 Forecast-quality metrics (`axiom_eval`)

For each model × timeframe × horizon on walk-forward test windows:

- **RankIC** (Spearman between predicted and realized H-bar returns across the universe, per timestamp, then averaged) + IC t-stat.
- **Directional accuracy** vs. the *cost-aware* threshold (predicting `|r| > round-trip cost` matters more than sign).
- **MAE/RMSE on log-returns** (never on raw prices).
- **Calibration of the MC fan:** empirical coverage of the 10–90% band (target ≈ 80%), plus PIT histogram. You will turn MC paths into probabilities in Phase 6 — if the fan is miscalibrated, every "P(up)=0.73" you display is fiction.
- Slice all of the above by regime (realized-vol terciles) and by year.

### 4.2 Baselines (the humiliation panel)

Every Axiom variant is compared against: (1) persistence / random-walk, (2) EWMA drift+vol, (3) **LightGBM** on lagged return/vol/volume features, (4) optional: Chronos-Bolt zero-shot. If Axiom can't beat LightGBM net of costs, model scaling is not your bottleneck.

### 4.3 Economic sanity check

A deliberately dumb **vectorbt** long/flat threshold strategy over the signals, with taker fees (0.10–0.25%) + 5–10 bps slippage. Not the real backtest (that's nautilus_trader in Phase 8) — just an early tripwire that converts "nice RankIC" into "does it survive costs at all?"

### 4.4 Mechanics

- CLI: `axiom-eval run --model axiom-zero-base --data crypto_v1 --split test` → writes `reports/{run_id}/report.html` + metrics JSON + W&B run.
- Deterministic seeds; fixed MC `sample_count` (e.g. 64) for comparability.
- **Leakage checklist** enforced in code: no future bars in context, embargo respected, normalization stats from context window only, universe fixed ex-ante.
- ⚠️ **Pretraining-leakage caveat:** Kronos has *seen* 45 exchanges' history through ~2025 in pretraining. Zero-shot numbers on ≤2025 test data are optimistic. Weight your conclusions toward the most recent, definitely-post-training months, and say so in reports.

**Phase 2 gate ✅:** harness runs end-to-end on `axiom-zero-{mini,small,base}` + all baselines; numbers reproduce across two machines (XTX and a Modal L4) within tolerance; report auto-generated.

---

## 5. Phase 3 — Zero-Shot Baseline & First Fine-Tune (Weeks 2–3)

**Goal:** `axiom-ft-crypto-v0` — a fine-tuned 102M model that beats zero-shot Kronos *and* the humiliation panel on walk-forward crypto data. This is Milestone **M1** and the foundation everything else stands on.

### 5.1 Zero-shot pass

Run the Phase 2 harness on `axiom-zero-mini/small/base` across 15m/1h/4h and horizons {6, 12, 24 bars}. Pick the timeframe×horizon cells where the model shows *any* signal — that's where you fine-tune first. (Expect: hourly with 12–24 bar horizons is the usual sweet spot; verify, don't assume.)

### 5.2 Adapt the fine-tuning pipeline

Upstream ships two pipelines: `finetune/` (Qlib, A-share oriented) and **`finetune_csv/`** (generic CSV — your starting point for crypto). Port into `axiom_model/train/`:

1. **Stage A — tokenizer fine-tune** on your crypto corpus (adapts the quantizer's codebook to crypto's distribution: fatter tails, 24/7 sessions, stablecoin quirks).
2. **Stage B — predictor fine-tune** from `NeoQuasar/Kronos-base` init, using the Stage-A tokenizer.
3. Keep upstream hyperparameters as the first run's defaults; change one thing at a time afterward.

### 5.3 Where to run what

- **XTX (local):** subset-data runs (e.g. 10 symbols, 2 years, 1h) to debug the loop overnight. bf16, big batches — 24 GB is generous for 102M.
- **Modal (canonical):** full-corpus runs on A100-80GB, 4–12 h each, checkpoints to the `axiom-ckpts` volume, W&B logging. Budget ≈ $10–35/run; expect 5–15 runs to reach M1 (≈ $100–400 total).

### 5.4 Modal training skeleton (`infra/modal_app/train.py`)

```python
import modal

app = modal.App("axiom-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "pandas", "pyarrow", "numpy", "einops",
                 "huggingface_hub", "safetensors", "wandb", "pyyaml")
    .add_local_dir("packages", "/root/packages")          # ship your code
)
data_vol = modal.Volume.from_name("axiom-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("axiom-ckpts", create_if_missing=True)

@app.function(
    image=image, gpu="A100-80GB", timeout=12 * 60 * 60,
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(config_yaml: str):
    from axiom_model.train.finetune import run
    run(config_yaml, data_root="/data", ckpt_root="/ckpts")
    ckpt_vol.commit()

# usage: modal run infra/modal_app/train.py::train --config-yaml "$(cat configs/finetune/crypto_v0.yaml)"
```

*(Skeleton — decorator/arg names drift between Modal releases; check current docs when wiring it up. Checkpoint every N steps: Modal functions have timeouts and you want resumability anyway.)*

### 5.5 Milestone M1 gate ✅

On walk-forward test windows, `axiom-ft-crypto-v0` must show — net of the cost-aware threshold — (a) RankIC > zero-shot base **and** > LightGBM with a positive t-stat, (b) calibration coverage within ±10pp of nominal, (c) the dumb vectorbt strategy not obviously bleeding after fees. If (a) fails against LightGBM repeatedly: stop scaling ambitions and revisit features/horizons/universe — that's signal, not failure.

---

## 6. Phase 4 — FASTER (Weeks 3–4)

**Goal:** make Axiom inference fast enough that 50 symbols × 64 MC paths is trivial hourly, and iteration feels instant. Community consensus is that naive Kronos sampling is slow — this phase is where "Axiom" earns its fork.

### 6.1 Profile first (Day 1)

`axiom-bench infer --model axiom-ft-crypto-v0 --symbols 50 --samples 64 --pred-len 24` → record tokens/sec, wall time, peak VRAM on **both** the XTX and a Modal L4. These are your baseline numbers; put them in the README.

### 6.2 Optimization ladder (in order; measure after each rung)

1. **Batch the Monte Carlo dimension.** The predictor already has `predict_batch`; ensure *samples* are batched too — one forward pass over a `(symbols × samples)` batch, not `sample_count` sequential generations. Usually the single biggest win, and it's pure reshaping.
2. **KV-cache audit.** Read the generation loop. If each new bar re-encodes the full prefix (no key/value caching), implement per-layer KV caching — for autoregressive decoding at 512 context this is typically a 5–20× step-time win. Guard with the parity test.
3. **bf16 everywhere** (weights + autocast). Free on both RDNA3 and NVIDIA.
4. **SDPA attention path.** Route attention through `F.scaled_dot_product_attention` — portable (ROCm ✅, CUDA ✅), picks fused kernels where available. Keep `flash-attn` as an optional CUDA-only extra, never a hard dependency.
5. **`torch.compile`** on the decode step (`mode="reduce-overhead"`); tolerate graph breaks, keep a `--no-compile` flag for debugging (especially on ROCm).
6. **Optional / stretch:** int8 weight-only quant via `torchao` (verify it doesn't shift the MC distribution); ONNX export for the cron job; a CPU-only benchmark as a fallback story (102M is CPU-feasible, just slow).

### 6.3 Parity harness (sacred)

`tests/test_parity.py`: fixed seed + greedy decoding → token-identical outputs before/after each optimization; for sampled MC, compare distribution moments (mean/std/quantiles of predicted returns over ~1k paths) within tight tolerance. Runs in CI on CPU with a tiny config; runs on both GPUs before any perf PR merges.

### 6.4 Phase 4 gate ✅

≥ **8×** end-to-end throughput vs. the day-1 baseline on the standard bench; the full 50 × 64 × 24 batch **< 3 min on Modal L4** and **< 10 min on the XTX**; parity suite green on ROCm *and* CUDA. Tag the runtime `axiom-runtime-v1`.

---

## 7. Phase 5 — BETTER (Weeks 4–10, research track; runs in parallel with Phases 6–8)

Each sub-milestone is gated by the Phase-2 harness. Stop climbing when a rung stops paying.

### M2 — Context extension to 2048 (highest-value architectural change)

- **Audit the positional encoding first.** If upstream uses learned/absolute positions, migrate to **RoPE** (weights-compatible swap + short continued pretrain); if it's already rotary, apply NTK / positional-interpolation scaling.
- Continued pretraining at 2048 ctx from the M1 checkpoint on your crypto corpus (optionally + free multi-market data). Rough budget: ~50–150 H100-hours ≈ **$200–650 on Modal** — cheap for what it buys.
- Gate: at fixed compute, the 2048-ctx model ≥ the 512-ctx model on RankIC/calibration, with the gap widening on regime-shift slices (longer memory is the whole point).

### M3 — Trading-aligned heads

Next-token loss optimizes path realism, not tradability. Add small heads on the backbone's final hidden state: a **direction logit** (P(r_H > cost)) and **quantile regression** (q10/q50/q90 of r_H), trained frozen-then-unfrozen. Bonus: heads give you *fast* signals in one forward pass, with MC sampling retained for the fan chart. Gate: heads beat MC-derived probabilities on Brier score / calibration at equal or lower latency.

### M4 — Covariates (funding rate, open interest) — *gated on M1–M3 showing real edge*

Extend inputs beyond OHLCV(A) with perp funding + OI (already downloaded in Phase 1). Two designs: (a) extra continuous channels in a **tokenizer v2** (retrain Stage A), or (b) side-channel embeddings added to bar embeddings (no tokenizer change — try this first). Gate: improvement on futures symbols specifically, no regression on spot.

### M5 — Pretrain **Axiom-base (300–500M) from scratch** — *the big gate*

Only if: M1–M4 shipped, net-of-fee edge is real and stable in paper trading (Phase 8), **and** you've identified a limitation that only scale/data fixes. Recipe: muP for hyperparameter transfer, a proxy ladder (20M → 60M → 150M) to fit scaling curves before committing, corpus expanded to multi-market, 2048 ctx native, modern block (RMSNorm / SwiGLU / RoPE). Budget reality: ~1.5–4k H100-hours ⇒ **$6k–18k on Modal compute alone, $10k–40k all-in** with retries and data engineering. This is a milestone, not a default.

**Naming/versioning:** `axiom-{stage}-{params}-{data}-{ctx}-v{N}` (e.g. `axiom-ft-102m-crypto1-512-v0`, `axiom-cpt-102m-crypto1-2048-v1`). The registry maps names → HF repo or Modal volume path. If you ever publish weights derived from NeoQuasar checkpoints, keep the NOTICE and check each upstream HF model card's license first.

---

## 8. Phase 6 — Signal Service (Weeks 5–6; product track starts, parallel to Phase 5)

**Goal:** every hour, fresh forecasts + bull/bear signals for the whole universe land in Postgres, served by a small API.

### 8.1 Signal math (`axiom_signals`)

From S Monte Carlo paths of length H per symbol (and/or the M3 heads):

```text
r_i      = close_i[H] / last_close − 1                 (per path)
p_up     = mean( r_i > round_trip_cost )               # beat costs, not zero
exp_ret  = median(r_i);   band = (q10, q90)
conf     = f( |p_up − 0.5|, band width vs. recent realized vol )
stance   = BULL  if p_up ≥ 0.60 and exp_ret >  cost
           BEAR  if p_up ≤ 0.40 and exp_ret < −cost
           else NEUTRAL      # thresholds live in config, tuned on val only
```

Persist both the **summary** (signals row) and the **fan** (per-horizon quantiles) for charting.

### 8.2 Storage (Postgres)

Hetzner VPS with the Timescale extension (~€6/mo) or Neon free tier. Core tables: `candles(symbol,tf,ts,o,h,l,c,v)`, `forecasts(run_id,symbol,tf,made_at,horizon,quantiles jsonb,model)`, `signals(symbol,tf,made_at,horizon,p_up,exp_ret,conf,stance,model)`, `model_health(date,rankic,hit,coverage80,…)`, `runs(run_id,git_sha,config,started,status)`.

### 8.3 Modal cron (`infra/modal_app/infer_cron.py`)

```python
@app.function(image=infer_image, gpu="L4",
              schedule=modal.Cron("2 * * * *"),        # hh:02, just after candle close
              volumes={"/ckpts": ckpt_vol},
              secrets=[modal.Secret.from_name("postgres")])
def hourly_signals():
    bars = pull_latest_bars_ccxt(UNIVERSE, tf="1h", lookback=CTX)   # REST, retries
    guard_stale(bars, max_age_bars=2)          # refuse to run on stale data
    fc = predictor.predict_mc(bars, samples=64, pred_len=24)
    rows = signals_from_paths(fc, costs=COSTS)
    upsert(rows); upsert_fan(fc); log_health()
```

Idempotent upserts keyed on `(symbol, tf, made_at)`; alert to a Telegram/Discord webhook on failure or empty output. Cost: minutes per hour on an L4 ⇒ a few $/month, likely inside Modal's free credits.

### 8.4 API (`services/signal_api`, FastAPI — deployed as a Modal ASGI app or on the VPS)

`GET /signals?tf=1h`, `GET /forecast/{symbol}`, `GET /health/model`, `GET /universe`. Read-only, token-auth'd. The dashboard and the AI agent both consume this API — nothing else touches the DB directly.

**Phase 6 gate ✅:** 7 consecutive days of on-time hourly runs; the stale-data guard has fired zero false trades; `model_health` is populating.

---

## 9. Phase 7 — Dashboard + AI Wrapper (Weeks 6–8)

**Goal:** see the system. Next.js + TypeScript + Tailwind + shadcn/ui in `apps/dashboard`; deploy on Vercel Pro (~$20/mo) or self-host next to the DB.

```bash
cd apps && npx create-next-app@latest dashboard --ts --tailwind --eslint --app
cd dashboard && npx shadcn@latest init && npm i lightweight-charts ai @ai-sdk/react zod
```

- **Watchlist page:** universe table with stance badge (BULL / BEAR / NEUTRAL), `p_up`, `exp_ret`, `conf`, sparkline; sortable; refresh via polling or SSE each hour.
- **Symbol page:** TradingView **lightweight-charts v5** candlesticks; forecast overlay = median LineSeries + q10–q90 band (paired AreaSeries or a custom series) anchored at `made_at`; a toggle to replay past forecasts against realized bars — the honesty view.
- **Model-health page:** rolling RankIC, hit rate, 80%-band coverage, last-run status, current runtime tag. If this page looks bad, the trader (Phase 8) should already have halted itself.
- **Chat page (the "AI wrapper"):** **Vercel AI SDK v5** (`streamText` + `useChat`) with tool calling — tools are thin wrappers over the signal API: `get_signals`, `get_forecast(symbol)`, `explain_signal(symbol)` (the LLM narrates numbers; the numbers come from Axiom), `get_model_health`. AI SDK 6 adds agents + tool-execution approval — start on v5 stable, evaluate v6 later for an approval-gated `propose_trade` tool. Vercel's *StockBot* template is a reasonable skeleton to strip for parts.
- Auth: single-user basic auth / IP allowlist is fine for now.

**Phase 7 gate ✅:** you check the dashboard on your phone in the morning instead of running psql.

---

## 10. Phase 8 — Paper Trading & Risk Engine (Weeks 8–16)

**Goal:** the full signal → risk → order loop running for **8–12+ weeks** against simulated and testnet venues, with the exact code path you'd use live.

### 10.1 `axiom_trader` design

```text
signals (DB) ──> Strategy (thresholds, per-symbol config)
            ──> RiskEngine (may veto / resize)
            ──> Broker interface ──> SimBroker | BybitTestnet | (live adapters later)
            ──> fills/positions/PnL back to DB ──> dashboard
```

- **SimBroker first:** your own fill model — taker fee (0.10–0.25%), slippage = k·spread + participation penalty, partial fills on thin bars. Exchange testnets have unrealistically kind fills; your simulator is the pessimist.
- **Then Bybit testnet** (and/or Kraken demo futures) via ccxt for end-to-end API realism: auth, rate limits, order lifecycle, reconnects.
- **Cost-aware backtest pass:** before the soak, replay 1–2 years of signals through **nautilus_trader** with the same fee/slippage model to set expectations (vectorbt was the quick tripwire; nautilus is the grown-up check). freqtrade remains a legitimate shortcut if you want a battle-tested crypto loop and are willing to adapt its strategy interface to consume your signal DB.

### 10.2 Risk engine defaults (config, not code — tune later, never delete)

| Rule | Default |
|---|---|
| Max position per symbol | 5% of equity |
| Max gross exposure | 20% of equity |
| Position sizing | volatility-targeted, capped fractional-Kelly (¼ Kelly max) |
| Per-trade stop | derived from forecast q10 (bull) / q90 (bear) |
| Daily loss limit | −2% equity ⇒ flatten + halt until next day |
| Cooldown | 1 bar after any stop-out per symbol |
| Stale-data guard | no orders if last bar older than 2 intervals |
| Model-health circuit breaker | halt if 30-day rolling hit rate < 45% or coverage badly off |
| Global kill switch | env flag + dashboard button; checked before every order |

### 10.3 Soak-period KPIs (the go/no-go dataset for live)

Daily job writes: net PnL, Sharpe (annualized from daily), max drawdown, turnover, realized slippage vs. modeled, win rate vs. `p_up` calibration ("of trades taken at p_up≈0.65, did ~65% win net of costs?"). **Gate to Phase 9 ✅:** ≥ 8 weeks soak spanning at least one vol-regime change; paper Sharpe > ~1; drawdown within tolerance; slippage model within ~2× of realized; zero risk-rule violations caused by bugs.

---

## 11. Phase 9 — Live, Tiny, Gated (Month 4+ at the earliest)

- **Venues:** Bitvavo (AFM-licensed, iDEAL, ~0.15/0.25% base fees) or Kraken for crypto spot. If/when you add equities: IBKR paper → IBKR live. Verify the exact licensed entity in the ESMA CASP register before depositing.
- **Sizing:** an amount you can lose entirely without caring — e.g. €250–500 total, weeks at that level before any increase; scale only on evidence, never on excitement.
- **Ops:** same code path as paper with the live adapter behind a feature flag; alerting on every order, fill, halt, and error; a written runbook ("exchange down", "model degraded", "position stuck") — you will need it at 3 a.m. eventually.
- **NL admin (from the research report):** trading your own money needs no AFM license (no HFT/DEA/venue membership at retail scale). **But:** the Belastingdienst has argued bot-driven trading can tip Box 3 → Box 1; before scaling live capital, spend an hour with a belastingadviseur. If you ever open Axiom's signals or auto-trading to *other people*, stop and map MAR / MiFID II / MiCA first — auto-execution for others is portfolio management and requires authorization.

---

## 12. Milestones, Gates & Timeline at a Glance

| # | Phase | Calendar (solo, part-time honest) | Key output | Gate |
|---|---|---|---|---|
| 0 | Bootstrap | Days 1–2 | repo, ROCm working, Modal hello | forecast runs on XTX |
| 1 | Data | Days 3–7 | Parquet corpus + QA + volume sync | reproducible dataset hash |
| 2 | Eval harness | Days 5–10 | metrics, baselines, reports | zero-shot numbers reproduce on 2 machines |
| 3 | Fine-tune | Weeks 2–3 | **M1:** `axiom-ft-crypto-v0` | beats zero-shot + LightGBM, calibrated |
| 4 | Faster | Weeks 3–4 | `axiom-runtime-v1` | ≥8× throughput, parity green on ROCm+CUDA |
| 5 | Better | Weeks 4–10 (parallel) | **M2** ctx-2048 → **M3** heads → **M4** covariates → **M5** pretrain (gated) | each beats predecessor on harness |
| 6 | Signals | Weeks 5–6 | hourly cron + Postgres + API | 7 clean days |
| 7 | Dashboard | Weeks 6–8 | Next.js + fan charts + AI chat | daily-driver usable |
| 8 | Paper | Weeks 8–16 | risk engine + soak | Sharpe > 1 over ≥8 wks, calibrated, no bug-violations |
| 9 | Live | Month 4+ | tiny capital, kill switches | evidence, adviseur consulted |

**Budget through Phase 8:** ≈ $150–600 one-off GPU (fine-tunes + ctx-2048) + ~$30–60/month (VPS, Vercel, LLM API, Modal mostly inside free credits). **M5 pretraining is a separate, explicitly gated $10–40k decision.**

---

## 13. Conventions, Secrets & Hygiene

- **Branches:** trunk-based; `research/*` branches may be messy, `main` is always deployable; perf PRs require parity-suite screenshots.
- **Configs are law:** any run not reproducible from `configs/` + git SHA + dataset hash didn't happen.
- **Secrets:** local `.env` (gitignored) + Modal Secrets (`wandb`, `postgres`, `telegram`, exchange keys). Exchange API keys: **withdrawals disabled, IP-restricted**, separate keys for testnet/live, live keys nowhere near CI.
- **Model registry:** checkpoints on the `axiom-ckpts` volume; promoted models optionally mirrored to a private HF repo `you/axiom-…`; `registry.py` is the single source of truth for "which model is prod."
- **Licensing:** your code MIT; `NOTICE` credits Kronos (Shi et al., 2025, arXiv:2508.02739, MIT) forever; check upstream HF model-card licenses before redistributing derived weights.

---

## 14. Risk Register (top failure modes → mitigations)

| Risk | Mitigation baked into this plan |
|---|---|
| Normalization mismatch train↔inference silently ruins forecasts | single shared preprocessing module; parity tests; calibration monitoring |
| Look-ahead / resampling leakage | right-closed resampler unit tests; embargoed splits; leakage checklist in harness |
| Pretraining leakage inflates backtests | weight conclusions to post-2025 windows; report it explicitly |
| Overfitting via experiment shopping | frozen test years; every run logged; baselines always shown |
| Fees + slippage eat the edge | cost-aware thresholds everywhere; pessimistic SimBroker; nautilus pass |
| ROCm-specific numerical drift | dual-backend parity suite (XTX + Modal CUDA) in the merge path |
| Modal cron silently failing | idempotent upserts, staleness guard, webhook alerts, `runs` table |
| Regime shift breaks the model | per-regime eval slices; model-health circuit breaker halts trading |
| Bot-trading tax reclassification (Box 1) | consult belastingadviseur before scaling live |
| Scope creep (dashboard before model works) | phase gates; product track starts only after M1 |

---

## 15. Day-1 Checklist (copy-paste order)

```text
[ ] Ubuntu 24.04 (native or WSL2) + ROCm installed; rocminfo shows gfx1100
[ ] uv venv (3.11) + ROCm PyTorch wheel; torch.cuda.is_available() == True
[ ] mkdir axiom && git init; monorepo skeleton from §2.1; .gitignore data/
[ ] Vendor Kronos (subtree or copy); write NOTICE with MIT attribution + citation
[ ] axiom_model compat classes; Axiom.from_pretrained("NeoQuasar/Kronos-base") loads
[ ] Run first forecast locally; save the plot to research/day1/
[ ] pip install modal; modal setup; hello-GPU on T4; create volumes axiom-data, axiom-ckpts
[ ] W&B account + `wandb` Modal secret; GitHub Actions: ruff + pytest green
[ ] Start scripts/download_binance.py for the v1 universe overnight
[ ] Read upstream's normalization + generation loop code with coffee — Phases 2 & 4 depend on it
```

---

## Appendix A — Quick ROCm troubleshooting

- `torch.cuda.is_available() == False` → wrong wheel (CPU/CUDA build instead of ROCm), or user not in `render`/`video` groups (re-login), or kernel/driver mismatch — rerun `amdgpu-install`.
- Crash inside attention → force the math SDPA backend to isolate (`torch.nn.attention.sdpa_kernel([...])`), then re-enable fused backends; report kernels that fail only on ROCm and keep the `--no-compile` escape hatch.
- OOM that shouldn't happen → `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`; check nothing imported a CUDA-only extension half-way.
- Golden rule: any "works on Modal, breaks on XTX" bug goes in `docs/rocm-notes.md` — future-you forgets.

## Appendix B — Definition of Done, v1

> Axiom v1 is done when: a fine-tuned (M1) — ideally context-extended (M2) — model, served by `axiom-runtime-v1`, publishes calibrated hourly bull/bear signals for 50 crypto pairs to Postgres via Modal cron; a Next.js dashboard shows watchlist, fan charts, model health, and an AI chat over the signal API; and the paper-trading loop with the full risk engine has completed an 8-week soak whose report supports an explicit, documented go/no-go decision on tiny live capital.

*Prices, library versions, and Modal API details are 2026 snapshots — verify on the day you wire each piece. This plan is engineering guidance, not investment, legal, or tax advice.*