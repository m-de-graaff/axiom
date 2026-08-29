# AXIOM — Development TODO

Source of truth for task state. IDs are stable — reference them in commits (`feat(P1-06): right-closed resampler`).
Phases 0–4 are ticket-granular; **Phases 5–9 are deliberately coarse** and get refined when you arrive (plans made now would be stale after M1).
Full context per phase: `docs/AXIOM_BUILD_ORDER.md`. Rules: `CLAUDE.md`.

**GPU-optional track:** a CPU-only laptop blocks *nothing*. Items tagged *(GPU)* run on the RX 7900 XTX **or** on Modal — pick whichever machine you're at. XTX-specific items are marked *(XTX box, optional)*.

Legend: `[ ]` open · `[x]` done · `[~]` in progress · `[!]` blocked (note why)

---

## Phase 0 — Bootstrap (Days 1–2)

- [x] **P0-01** *(XTX box, optional — whenever you're at that machine)* Ubuntu 24.04 (native or WSL2) + ROCm installed; `rocminfo | grep gfx` shows `gfx1100`
- [x] **P0-02** `uv` venv (3.11) + torch installed **for this machine** (CPU wheel on laptop, ROCm wheel on XTX box); `import torch` OK; versions recorded in `docs/rocm-notes.md`
- [x] **P0-03** `scaffold.sh` run in empty folder; `git init`; first commit
- [x] **P0-04** Vendor Kronos: `git subtree add --prefix vendor/kronos https://github.com/shiyu-coder/Kronos master --squash`
- [x] **P0-05** Verify `NOTICE` (MIT attribution + citation) and `LICENSE` present
- [x] **P0-06** `axiom_model` compat layer: `AxiomTokenizer/Axiom/AxiomPredictor` subclasses; `registry.py` with `axiom-zero-{mini,small,base}` → `NeoQuasar/Kronos-*`
- [x] **P0-07** *(GPU)* Smoke-test forecast via `AxiomPredictor`: **`modal run infra/modal_app/smoke.py`** (T4, works from the laptop) or CPU locally (slow, fine) or the XTX; output saved to `research/day1/`
- [x] **P0-08** `modal setup`; hello-GPU function on T4 prints device name
- [x] **P0-09** Modal volumes `axiom-data`, `axiom-ckpts` created; secrets `wandb`, `postgres` created (placeholders OK)
- [x] **P0-10** GitHub repo + Actions CI green (`ruff` + `pytest`)
- [x] **P0-11** W&B project `axiom` created; one test run logged
- [x] **P0-12** `AXIOM_BUILD_ORDER.md` moved into `docs/`; linked from README
- [x] **GATE P0 ✅** forecast runs (CPU **or** Modal GPU) · Modal GPU works · CI green · NOTICE in place

## Phase 1 — Data Foundation (Days 3–7) — *all CPU/laptop-friendly*

- [x] **P1-01** `configs/universe_v1.yaml` frozen: 50 USDT pairs (all listed before 2022-01), listing month + perp flag + volume snapshot per symbol; regenerate with `scripts/build_universe.py`
- [x] **P1-02** `axiom-data download`: async spot 1m monthly zips, resume-safe, CHECKSUM-verified (`axiom_data.binance`)
- [x] **P1-03** USD-M futures klines + fundingRate download *and* ingest (`--feed futures|funding`, venue `binance-um`). OI/metrics parsing deferred to M4, which is the first thing that uses it
- [x] **P1-04** `infra/modal_app/download.py` — same downloader on Modal, writing straight to the volume (also the P1-11 path). binance.vision is reachable from NL today; this is the fallback if that changes
- [x] **P1-05** `axiom_data.store`: `{venue}/{symbol}/{tf}/year=/month=` parquet, merge-on-write so the month-boundary bar survives re-ingest
- [x] **P1-06** `axiom_data.resample`, right-closed/right-labeled, `ts` = bar close; unit tests + an opt-in network test proving 1m→1h equals Binance's own 1h klines exactly (`pytest --network`)
- [x] **P1-07** DuckDB helpers `store.read` / `store.query` over the parquet tree
- [x] **P1-08** `axiom-data qa` (gaps, dupes, OHLC sanity, zero-volume, coverage); thresholds in `configs/data/crypto_v1.yaml`, `build` refuses a dirty corpus
- [x] **P1-09** `docs/normalization.md` written from upstream source; `axiom_data.normalization` is the single implementation, asserted against the upstream formula in tests
- [x] **P1-10** `axiom-data build`: gap-free segment index (windows = offsets into segments), chronological splits, embargo enforced by full-window containment; prints the dataset hash
- [x] **P1-11** Corpus on the Modal `axiom-data` volume, built there directly by `infra/modal_app/download.py` (no multi-GB upload)
- [x] **GATE P1 ✅** hash `dc6d1a9d…` reproduces twice locally *and* on Modal (identical window counts) · QA clean on 50 symbols × 4 timeframes · corpus on the `axiom-data` volume

## Phase 2 — Eval Harness (Days 5–10, overlaps P1) — *CPU except model-dependent runs*

- [x] **P2-01** RankIC (per-timestamp cross-sectional Spearman) + t-stat
- [x] **P2-02** Directional accuracy vs **cost-aware** threshold (from `configs/eval/default.yaml`)
- [x] **P2-03** MAE/RMSE on log-returns
- [x] **P2-04** Calibration: empirical 10–90 band coverage + PIT histogram. Needed the individual
      MC paths, so the vendored generation loop gained a behaviour-preserving `reduce="none"`
- [x] **P2-05** Slicing: by year and realized-vol tercile
- [x] **P2-06** Baselines: persistence, EWMA drift+vol
- [x] **P2-07** Baseline: LightGBM on lagged return/vol/volume features. **No walk-forward
      refit**: an expanding refit inside the test period fits on test bars (CLAUDE.md rule 3) and
      would hand the baseline an advantage the models don't get. Fit once on train+val; rationale
      in `docs/eval.md`
- [ ] **P2-08** (Optional) Chronos-Bolt zero-shot adapter as extra baseline — not implemented
- [x] **P2-09** "Tripwire" long/flat threshold strategy with fees + slippage. **Without vectorbt**:
      the strategy the build order describes is 25 lines of pandas (`metrics.tripwire`); the
      dependency is worth it for nautilus_trader in P8, not for this
- [x] **P2-10** CLI `axiom-eval run --config …` → `reports/{run_id}/` (HTML + metrics JSON + panel
      parquet + both configs) + optional W&B run
- [x] **P2-11** Leakage checklist enforced as asserts (no future bars, embargo, context-only normalization, ex-ante universe)
- [x] **P2-12** Determinism: per-window seeds derived from `sha256(seed|model|symbol|tf|anchor)`,
      so results don't depend on evaluation order or sharding; `mc_samples=64`, `T=1.0`, `top_p=0.9`
- [x] **P2-13** Cross-machine reproduction: baseline legs on Modal L4 (Linux) and the laptop
      (Windows CPU) give identical keys over 73,971 rows, max deviation 9e-16, bit-identical PIT
- [x] **GATE P2 ✅** {mini, small, base} × {15m, 1h, 4h} × horizons {6, 12, 24} + all baselines
      scored on the test split · reproduces on two machines · report auto-generated. Numbers and
      the honest reading: `docs/results/p2-zero-shot.md`. Headline: `axiom-zero-small` at 1h/24
      bars is the only cell with RankIC t > 2 (0.068, t=2.56) and it beats LightGBM (0.005) and
      the naive baselines (-0.083); `axiom-zero-base` is worse than `small` almost everywhere.
      **The MC fan is badly miscalibrated** — 10–90 coverage 0.19–0.47 against a nominal 0.80 —
      which blocks Phase 6's `p_up` until it is fixed

## Phase 3 — Zero-Shot Baseline & Fine-Tune → **M1** (Weeks 2–3)

**Start here.** In order, cheapest first:

- [x] **P3-00a** *(XTX box)* ROCm parity leg + machine notes — token-identical on both
      `axiom-zero-small` and `axiom-zero-base`, `max_abs_diff 0.0`, torch 2.13.0+rocm7.2.
      Runs in **WSL2**, not Windows (no ROCm wheel for `win_amd64`); rewritten checklist and
      five new incidents in `docs/rocm-notes.md`. Unblocks `axiom-runtime-*` tags.
      Found on the way: `parity_and_speed` has no warmup run, so the first model timed
      reports a bogus speedup — parity unaffected, timings need one sig fig until fixed
- [x] **P3-00b** Re-run the winning cell with more anchors before trusting it — done on the
      **XTX** (ROCm, ~1.7h) rather than Modal; run `20260829T151050-default-d309bd8`, W&B
      `f6age13x`, writeup `docs/results/p3-00b-anchor-recheck.md`. The signal survives: 1h/24
      goes t=2.56 -> **t=3.45** on 240 cross-sections, with all three 1h horizons clearing t>3
      and the baselines still negative. **But the target cell is now in question** — RankIC at
      24 bars fell 0.068 -> 0.043, making it the weakest 1h horizon (12 > 6 > 24), and its
      post-cost tripwire reversed from +40.5 to -16.3 bps. Do NOT retarget on those numbers:
      they are the best of three *on test*. Pick the horizon on val first (see below)
- [x] **P3-00c** **Iterate on `val` from here on.** `configs/eval/val.yaml` added;
      `default.yaml` stays frozen for the M1 verdict. It is not just `split: val` —
      `lightgbm.fit_splits` drops to `[train]`, since `default.yaml` fits on `[train, val]`,
      which is correct against test and leakage against val. `load_config()` now raises when
      the eval split appears in `fit_splits` (test in `tests/test_eval.py`). Noted in the
      config: at 4h, 488 context bars eat 81 of val's 167 days, so only ~11 anchors are
      available there — 15m and 1h still hit the 60 cap, including the target cell
- [x] **P3-00d** *(GPU, ran ~1.2h on the XTX)* **Val cannot rank the 1h horizons** — run
      `20260829T172648-val-203dfe3`, W&B `hzeaerq9`, writeup
      `docs/results/p3-00d-val-horizon.md`. All three horizons land t < 1.2 at the 60
      non-overlapping anchors a 167-day split can supply, and the point ordering
      (6 > 24 > 12) disagrees with test's (12 > 6 > 24) — noise, not a verdict. Resolution:
      **no single-horizon target.** Training is horizon-agnostic (next-token over 512-bar
      windows) and the harness scores 6/12/24 jointly, so P3-01 targets 1h × `small` with
      all three horizons, and M1 is judged per-horizon at 1h — no hand-picked cell. Bonus
      on record: the zero-shot val reference for P3-08 (on val, zero-shot beats nothing —
      LightGBM matches its RankIC at h6 and every model is under water after costs), and a
      third confirmation that calibration is broken (coverage 0.32–0.45 vs 0.80)

- [x] **P3-01** *(GPU)* Zero-shot grid: {mini, small, base} × {15m, 1h, 4h} × horizons {6, 12, 24};
      target: **1h bars, starting from `axiom-zero-small`, all three horizons scored jointly** —
      `small` beats `base` almost everywhere (`docs/results/p2-zero-shot.md`); no single-horizon
      cell, because the 24-bar pick died at 240 anchors (P3-00b) and val lacks the power to
      crown a replacement (P3-00d)
- [x] **P3-02** Port `finetune_csv` → `axiom_model/train/` with config-driven entrypoint
      (`axiom-train --config configs/finetune/crypto_v0.yaml [--stage a|b]`). Losses,
      optimizer settings, OneCycle schedule and clipping norms are upstream's, unchanged;
      the CSV loader and its ratio splits are replaced by the segment index (embargoed
      chronological splits, gap-free windows) and `axiom_data.normalization` (context stats
      only). DDP dropped — every machine through M1 is a single GPU. bf16 autocast via
      `precision:`. `configs/finetune/crypto_v0.yaml` TODOs filled with upstream defaults,
      init retargeted to `axiom-zero-small` per P3-01. Guards: `splits.*: test` refused,
      window must fit `max_context`, stale manifest refused. CPU tests in
      `tests/test_train.py`; a GPU smoke run is P3-03's first step
- [ ] **P3-03** *(GPU: XTX overnight or Modal A10G/L4)* Stage A (tokenizer) subset fine-tune
- [ ] **P3-04** *(GPU: XTX overnight or Modal A10G/L4)* Stage B (predictor) subset fine-tune
- [ ] **P3-05** Modal training app (`infra/modal_app/train.py`): checkpoints to volume, resume, W&B
- [ ] **P3-06** Full-corpus Stage A on A100-80GB
- [ ] **P3-07** Full-corpus Stage B on A100-80GB → `axiom-ft-102m-crypto1-512-v0`
- [ ] **P3-08** Harness eval: comparison table vs zero-shot + all baselines, committed to `reports/`
- [ ] **P3-09** Iterate (one change per run, each logged) until M1 criteria met — budget 5–15 runs
- [ ] **GATE M1 ✅** net-of-cost RankIC > zero-shot **and** > LightGBM (positive t-stat) · coverage within ±10pp · tripwire strategy not bleeding after fees. *If LightGBM keeps winning: stop, rethink features/horizons/universe — do NOT proceed to scaling.*
      **The verdict eval runs on Modal L4** from the frozen `default.yaml` — Modal is the
      canonical backend (CLAUDE.md) and the record number should come from it. Iteration runs
      (XTX/ROCm) are decision-grade — effects that matter dwarf cross-device sampling noise —
      but they do not reproduce bitwise on an L4 and are not the number of record

## Phase 4 — FASTER → `axiom-runtime-v1` (Weeks 3–4)

- [ ] **P4-01** `axiom-bench infer` baseline (50 sym × 64 samples × 24 steps) recorded on **Modal L4** and on local hardware (XTX if available; CPU reference otherwise); numbers in README
- [x] **P4-02** Real `tests/test_parity.py`: token-identical under a fixed seed + MC moment
      tolerance; tiny-config CPU version in CI, CUDA leg via `modal run infra/modal_app/parity.py`
      (ROCm leg still owed before any `axiom-runtime-*` tag)
- [ ] **P4-03** Batch the MC dimension: one forward pass over (symbols × samples)
- [x] **P4-04** KV-cache audit: absent upstream — every step re-ran a full forward over the whole
      window. Per-layer cache added (`axiom_model/generate.py`), **9.1x** on `axiom-zero-base`
      (64 samples, 24 steps, L4: 25.8s -> 2.8s), token-identical. Pulled forward from Phase 4
      because the P2 grid was otherwise a ~$60 Modal bill. Only valid when the window does not
      slide (`context + horizon <= max_context`); upstream's loop still serves the rest
- [ ] **P4-05** bf16 weights + autocast path
- [ ] **P4-06** Attention via SDPA everywhere; `flash-attn` only as optional CUDA extra
- [ ] **P4-07** `torch.compile(mode="reduce-overhead")` on decode step; `--no-compile` flag preserved
- [ ] **P4-08** (Optional) torchao int8 weight-only; verify MC distribution unshifted
- [ ] **P4-09** Re-benchmark; before/after table committed
- [ ] **GATE P4 ✅** ≥8× vs P4-01 baseline · full batch <3 min (Modal L4; <10 min on XTX when that box is in use) · parity green on **CPU + CUDA** for merges, **plus the ROCm leg** before tagging `axiom-runtime-v1`

## Phase 5 — BETTER (Weeks 4–10, parallel research track) — *coarse; refine on arrival*

- [ ] **M2-01** Positional-encoding audit → RoPE migration plan (or NTK/PI scaling if already rotary)
- [ ] **M2-02** Continued pretrain @ 2048 ctx on Modal H100 (~50–150 GPU-h); **gate:** ≥ 512-ctx model on RankIC/calibration, gap widens on regime-shift slices
- [ ] **M3-01** Direction + quantile heads (frozen→unfrozen); **gate:** beat MC-derived probabilities on Brier/calibration at ≤ latency.
      **Candidate to pull forward:** the zero-shot MC fan is badly miscalibrated (10–90 coverage
      0.19–0.47 vs nominal 0.80), and Phase 6 cannot ship an honest `p_up` until that is fixed —
      try temperature/sample-count first, this head second
- [ ] **M4-01** Funding/OI side-channel embeddings (tokenizer v2 only if needed); **gate:** futures-slice improvement, no spot regression
- [ ] **M5-00** GO/NO-GO review for from-scratch 300–500M pretrain — requires M1–M4 shipped **+** stable paper-trading edge (P8) **+** an identified scale-only limitation. Budget $10–40k. Write the decision memo either way.

## Phase 6 — Signal Service (Weeks 5–6)

- [ ] **P6-01** Provision Postgres (Hetzner+Timescale or Neon); apply `db/schema.sql`
- [ ] **P6-02** `axiom_signals`: paths → `p_up / exp_ret / band / conf / stance`; unit tests; thresholds in config only.
      **Blocked on calibration** — see M3-01 and `docs/results/p2-zero-shot.md`
- [ ] **P6-03** ccxt latest-bars puller with retries + staleness guard (refuse > 2-bar-old data)
- [ ] **P6-04** `infer_cron.py` on Modal L4, `Cron("2 * * * *")`; idempotent upserts on `(symbol, tf, made_at)`
- [ ] **P6-05** Failure alerting (Telegram/Discord webhook) + `runs` table
- [ ] **P6-06** `signal_api` (FastAPI): `/signals`, `/forecast/{symbol}`, `/health/model`, `/universe`; token auth; deployed
- [ ] **P6-07** Daily `model_health` job (rolling RankIC, hit rate, coverage)
- [ ] **GATE P6 ✅** 7 consecutive clean hourly runs · zero stale-data outputs · health populating

## Phase 7 — Dashboard + AI Wrapper (Weeks 6–8)

- [ ] **P7-01** `create-next-app` + shadcn + `lightweight-charts` + `ai` SDK deps in `apps/dashboard`
- [ ] **P7-02** Watchlist page: stance badges, p_up / exp_ret / conf, sortable, hourly refresh
- [ ] **P7-03** Symbol page: candles + median line + q10–q90 fan anchored at `made_at`
- [ ] **P7-04** Forecast-vs-realized replay toggle (the honesty view)
- [ ] **P7-05** Model-health page
- [ ] **P7-06** Chat page: AI SDK v5 `streamText` + tools (`get_signals`, `get_forecast`, `explain_signal`, `get_model_health`)
- [ ] **P7-07** Auth (basic/IP allowlist) + deploy (Vercel Pro or self-host)
- [ ] **GATE P7 ✅** it's your phone's morning tab

## Phase 8 — Paper Trading & Risk Engine (Weeks 8–16) — *coarse; refine on arrival*

- [ ] **P8-01** Broker interface + pessimistic `SimBroker` (taker fees, k·spread slippage, partial fills)
- [ ] **P8-02** `RiskEngine` with build-order defaults (5% pos, 20% gross, ¼-Kelly cap, −2% daily halt, cooldowns, staleness guard, health circuit-breaker, kill switch) — all config
- [ ] **P8-03** nautilus_trader cost-aware replay of 1–2 years of signals; expectations memo
- [ ] **P8-04** Bybit testnet (and/or Kraken demo) adapter via ccxt; full order lifecycle handled
- [ ] **P8-05** Fills/positions/PnL persisted; dashboard panels
- [ ] **P8-06** **Soak ≥ 8 weeks** across ≥ 1 vol-regime change; weekly KPI notes in `research/soak/`
- [ ] **GATE P8 ✅** paper Sharpe > ~1 · drawdown in tolerance · slippage model within ~2× realized · p_up calibrated on taken trades · zero bug-caused risk violations

## Phase 9 — Live (Month 4+, gated) — *coarse*

- [ ] **P9-01** Belastingadviseur consult (bot-trading → Box 1 risk) — before scaling capital
- [ ] **P9-02** Venue account (Bitvavo/Kraken; verify entity in ESMA CASP register); API keys withdrawal-disabled + IP-locked
- [ ] **P9-03** Live adapter behind `AXIOM_LIVE=1`; kill switch tested end-to-end
- [ ] **P9-04** Runbook written (exchange down / model degraded / position stuck)
- [ ] **P9-05** Deploy tiny capital (€250–500); weekly review cadence; scale only on evidence
- [ ] **GATE P9 ✅** ongoing: evidence-based sizing, no rule violations, health green

---

## Backlog / stretch (unscheduled)

- [ ] **B-01** ONNX/TensorRT export for the cron job
- [ ] **B-02** CPU-inference fallback benchmark
- [ ] **B-03** Equities expansion (EODHD data, IBKR paper) — new universe + data adapters
- [ ] **B-04** AI SDK v6 evaluation: approval-gated `propose_trade` tool
- [ ] **B-05** Publish Axiom weights (check NeoQuasar model-card licenses first) + model card
- [ ] **B-06** Export this TODO to GitHub Issues (`gh issue create` script) if file-based tracking stops scaling
- [ ] **B-07** Survivorship-free universe: enumerate every symbol ever published under `data/spot/monthly/klines/` on data.binance.vision (delisted ones are still hosted) and re-select `universe_v2` from that pool. Today's candidates come from the pairs Binance lists *now*, so coins that died before today are invisible to the screen
- [x] **B-08** `wandb` was undeclared and `_log_wandb` swallowed the ImportError — a run with
      `wandb.enabled: true` could produce no W&B record and no failure. Fixed with all three:
      `wandb` declared (axiom-eval and axiom-model), `run()` preflight-imports it when the
      config asks for W&B (fails in seconds, not after the multi-hour panel build), and the
      end-of-run handler prints a loud stderr banner instead of a footnote — kept non-fatal
      because the report is already on disk by then
- [x] **B-09** `parity_and_speed` now discards one cached + one uncached warmup generate
      before timing, and all five reference rows were re-measured the same day on L4, ROCm
      and CPU (table in `docs/rocm-notes.md`). The suspect L4 `small` row (1.2x) was indeed
      the warmup artifact — it re-measures at 8.7x. Parity re-confirmed everywhere:
      `token_identical: True`, `max_abs_diff: 0.0` on every backend
- [x] **B-10** `build_panel` printed only on forecaster completion and Python block-buffers
      to a file, so P3-00b ran 97 minutes with zero progress output. Now prints a flushed
      line every 10 anchors plus the flushed completion line; confirmed live during P3-00d
