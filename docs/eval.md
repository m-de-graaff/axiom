# The eval harness (Phase 2)

`axiom-eval` is the frozen measuring stick. Nothing about the model or the
preprocessing changes without before/after numbers from here (CLAUDE.md rule 1).
This document says what it measures, what it deliberately does *not* do, and where
it departs from `docs/AXIOM_BUILD_ORDER.md` §4.

```bash
uv run axiom-eval run --config configs/eval/default.yaml
uv run axiom-eval run --config configs/eval/default.yaml \
    --models persistence ewma --timeframes 1h --max-anchors 4      # laptop smoke
modal run infra/modal_app/eval.py                                  # the second machine
```

Output lands in `reports/{run_id}/`: `report.html`, `metrics.json`, `panel.parquet`,
and copies of both configs. `run_id` is `{utc}-{config}-{git sha}`.

## The panel

Everything is computed from one long table, one row per scored
(model, timeframe, horizon, anchor, symbol):

| column | meaning |
|---|---|
| `pred` | median MC forecast, cumulative **log return** over the horizon |
| `realized` | realized cumulative log return over the same horizon |
| `q_lo`, `q_hi` | 10th/90th percentile of the MC fan |
| `pit` | probability-integral transform of `realized` under the fan |
| `ctx_vol` | realized vol of the context window — ex ante, used for slicing |

Forecasters — models and baselines alike — implement one method:

```python
forecast(windows, horizons, samples, seed) -> (n_windows, samples, n_horizons)
```

cumulative log returns from the last context close. Returns rather than prices
because prices are not comparable across symbols; distributions rather than points
because calibration is half of what this harness exists to measure.

### Anchors

An *anchor* is the last bar of a context window. Anchors sit on a shared,
epoch-aligned grid (`panel.stride_bars`), so every symbol is scored at the same
timestamps — without that, a cross-sectional RankIC is meaningless. Two machines
derive the same anchors from the config alone. `panel.max_anchors` thins them
evenly across the split, so a smoke run still spans the whole test period rather
than its first week.

Horizons are nested: one MC run at `max(horizons)` steps yields 6, 12 and 24-bar
forecasts, so the horizon grid is nearly free.

## Leakage checklist (P2-11)

Enforced as asserts in `axiom_eval/panel.py::_check`, which runs on every window:

1. context and horizon lie entirely inside the embargoed split bounds from
   `axiom_data.datasets.split_bounds`;
2. no window spans a data gap — segments come from `axiom_data.datasets.segments`,
   the same function the dataset builder uses;
3. forecasters receive `Window.context` only; realized bars live in
   `Window.future_close`, which nothing on the forecasting path reads;
4. normalization statistics come from the context window only — every forecaster
   goes through `axiom_data.normalization`, the single implementation;
5. the universe is fixed ex ante by `configs/universe_v1.yaml`.

Never soften these to make a run finish.

## Metrics

* **RankIC** (P2-01) — per-timestamp cross-sectional Spearman, then averaged, with
  the IID t-stat over timestamps. Keep `stride_bars` ≥ the horizon: overlapping
  anchors inflate the t-stat.
* **Directional accuracy** (P2-02) — computed *only* on forecasts whose magnitude
  clears the round-trip cost, `2 × (taker_fee + slippage)` from the config. A hit
  that doesn't clear costs is not a hit (CLAUDE.md rule 4). `net_edge_bps` is the
  average signed return of those forecasts after paying the round trip.
* **MAE / RMSE** (P2-03) — on log returns, never prices.
* **Calibration** (P2-04) — empirical coverage of the 10–90 band (nominal 0.8) plus
  a PIT histogram and its KS statistic. If the fan is the wrong shape, every
  `P(up) = 0.73` shipped in Phase 6 is fiction.
* **Slices** (P2-05) — by year and by realized-vol tercile. Tercile edges are cut
  post hoc over the evaluated panel; `ctx_vol` itself is ex ante.

## Baselines (P2-06, P2-07)

* **persistence** — the drift of the last `max_horizon` bars, carried forward. A
  pure random walk (drift 0) is constant across the cross-section and would make
  RankIC undefined, so "persistence" here means *the return persists*, which is the
  standard framing.
* **ewma** — EWMA drift and EWMA vol over the context, Gaussian increments.
* **lightgbm** — gradient boosting on lagged returns, rolling vol, cumulative
  returns and volume z-scores; one model per horizon.

## Deviations from the build order

* **No walk-forward refit for LightGBM.** §4.2 and TODO P2-07 ask for one. An
  expanding refit inside the test period fits on test bars, which CLAUDE.md rule 3
  forbids outright, and it would also give the baseline an advantage the Axiom
  models (trained once, on train) do not have. LightGBM is fit once on
  `lightgbm.fit_splits` (train + val). If the rule is ever relaxed for baselines,
  the refit goes here and the config gets a cadence key.
* **No vectorbt.** §4.3 names it; the tripwire it describes — enter when the
  forecast clears costs, hold the horizon, pay the round trip — is 25 lines of
  pandas (`metrics.tripwire`), and vectorbt would add numba to a CPU-only laptop
  install for nothing. The grown-up, cost-aware replay is nautilus_trader in
  Phase 8, and that is the one worth a dependency.
* **Chronos-Bolt (P2-08) not implemented.** Marked optional in the build order.
* **`reduce="none"` added to the vendored generation loop.** Upstream averages the
  MC samples inside `auto_regressive_inference`; calibration needs the individual
  paths. The default is unchanged, so `predict()` behaves exactly as upstream.

## Determinism (P2-12)

Every forecaster is seeded per window from
`sha256(seed | model | symbol | timeframe | anchor)`, so results do not depend on
evaluation order, on which symbols happened to be present, or on how the run was
sharded. MC settings (`samples=64`, `T=1.0`, `top_p=0.9`) live in the config and
are frozen.

Baselines are pure NumPy and reproduce bit-for-bit across machines. Model metrics
will not: `torch.multinomial` is not bit-identical across CPU/CUDA/ROCm. Compare
them within tolerance, not exactly.

## Caveats printed on every report

* **Pretraining leakage** — Kronos saw 45 exchanges' history through ~2025.
  Zero-shot `axiom-zero-*` numbers on ≤2025 test data are optimistic; weight
  conclusions toward the most recent months.
* **Survivorship** — the universe is a survivor set (see TODO B-07). XMRUSDT and
  WAVESUSDT were delisted during the test window and contribute no test windows.
* **The tripwire is not a backtest** — equal weight, no sizing, no portfolio
  construction, per-trade Sharpe optimistic about overlap.
