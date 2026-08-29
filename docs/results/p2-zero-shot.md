# Phase 2 results — zero-shot Kronos vs the humiliation panel

Run `20260829T022605-default-16c3b37` · git `16c3b37` · dataset
`dc6d1a9d976d5efdcd98ba57df234be5a8ab75e79700efc10771fd4a9c1747aa` · split `test`
(2024-07-15 to 2026-06-30) · 60 anchors per timeframe, 48 symbols, 64 MC samples ·
round-trip cost 34 bps · Modal L4, assembled from 45 checkpointed chunks.

Reproduce: `modal run infra/modal_app/eval.py --chunks 4`, or the baseline legs on any
CPU with `uv run axiom-eval run --config configs/eval/default.yaml --models persistence
ewma lightgbm`.

## What the numbers say

**There is a signal at 1h, and it is the small model that has it.** `axiom-zero-small`
scores RankIC 0.068 with t = 2.56 at the 24-bar horizon, against 0.005 for LightGBM and
-0.083 for persistence and EWMA. It is the only cell in the grid with a t-stat above 2.
The 102M `axiom-zero-base` is *worse* than the 24.7M `axiom-zero-small` in almost every
cell — parameter count is not the constraint here, which is worth remembering before
anyone proposes a from-scratch 300M pretrain (M5-00).

**The naive baselines are actively harmful after costs**, which is the useful part of
having them: persistence and EWMA post negative RankIC and -60 to -100 bps of net edge
at 1h. LightGBM is roughly flat — it is not beaten by much, but it is beaten.

**The MC fan is badly miscalibrated, and this is the headline risk.** Model coverage of
the nominal 10-90 band runs 0.19 to 0.47 where it should be 0.80, with PIT KS statistics
of 0.19 to 0.44 (p ~ 0). The fan is far too narrow and its PIT histogram is U-shaped:
realized returns land outside the band roughly three times as often as they should.
LightGBM's Gaussian fan covers 0.72 to 0.94 by construction and is the only calibrated
thing in the panel.

Phase 6 turns these paths into `p_up`. On these numbers that probability would be
fiction, and the dashboard would be displaying confidence the model has not earned.
Fixing calibration — temperature, sample count, or an explicit quantile head (M3-01) —
is a prerequisite for the signal service, not a nice-to-have.

**Horizon matters more than timeframe.** Every model improves monotonically from 6 to 24
bars at 1h. 15m is noise for everyone. 4h has the best tripwire numbers but the weakest
t-stats, which is what 60 cross-sections of 4h bars buys you.

> **Superseded (2026-08-29, P3-00b).** The monotonic 6 → 24 claim does not hold at 240
> anchors: the ordering inverts to 12 > 6 > 24, and the 1h/24 economics reverse sign.
> The rank signal itself survives and strengthens. See
> [p3-00b-anchor-recheck.md](p3-00b-anchor-recheck.md) before using any number in the
> 1h tables below to choose a target cell.

## Full grid

**15m, horizon 6 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-mini | +0.0170 | +0.68 | 0.456 | -44.0 | 0.63 | 0.14 | -13.6 | 0.47 |
| axiom-zero-small | +0.0002 | +0.01 | 0.495 | -34.7 | 0.35 | 0.24 | -12.1 | 0.43 |
| axiom-zero-base | -0.0038 | -0.13 | 0.507 | -33.5 | 0.39 | 0.22 | -5.9 | 0.46 |
| lightgbm | -0.0145 | -0.87 | 0.480 | -42.5 | 0.94 | 0.19 | -13.7 | 0.41 |
| ewma | -0.0227 | -1.01 | 0.463 | -35.1 | 0.81 | 0.10 | +3.4 | 0.48 |
| persistence | -0.0262 | -0.94 | 0.475 | -27.1 | 0.80 | 0.13 | +11.5 | 0.50 |

**15m, horizon 12 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-mini | +0.0313 | +1.11 | 0.509 | -44.8 | 0.59 | 0.13 | -19.5 | 0.47 |
| axiom-zero-base | +0.0102 | +0.31 | 0.458 | -42.1 | 0.24 | 0.33 | -20.0 | 0.41 |
| axiom-zero-small | +0.0098 | +0.31 | 0.464 | -41.0 | 0.23 | 0.35 | -20.2 | 0.41 |
| lightgbm | -0.0129 | -0.65 | 0.480 | -44.7 | 0.92 | 0.15 | -16.6 | 0.42 |
| persistence | -0.0137 | -0.46 | 0.533 | -12.0 | 0.77 | 0.09 | +28.3 | 0.53 |
| ewma | -0.0342 | -1.22 | 0.494 | -17.3 | 0.77 | 0.07 | +14.9 | 0.45 |

**15m, horizon 24 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-mini | +0.0417 | +1.51 | 0.516 | -43.6 | 0.52 | 0.17 | -11.4 | 0.50 |
| axiom-zero-base | +0.0150 | +0.51 | 0.506 | -47.1 | 0.22 | 0.32 | -15.0 | 0.49 |
| axiom-zero-small | +0.0128 | +0.45 | 0.517 | -41.9 | 0.19 | 0.35 | -11.4 | 0.49 |
| lightgbm | -0.0118 | -0.62 | 0.475 | -37.1 | 0.90 | 0.15 | -5.9 | 0.45 |
| ewma | -0.0265 | -1.01 | 0.473 | -20.1 | 0.72 | 0.10 | +24.6 | 0.47 |
| persistence | -0.0274 | -1.10 | 0.485 | -29.5 | 0.69 | 0.11 | +16.7 | 0.49 |

**1h, horizon 6 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-small | +0.0264 | +0.92 | 0.535 | -5.1 | 0.47 | 0.19 | -8.1 | 0.46 |
| axiom-zero-mini | +0.0184 | +0.92 | 0.538 | -15.5 | 0.65 | 0.08 | -20.5 | 0.46 |
| axiom-zero-base | +0.0057 | +0.19 | 0.532 | -6.3 | 0.37 | 0.25 | -13.5 | 0.44 |
| lightgbm | -0.0097 | -0.39 | 0.468 | -45.7 | 0.88 | 0.10 | -54.3 | 0.39 |
| ewma | -0.0275 | -1.10 | 0.410 | -66.9 | 0.82 | 0.05 | -76.0 | 0.34 |
| persistence | -0.0447 | -1.66 | 0.449 | -56.7 | 0.82 | 0.05 | -61.5 | 0.37 |

**1h, horizon 12 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-small | +0.0559 | +1.91 | 0.560 | -4.6 | 0.40 | 0.22 | -12.6 | 0.49 |
| axiom-zero-base | +0.0349 | +1.12 | 0.546 | -11.5 | 0.23 | 0.37 | -19.9 | 0.47 |
| axiom-zero-mini | +0.0209 | +0.84 | 0.542 | -14.5 | 0.60 | 0.11 | -23.0 | 0.47 |
| lightgbm | +0.0053 | +0.25 | 0.481 | -37.3 | 0.90 | 0.11 | -47.8 | 0.41 |
| ewma | -0.0430 | -1.56 | 0.432 | -59.8 | 0.81 | 0.06 | -70.2 | 0.37 |
| persistence | -0.0562 | -1.94 | 0.451 | -55.8 | 0.79 | 0.05 | -59.0 | 0.39 |

**1h, horizon 24 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-small | +0.0681 | +2.56 | 0.567 | +22.8 | 0.37 | 0.26 | +40.5 | 0.54 |
| axiom-zero-base | +0.0515 | +1.70 | 0.557 | +12.8 | 0.22 | 0.38 | +21.8 | 0.52 |
| axiom-zero-mini | +0.0094 | +0.36 | 0.541 | +14.2 | 0.52 | 0.15 | +22.7 | 0.50 |
| lightgbm | +0.0054 | +0.27 | 0.496 | -38.0 | 0.92 | 0.13 | -10.0 | 0.47 |
| persistence | -0.0832 | -2.72 | 0.453 | -64.5 | 0.68 | 0.07 | -45.1 | 0.42 |
| ewma | -0.0835 | -2.95 | 0.410 | -100.2 | 0.72 | 0.05 | -80.4 | 0.37 |

**4h, horizon 6 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-mini | +0.0261 | +1.26 | 0.512 | -39.0 | 0.63 | 0.12 | +13.1 | 0.51 |
| axiom-zero-small | +0.0253 | +0.95 | 0.507 | -39.2 | 0.44 | 0.20 | +20.3 | 0.53 |
| axiom-zero-base | +0.0108 | +0.35 | 0.506 | -48.9 | 0.29 | 0.31 | +7.4 | 0.51 |
| lightgbm | -0.0008 | -0.04 | 0.490 | -55.2 | 0.75 | 0.11 | +14.0 | 0.51 |
| persistence | -0.0197 | -0.66 | 0.454 | -70.6 | 0.75 | 0.08 | -3.7 | 0.47 |
| ewma | -0.0449 | -1.79 | 0.461 | -52.5 | 0.79 | 0.10 | +17.7 | 0.47 |

**4h, horizon 12 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-small | +0.0206 | +0.78 | 0.524 | -36.3 | 0.36 | 0.25 | +2.5 | 0.51 |
| axiom-zero-mini | +0.0168 | +0.72 | 0.518 | -29.9 | 0.58 | 0.12 | +8.5 | 0.51 |
| axiom-zero-base | +0.0162 | +0.52 | 0.521 | -48.1 | 0.20 | 0.43 | -11.5 | 0.50 |
| lightgbm | +0.0062 | +0.29 | 0.525 | +0.4 | 0.73 | 0.08 | +53.6 | 0.53 |
| ewma | -0.0377 | -1.42 | 0.452 | -64.9 | 0.71 | 0.07 | -15.5 | 0.44 |
| persistence | -0.0426 | -1.57 | 0.438 | -131.9 | 0.64 | 0.10 | -98.3 | 0.41 |

**4h, horizon 24 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-mini | +0.0205 | +0.84 | 0.552 | +51.7 | 0.57 | 0.15 | +68.9 | 0.50 |
| persistence | +0.0097 | +0.30 | 0.453 | -205.1 | 0.62 | 0.12 | -207.1 | 0.39 |
| ewma | +0.0067 | +0.23 | 0.479 | -82.6 | 0.71 | 0.08 | -58.2 | 0.42 |
| axiom-zero-base | +0.0063 | +0.20 | 0.524 | +0.9 | 0.19 | 0.44 | +17.1 | 0.48 |
| lightgbm | -0.0001 | -0.00 | 0.497 | +10.9 | 0.79 | 0.04 | +33.3 | 0.45 |
| axiom-zero-small | -0.0098 | -0.39 | 0.536 | +6.9 | 0.37 | 0.26 | +26.9 | 0.49 |

## Caveats

- **Pretraining leakage.** Kronos was pretrained on 45 exchanges' history through ~2025.
  The test split starts 2024-07, so most of it is inside that window. These zero-shot
  numbers are optimistic and the 1h/24 result in particular needs re-checking on
  definitely-post-training months before anyone believes it.
- **Survivorship.** The universe is a survivor set (TODO B-07). XMRUSDT and WAVESUSDT
  delisted mid-test and contribute nothing; 4h cells show 47 symbols rather than 48.
- **The tripwire is not a backtest.** Equal weight, no sizing, no portfolio construction,
  per-trade Sharpe optimistic about overlapping trades. Phase 8 does this properly.
- **60 cross-sections is thin.** A t-stat of 2.56 on 60 points is suggestive, not
  established. The first thing P3-01 should do is re-run the winning cell with more
  anchors before fine-tuning toward it. **Done (P3-00b, 2026-08-29):** at 240 anchors the
  signal survives (t = 3.45) but RankIC falls to 0.043 and 1h/24 turns out to be the
  *weakest* 1h horizon, with its post-cost edge reversing sign. See
  [p3-00b-anchor-recheck.md](p3-00b-anchor-recheck.md).

## Cross-machine reproduction (P2-13)

The baseline legs were run on Modal (Linux, L4 container) and on the laptop (Windows,
CPU) from the same config and dataset hash: 73,971 rows, identical
`(model, tf, horizon, ts, symbol)` keys, maximum deviation 9e-16 on any metric column
and bit-identical PIT. Model legs are not expected to match bitwise across devices —
`torch.multinomial` does not — and are compared as distributions.
