# P3-00b — the winning cell at 240 anchors

Run `20260829T151050-default-d309bd8` · git `d309bd8` · dataset
`dc6d1a9d976d5efdcd98ba57df234be5a8ab75e79700efc10771fd4a9c1747aa` · split `test`
(2024-07-15 to 2026-06-30) · **240 anchors**, 48 symbols, 10,969 windows per model,
64 MC samples · round-trip cost 34 bps · W&B run `f6age13x`.

Backend: **RX 7900 XTX under WSL2** (torch 2.13.0+rocm7.2), not Modal — see the caveat
at the bottom. Reproduce on Modal with
`modal run infra/modal_app/eval.py --timeframes 1h --max-anchors 240 --chunks 8`, or
locally with `axiom-eval run --config configs/eval/default.yaml --timeframes 1h
--max-anchors 240 --models axiom-zero-small persistence ewma lightgbm`.

The question this run existed to answer: the Phase 2 grid picked 1h × 24 bars on a
t-stat of 2.56 from 60 cross-sections, which is suggestive rather than established. Does
it survive 4x the anchors?

## The signal survives. The cell choice does not.

`axiom-zero-small` at 1h, across the three horizons, 60 anchors vs 240:

| horizon | RankIC @60 | t @60 | RankIC @240 | t @240 |
|---|---|---|---|---|
| 6  | +0.0264 | +0.92 | +0.0594 | **+4.53** |
| 12 | +0.0559 | +1.91 | +0.0663 | **+5.02** |
| 24 | +0.0681 | +2.56 | +0.0433 | **+3.45** |

**The rank signal is real.** Every 1h horizon clears t > 3 on 240 cross-sections, and the
baselines stay firmly negative (persistence and EWMA post RankIC -0.05 to -0.07 with
t = -3.3 to -5.2). The model is genuinely ordering cross-sections better than the panel.

**But 1h × 24 is the weakest of the three, not the strongest.** Its RankIC fell by a
third (0.068 → 0.043) while the other two horizons roughly doubled. The t-stat still
rose, because 240 cross-sections buy more than the shrinking effect costs — that is what
"the effect is real but the 60-anchor estimate was noisy-high" looks like.

This also kills the Phase 2 claim that "every model improves monotonically from 6 to 24
bars at 1h". At 240 anchors the ordering inverts: 12 > 6 > 24.

## The economics at 24 bars reversed sign

1h × 24 was the cell that looked *tradeable*, not just the cell with the best t-stat.
That part did not survive at all:

| horizon | net bps @60 | net bps @240 | tripwire bps @60 | tripwire bps @240 | tw win @60 | tw win @240 |
|---|---|---|---|---|---|---|
| 6  | -5.1  | -4.4 | -8.1  | **+12.3** | 0.46 | 0.52 |
| 12 | -4.6  | +2.6 | -12.6 | **+7.6**  | 0.49 | 0.51 |
| 24 | +22.8 | -5.9 | +40.5 | **-16.3** | 0.54 | 0.48 |

The +40.5 bps tripwire and 0.54 win rate at 24 bars were a 60-anchor artifact. On 240
anchors that cell loses 16.3 bps per trade and wins 48% of the time.

**Read the whole row, not the RankIC.** Even at the horizons that improved, the
cost-aware picture is thin: the best net edge in the grid is +2.6 bps at 12 bars against
a 34 bps round trip, and the best tripwire is +12.3 bps at 6 bars. A positive, clearly
significant RankIC is not the same as an edge that clears costs. Rule 4 exists for this.

## Calibration is unchanged, and still the headline risk

| horizon | coverage 10-90 | PIT KS |
|---|---|---|
| 6  | 0.47 | 0.19 |
| 12 | 0.38 | 0.23 |
| 24 | 0.33 | 0.26 |

Against a nominal 0.80. More anchors did not help, and would not be expected to — this
is bias, not variance. The PIT histograms stay U-shaped with roughly a third of the mass
in each tail bucket, so the fan is far too narrow and gets narrower as the horizon grows.
LightGBM's Gaussian fan covers 0.89–0.90 and remains the only calibrated thing here.
The Phase 6 `p_up` problem from `p2-zero-shot.md` is untouched by this run.

## Full grid, 240 anchors


**1h, horizon 6 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-small | +0.0594 | +4.53 | 0.562 | -4.4 | 0.47 | 0.19 | +12.3 | 0.52 |
| lightgbm | +0.0274 | +2.57 | 0.487 | -38.1 | 0.89 | 0.13 | -19.2 | 0.46 |
| persistence | -0.0520 | -3.69 | 0.402 | -61.6 | 0.77 | 0.07 | -44.2 | 0.36 |
| ewma | -0.0533 | -4.15 | 0.385 | -69.9 | 0.82 | 0.07 | -55.7 | 0.35 |

**1h, horizon 12 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-small | +0.0663 | +5.02 | 0.564 | +2.6 | 0.38 | 0.23 | +7.6 | 0.51 |
| lightgbm | +0.0180 | +1.69 | 0.498 | -36.5 | 0.90 | 0.12 | -25.6 | 0.46 |
| persistence | -0.0676 | -4.58 | 0.429 | -67.4 | 0.71 | 0.06 | -60.0 | 0.39 |
| ewma | -0.0684 | -5.18 | 0.405 | -79.9 | 0.79 | 0.04 | -75.6 | 0.37 |

**1h, horizon 24 bars**

| model | RankIC | t | dir acc | net bps | cov 10-90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-small | +0.0433 | +3.45 | 0.528 | -5.9 | 0.33 | 0.26 | -16.3 | 0.48 |
| lightgbm | -0.0066 | -0.68 | 0.496 | -31.8 | 0.90 | 0.12 | -38.6 | 0.45 |
| ewma | -0.0477 | -3.30 | 0.458 | -67.2 | 0.74 | 0.04 | -78.9 | 0.40 |
| persistence | -0.0504 | -3.30 | 0.476 | -50.5 | 0.63 | 0.09 | -62.6 | 0.42 |

## What this licenses, and what it does not

**It licenses fine-tuning at 1h.** The question P3-00b asked was whether the signal is
an artifact of 60 cross-sections. It is not.

**It does not license retargeting Phase 3 at 1h × 12 on the strength of this table.**
That number is the best of three horizons *on the test split*, and test has now been
looked at twice. Picking the target cell because it won on test is selection on test —
the same mistake as threshold-picking, one level up, and it would quietly inflate
whatever M1 eventually reports. Rule 3 does not have an exception for "but the t-stat
was compelling".

The clean move is to re-run these three horizons on `configs/eval/val.yaml` (P3-00c) and
pick the target there. If val agrees that 12 beats 24, retarget with a clear conscience.
If it disagrees, that is worth knowing before spending a fine-tune on either. The
horizon choice is cheap to settle on val and expensive to get wrong.

What is *already* safe to carry forward, because it does not depend on which horizon
wins: `axiom-zero-small` at 1h beats the panel on rank ordering, the naive baselines are
actively harmful after costs, and the MC fan needs fixing before any of this becomes a
probability.

## Caveats

- **Run on ROCm, not Modal.** MC sampling is not bit-identical across devices, so this
  will not reproduce exactly on an L4 — `docs/rocm-notes.md` covers why. The effect sizes
  here are far larger than sampling noise, so the conclusion holds, but if a canonical
  Modal number is wanted for the M1 record, this run does not replace it. The backend is
  recorded in `metrics.json` under `environment`.
- **Pretraining leakage, unchanged from Phase 2.** Kronos saw exchange history through
  ~2025 and the test split starts 2024-07, so most of it sits inside that window. More
  anchors does nothing about this. It remains the biggest reason to distrust the level of
  these numbers, as opposed to the ordering.
- **Survivorship, unchanged.** 48 of 50 symbols; XMRUSDT and WAVESUSDT delisted mid-test
  and contribute nothing.
- **The tripwire is still not a backtest.** Equal weight, no sizing, overlapping trades.
  Phase 8 does this properly.
- **Only `axiom-zero-small` was scored.** `mini` and `base` were skipped to keep the run
  to ~1.7h on one GPU. Phase 2 found `base` worse than `small` nearly everywhere; that
  comparison has not been re-checked at 240 anchors.
