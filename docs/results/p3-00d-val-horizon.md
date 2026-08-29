# P3-00d — the horizon question, put to val

Run `20260829T172648-val-203dfe3` · git `203dfe3` · dataset
`dc6d1a9d976d5efdcd98ba57df234be5a8ab75e79700efc10771fd4a9c1747aa` · split `val`
(2024-01-15 to 2024-06-30) · **60 anchors** (the cap binds; ~73 available), 50 symbols,
2,941 windows per model, 64 MC samples · round-trip cost 34 bps · W&B run `hzeaerq9`.

Backend: RX 7900 XTX under WSL2 (torch 2.13.0+rocm7.2). Reproduce with
`axiom-eval run --config configs/eval/val.yaml --timeframes 1h
--models axiom-zero-small persistence ewma lightgbm`.

The question: P3-00b showed the 1h rank signal is real on test but inverted the horizon
ordering (12 > 6 > 24) and killed the 24-bar cell's economics. Choosing 12 because it won
on test would be selection on test, so the choice was sent to val. This run is val's
answer.

## Val's answer: it cannot rank the horizons

`axiom-zero-small` at 1h on val, against the test numbers from P3-00b:

| horizon | val RankIC | val t | test RankIC @240 | test t @240 |
|---|---|---|---|---|
| 6  | +0.0289 | +1.15 | +0.0594 | +4.53 |
| 12 | +0.0108 | +0.46 | +0.0663 | +5.02 |
| 24 | +0.0252 | +0.94 | +0.0433 | +3.45 |

**No horizon clears t = 2 on val, and the point ordering (6 > 24 > 12) disagrees with
test's (12 > 6 > 24).** That is what noise looks like at 60 cross-sections — the same
count that made Phase 2's 60-anchor test estimates unstable in the first place. Val
cannot go denser without lying: at stride 48 bars the split offers ~73 anchors, and
shrinking the stride below the 24-bar horizon would overlap the outcome windows and
inflate the t-stat. A 167-day split does not have 240 independent cross-sections to give.

So the honest output of P3-00d is not "the horizon is H". It is: **val lacks the power
to rank the 1h horizons, and no defensible single-horizon pick exists today.**

## What that decides

**The fine-tune does not wait on a horizon, because it never needed one.** Stage A/B
training is next-token over 512-bar windows — horizon-agnostic — and the eval harness
scores 6/12/24 jointly in every run. P3-01's target is therefore **1h ×
`axiom-zero-small`, all three horizons scored jointly**; no cell-level commitment is made,
which also closes the door on quietly retargeting to 12-on-test.

**M1 is judged at 1h across all three horizons** on the frozen test config: net-of-cost
RankIC above zero-shot and LightGBM per horizon, not in one hand-picked cell. A fine-tune
that only helps one horizon has to say so out loud.

## The rest of the val table, for the P3-08 before/after

This run doubles as the zero-shot reference on the iteration split — the numbers the
first fine-tune has to beat, on the split it will be compared on. Val-to-val only;
nothing here is comparable to test numbers (different period, wider error bars).

**1h, horizon 6 bars**

| model | RankIC | t | dir acc | net bps | cov 10–90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-small | +0.0289 | +1.15 | 0.506 | -60.5 | 0.45 | 0.20 | -60.0 | 0.40 |
| lightgbm | +0.0283 | +1.41 | 0.491 | -42.1 | 0.88 | 0.11 | -38.6 | 0.39 |
| persistence | -0.0071 | -0.27 | 0.474 | -13.1 | 0.79 | 0.03 | -12.8 | 0.40 |
| ewma | -0.0092 | -0.41 | 0.486 | -8.0 | 0.84 | 0.06 | -0.9 | 0.43 |

**1h, horizon 12 bars**

| model | RankIC | t | dir acc | net bps | cov 10–90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| persistence | +0.0133 | +0.50 | 0.527 | +2.5 | 0.77 | 0.03 | +4.4 | 0.46 |
| axiom-zero-small | +0.0108 | +0.46 | 0.483 | -71.6 | 0.33 | 0.26 | -67.9 | 0.40 |
| ewma | -0.0005 | -0.02 | 0.518 | +10.8 | 0.84 | 0.05 | +16.8 | 0.46 |
| lightgbm | -0.0023 | -0.13 | 0.497 | -31.1 | 0.88 | 0.11 | -19.6 | 0.44 |

**1h, horizon 24 bars**

| model | RankIC | t | dir acc | net bps | cov 10–90 | PIT KS | tripwire bps | tw win |
|---|---|---|---|---|---|---|---|---|
| axiom-zero-small | +0.0252 | +0.94 | 0.487 | -51.3 | 0.32 | 0.27 | -42.2 | 0.41 |
| persistence | -0.0159 | -0.53 | 0.497 | -29.8 | 0.68 | 0.08 | -23.0 | 0.43 |
| lightgbm | -0.0300 | -1.46 | 0.507 | -45.0 | 0.92 | 0.14 | -42.7 | 0.41 |
| ewma | -0.0497 | -1.76 | 0.490 | -21.7 | 0.78 | 0.04 | -10.8 | 0.42 |

Readings, all with the 60-cross-section caveat attached:

- **On val, zero-shot beats nothing.** The model's rank edge over the panel — decisive on
  test — is not visible here; LightGBM matches it at h6 and everything drowns after
  costs (the model worst of all: −51 to −72 bps net, tripwire losing at every horizon).
  H1-2024 was choppy and 5.5 months is short; either way, this is the bar P3-08 measures
  the fine-tune against, and it is currently on the floor.
- **A leakage footnote that cuts the other way.** Val (H1 2024) sits *inside* Kronos's
  pretraining window; test extends past it. If pretraining leakage were inflating
  results, val — fully leaked — should flatter the model *more* than test, not less.
  The weak val showing is evidence for "regime/noise", not for "test numbers are
  leakage".
- **Calibration is the one loud, stable fact: coverage 0.32–0.45 vs 0.80 nominal, PIT
  U-shaped with ~⅓ of the mass in each tail.** Third split-independent confirmation;
  still blocks Phase 6's `p_up`; still the strongest argument for M3-01.

## Caveats

- **60 cross-sections.** The absence of significance is not evidence of absence — test
  needed 240 anchors before its ordering stabilized. Val cannot supply that count without
  overlapping horizons; treat every val number as wide.
- **ROCm backend**, not Modal: MC sampling will not reproduce bitwise on an L4. Backend
  recorded in `metrics.json` under `environment`.
- **All 50 symbols contribute** (the XMRUSDT/WAVESUSDT delistings fall inside test, not
  val), so the survivor-set caveat differs from the test runs: val is the survivor set
  *plus* the two symbols that later died.
- **Only `axiom-zero-small`** was scored, plus the baseline panel. `mini`/`base` were not
  re-checked on val.
