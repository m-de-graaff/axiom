# Fine-tune run 1 vs zero-shot, on val

Run `20260830T032521-val-9c4faa1` · git `9c4faa1` · dataset `dc6d1a9d…` · split `val`,
60 anchors, 50 symbols, 2,941 windows · W&B `umuxg5sy` · XTX/ROCm. Directly comparable
row-for-row to the zero-shot reference `hzeaerq9`
(`docs/results/p3-00d-val-horizon.md`): same config, anchors, per-window seeds and
backend. Training run: W&B `6swnysvk`/`4jf1fhsy`, checkpoint
`axiom-ft-25m-crypto1-512-v0` (Stage B best = epoch 2 of 20; it overfits after).

## What the first fine-tune changed

`axiom-ft-25m-crypto1-512-v0` vs `axiom-zero-small`, 1h on val:

| h | RankIC ft (t) | RankIC zs (t) | MAE ft | MAE zs | cov ft | cov zs | PIT KS ft | PIT KS zs | net ft | net zs |
|---|---|---|---|---|---|---|---|---|---|---|
| 6  | +0.012 (0.61) | +0.029 (1.15) | **0.0173** | 0.0237 | **0.59** | 0.45 | **0.12** | 0.20 | -73.7 | -60.5 |
| 12 | +0.020 (0.98) | +0.011 (0.46) | **0.0254** | 0.0399 | **0.53** | 0.33 | **0.15** | 0.26 | -86.0 | -71.6 |
| 24 | +0.030 (1.17) | +0.025 (0.94) | **0.0379** | 0.0604 | **0.48** | 0.32 | **0.22** | 0.27 | -51.2 | -51.3 |

Three readings:

1. **The distribution got much better.** MAE fell 27–37% at every horizon and the
   miscalibrated fan moved a third of the way to nominal (10–90 coverage 0.32–0.45 →
   0.48–0.59 against 0.80; PIT tail mass at h12 dropped from ~33% per tail to ~23%).
   One two-epoch fine-tune on in-domain data did more for calibration than anything
   tried so far — evidence the miscalibration is largely a domain-shift artifact, which
   strengthens the "fixable" reading over the "architectural" one (M3-01).
2. **The rank signal is unchanged within val's power.** Mixed point moves (up at 12/24,
   down at 6), everything t ≤ 1.2 — val cannot see rank changes of this size (P3-00d).
   The rank verdict needs test, and test waits for M1.
3. **Economics still under water everywhere** (net −51 to −86 bps), as is the entire val
   panel including baselines. No tradeable claim from this run.

## What run 2 changes (one thing)

Stage B epochs. Both training attempts showed val CE bottoming at epoch 2 of 20 and
worsening monotonically after; 18 epochs are pure overfitting spend. Everything else in
`crypto_v0.yaml` stays fixed.

Caveats: 60 cross-sections (wide everything); ROCm backend; val sits inside Kronos's
pretraining window; the tripwire is not a backtest.
