# Fine-tune run 2 (3-epoch Stage B) vs run 1 and zero-shot, on val

Run `20260830T155409-val-bf81ee8` · git `bf81ee8` · split `val`, 60 anchors, 2,941
windows · W&B `nob2fcsh` · XTX/ROCm. Same anchors, per-window seeds and backend as the
v0 eval (`umuxg5sy`) and the zero-shot reference (`hzeaerq9`) — rows are comparable.
Training: `crypto_v1.yaml` (the one change: `stage_b.epochs` 20 → 3, v0's tokenizer
reused), W&B `2ydtny0x`, best val CE 2.7922 at epoch 3 (v0: 2.7930 at 2 of 20).

## v1 vs v0 vs zero-shot, 1h on val

RankIC (t) · MAE · 10–90 coverage (nominal 0.80):

| h | v1 | v0 | zero-shot |
|---|---|---|---|
| 6  | **+0.037 (1.94)** · 0.0176 · 0.60 | +0.012 (0.61) · 0.0173 · 0.59 | +0.029 (1.15) · 0.0237 · 0.45 |
| 12 | **+0.024 (1.08)** · 0.0259 · 0.53 | +0.020 (0.98) · 0.0254 · 0.53 | +0.011 (0.46) · 0.0399 · 0.33 |
| 24 | **+0.041 (1.58)** · 0.0392 · 0.47 | +0.030 (1.17) · 0.0379 · 0.48 | +0.025 (0.94) · 0.0604 · 0.32 |

Readings:

1. **v1 ≥ v0 on rank at every horizon, and ≥ zero-shot at every horizon** — the
   less-overfit predictor points the right way, consistently. But no difference here is
   *established*: all t < 2 (h6 grazes it at 1.94), and val's 60 cross-sections cannot
   separate effects of this size (P3-00d). Point-estimate ordering, noted, not celebrated.
2. **The distributional gains are unchanged from v0** (MAE and coverage within noise of
   each other, both far better than zero-shot). They evidently come from the fine-tuned
   tokenizer + early-stopped CE, not from Stage B epoch count.
3. **Economics: still negative everywhere** (net −51 to −81 bps), like the whole val
   panel including baselines.

## Standing conclusions after two runs

- 3 Stage B epochs dominate 20: equal-or-better CE and rank at 1/7th the training cost.
  `crypto_v1.yaml` is the new baseline recipe.
- The rank question cannot be settled on val. Iteration from here should target
  calibration/distribution (where val has signal) and leave the rank verdict to the
  M1-record eval on test.

Caveats: 60 cross-sections; ROCm backend; val inside Kronos's pretraining window;
tripwire is not a backtest.
