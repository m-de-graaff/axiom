# Normalization (`upstream_v1`)

The scheme Axiom inherits from Kronos, written down so it can never drift. It is
implemented **once**, in `packages/axiom_data/axiom_data/normalization.py`, and
imported by training, evaluation and inference alike. Re-implementing it locally
"for convenience" is the project's #1 known failure mode (CLAUDE.md).

## The scheme

For one window of `context + horizon` bars and features
`[open, high, low, close, volume, amount]`:

```
mean, std = mean(x[:context], axis=0), std(x[:context], axis=0)   # context only
x_norm    = clip((x - mean) / (std + 1e-5), -5.0, +5.0)
x_hat     = x_norm * (std + 1e-5) + mean                          # inverse
```

- **Statistics come from the context window only.** The horizon rows are scaled with
  the context's mean and std, never their own. This is what stops the target leaking
  into its own inputs.
- **Per window, per symbol, per feature.** No global or cross-sectional statistics;
  nothing is fitted on the training split and reused at inference.
- **`eps = 1e-5`, added to `std` before dividing** (and again when inverting). It is
  part of the arithmetic, not a guard bolted on — a different epsilon changes the
  numbers.
- **`clip = 5.0`, symmetric, applied after scaling.** Denormalization does not undo
  the clip; upstream doesn't either.
- **`amount` is quote-asset volume.** For Binance feeds it comes straight from the
  `quote_volume` column. Where a feed lacks it, `ensure_amount` falls back to
  `volume * mean(open, high, low, close)`, matching `KronosPredictor.predict`.
- **Time features**, in this order: `minute, hour, weekday, day, month`, taken from
  the bar's timestamp (bar close, see below). They are *not* normalized.

## Where upstream does this

| Path | What it does |
|---|---|
| `vendor/kronos/finetune/dataset.py` (`QlibDataset.__getitem__`) | mean/std over `lookback_window` only, `/(std + 1e-5)`, `clip=5.0` from `finetune/config.py` |
| `vendor/kronos/model/kronos.py` (`KronosPredictor.predict`, `predict_batch`) | same arithmetic over the supplied context; denormalizes predictions with the stored mean/std |

`tests/test_normalization.py` asserts the formula against a literal transcription of
that arithmetic, so a future refactor of either side breaks the test rather than the
model.

## Timestamp convention (why it matters here)

`ts` is the instant a bar **closes** (`axiom_data.resample`). Time features are
therefore derived from the close label. Any live feed — ccxt in Phase 6 included —
labels bars with their *open* time and must be shifted on ingest, or the model gets
calendar features one bar out of phase with training.

## Changing this

Don't, silently. `configs/data/*.yaml` names the scheme (`normalization: upstream_v1`)
and that name is part of the dataset hash. A new scheme means a new name, a new
dataset hash, a re-run of the eval harness, and a before/after table — not an edit to
this file.
