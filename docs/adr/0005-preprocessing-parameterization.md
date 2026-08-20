# ADR-0005: Preprocessing parameterization

**Status:** Accepted (v0.0)

## Context

A bar has four prices, and they are heavily correlated. Feeding open, high, low, and close as four
independent log-returns spends model capacity relearning that high is above low.

The alternative parameterizes the candle's geometry directly: body and wicks relative to the open,
plus the gap from the previous close. That encodes the constraints in the representation instead
of asking the model to infer them.

Separately, any normalization that uses statistics from the full series leaks the future into the
past, and the leak is invisible in training metrics. It shows up only as an evaluation result too
good to be true.

## Decision

Candle geometry is the default contract: `log(h/o)`, `log(l/o)`, `log(c/o)`, and the gap
`log(o_t / c_{t-1})`.

Plain per-field log-returns is the A/B arm, kept so the geometry choice is measured rather than
assumed.

Normalization is causal only. Volume and amount get `log1p` followed by robust scaling against an
expanding median and IQR computed strictly from past bars, per asset, with a global fallback for
the cold-start window. No statistic may be computed over a window that includes the bar it
normalizes, or any bar after it.

## Consequences

v0.4 freezes this as `schema_version = 1` and ships golden test vectors. Training pre-tokenization
and the inference Predictor call the same code, so drift between them is a test failure rather
than a silent skew.

G2 includes a causality audit that fails loudly if a future-window statistic appears. It blocks
tokenizer work entirely, because a leak baked into the tokenizer contaminates every number
downstream of it.
