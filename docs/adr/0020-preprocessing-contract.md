# ADR-0020: The preprocessing contract — six causal features, frozen at `schema_version = 1`

**Status:** Accepted (v0.4). **Frozen** — changes require a `schema_version` bump, refitted
constants, and re-cut golden vectors and regression snapshots.

Supersedes nothing; implements ADR-0005, which fixed the parameterization but left every number
open.

## Context

Two pieces of code will turn bars into model inputs: the v0.6 pre-tokenization job, which does it
once over forty million bars, and the v0.9 Predictor, which does it every time somebody asks for a
forecast. If they disagree by so much as a rounding rule, the model is evaluated on a distribution
it was never trained on, and nothing in the evaluation says so.

Kronos's own preprocessing cannot be copied, because it leaks. `KronosPredictor` z-scores each
feature against the mean and standard deviation of the window it is normalizing — so the first bar
of a context knows the last bar's volatility. In training that is merely optimistic; in a
walk-forward evaluation it is the difference between a real number and a fake one, and it is
invisible in the training curves.

So the contract has to answer three questions at once: which six features, how they are scaled
without seeing the future, and how anybody proves the second claim.

## Decision

### The contract is a pure function of the bars it is handed and frozen constants

Nothing else. Not the corpus, not the file the bars came from, not a statistic computed at call
time. Every other decision below is this rule applied somewhere specific, and the rule is what
makes "causal" testable rather than aspirational.

The public surface is exactly four functions, exported from `axiom.contract`:

```
load_spec  load_constants  transform  inverse
```

v0.6 and v0.9 import these. There is no second path, and a test asserts the package exports
nothing else.

### The anchor-bar rule

Feature rows exist for bars `t >= 1` of each segment. **Bar 0 produces no row**; it is consumed as
the anchor. Its close seeds the gap feature and the Predictor's price inversion, and its volume and
amount seed the first strictly-past median.

Consequence, and it is not cosmetic: usable windows are `Σ max(0, (n_bars - 1) - 511)`, not
`Σ max(0, n_bars - 511)`. Every segment in the corpus yields one window fewer than the v0.3 report
published. The corrected table is in `docs/reports/v0.4-contract-qa.md`, which supersedes it.

### `geo-v1`, the primary: candle geometry

| feature | definition |
|---|---|
| `gap` | `log(open_t / close_{t-1})` |
| `body` | `log(close_t / open_t)` |
| `upper` | `log(high_t / open_t)` |
| `lower` | `log(low_t / open_t)` |
| `volume` | flow feature, below |
| `amount` | flow feature, below |

Two structural identities hold for every bar, and they are the reason this is the primary
parameterization rather than four independent returns:

```
upper >= max(0, body)        because high >= max(open, close)
lower <= min(0, body)        because low  <= min(open, close)
```

The representation encodes them. Four independent log-returns spend model capacity relearning that
high is above low.

### `ret-v1`, the challenger: per-field log-returns

`ret_open, ret_high, ret_low, ret_close = log(x_t / close_{t-1})`, plus the same two flow features.
Invariants `ret_high >= max(ret_open, ret_close)` and `ret_low <= min(ret_open, ret_close)`.

Identical to `geo-v1` in every respect a consumer can see — same window, same clip, same emitted
dtype, same six columns — so v0.5's reconstruction A/B measures the parameterization and nothing
else. v0.5 decides the survivor.

### Price scaling: frozen affine constants, per (asset class × frequency × feature)

`center` is a robust median and `scale` an IQR/1.349, fitted once over **pre-firewall bars only**
(ADR-0021) and committed to `src/axiom/configs/contract_constants_v1.yaml`. No runtime fitting.

The honest cost: a class's constants are a compromise across every instrument in it, and the
volatility spread inside "crypto" is wide. Three things absorb that — the clip, the quantizer range
v0.5 chooses, and the conditioning embeddings v0.7 adds. **Per-series adaptive scaling is a named
post-1.0 experiment**, not an oversight. It is rejected here because a per-series statistic is a
per-series fit, and a fit is the thing this contract exists not to do at run time.

### Flow features: relative to their own past

```
volume_t = log1p(volume_t) - median( log1p(volume) over [max(0, t-L), t-1] ),  L = 256
```

and the same for `amount`. The window is **half-open on the right**: it ends at `t-1`, never at
`t`. It expands from the segment start until it reaches `L`, then rolls.

No prior blending and no cold-start constant. The expanding phase *is* the warm-up, and `RM_1` is
the median of a single value, which is well-defined. Segment starts in *training* pass through the
same phase, so an inference context shorter than `L` sees a feature distribution the model has
genuinely trained on. That alignment is the argument for the design, not a consolation for it.

`amount` is kept for six-dimensional parity with Kronos. It is close to redundant with `volume`
plus the price features, and the Phase E distributions say how close.

### Clip at ±5, in scaled units, counted

The Kronos clip, applied to causal values. Clip events are counted per feature and reported. A rate
above 0.5 % on any (class × frequency × feature) is a red flag requiring a written investigation
before G2 closes — either the constants are wrong for that slice or the slice is pathological, and
v0.5 needs to know which before it picks quantizer ranges.

### Validity: refuse, never repair

A typed `ContractError` with a code, and no partial output, for: a non-positive price
(`non_positive_price`), any NaN or Inf (`non_finite`), a negative volume or amount
(`negative_flow`), a timestamp that does not strictly increase (`ts_not_increasing`), a high or low
outside the open/close range (`ohlc_inconsistent`), fewer than two bars (`too_short`).

NaN or Inf in the *output* on valid input is a bug in the contract, not a property of the data. It
is checked at the end of every transform and property-tested to zero.

**Gaps in the timestamp grid are not rejected.** A 24x5 series legitimately skips a weekend, and
the clean layer already adjudicated which absences are boundaries and which are sessions
(ADR-0018). Requiring exact grid contiguity here would reject every FX segment in the corpus. The
contract requires strictly increasing timestamps and nothing more. This is a deliberate reading of
the plan's "non-contiguous `ts`" clause, and it is the only one that survives contact with the
corpus.

### Dtypes: compute in float64, emit float32

Same-platform runs are bit-identical, and CI enforces it. Cross-platform golden comparisons use
`atol 1e-9`.

The emission cast has two visible consequences, both documented rather than papered over. Round-trip
error is bounded in **log** space, so a zero-volume bar following a very large one inverts to about
`1e-6` rather than to `0`. And a wick of exactly zero can round to put `high` an ulp under `open`, so
`inverse` projects reconstructed bars onto the valid-candle set. The projection also matters for
v0.9, where a model samples feature rows freely and nothing stops it sampling an upper wick below
the body.

### Causality, defined as a property rather than asserted as a claim

**Prefix-consistency:** `transform(bars[:t+1])` equals the first `t` rows of `transform(bars)` —
exactly, bit for bit, on the same platform. This is the definition. It is property-tested on
generated series in CI, marked `@causality` so v0.8's leakage suite can re-run it by marker, and
audited cloud-side on real segments in Phase E.

**Perturbation-invariance** is the second, independent probe: mutate bar `j`, and every row before
`j` is bit-identical. A transform that reached forward through a running statistic could conceivably
satisfy one formulation and not the other, so both run.

Note what prefix-consistency does *not* catch, because the distinction cost a test to find: a window
that includes bar `t` itself is still prefix-consistent. Self-normalization leaks nothing from the
future. That case is forbidden separately by the strictly-past window and caught by a boundary test
on the median. Two leaks, two properties, neither one sufficient alone.

### The rolling median is numpy, not `bottleneck`

The plan named `bottleneck.move_median` as the default with pandas and a two-heap streaming median
as fallbacks. **Deviation:** the implementation is the numpy sliding-window path — blocked
`np.median` over `sliding_window_view` — and no new dependency was added.

The reason is arithmetic. The Phase E corpus pass computes about 155 million rolling medians
(38.8 M bars × 2 flow features × 2 specs). The numpy path measures fast enough that the pass fits
comfortably in the job's budget, and a dependency bought nothing measurable. `bottleneck` remains
the documented upgrade path if v0.6's throughput requirement changes the arithmetic.

### The leaky baseline exists, guarded

`kronos-zscore-v0` reproduces Kronos's per-window z-score faithfully, flagged `leaky: true`.
`transform` refuses it unless the caller passes `allow_leaky=True`; `inverse` refuses it outright,
because the statistics it would need are not carried with the features. A test asserts it *fails*
prefix-consistency — documentation by test — and it exists only so v0.5 can put a number on what
the leak buys in reconstruction.

## Consequences

`SCHEMA_VERSION = 1` is a code constant re-exported from `axiom.contract`. Changing any number
above changes it, and changing it invalidates the constants, the golden vectors, the regression
snapshots, the tokenizer and every shard downstream. That is the intended cost.

Gate **G2** closes on: golden vectors and the property battery green across the Python matrix, the
`@causality` marker wired, prefix-consistency passing on synthetic data in CI and on real segments
cloud-side, constants committed with a firewall-respecting manifest, snapshot hashes committed, clip
rates within bounds or investigated in writing, and no tokenizer code in the repo.

The single-implementation rule is the one that will be tested by time rather than by pytest. When
v0.9's Predictor needs a feature the contract does not emit, the correct move is a
`schema_version = 2` and a retrained model, not a second transform beside this one.
