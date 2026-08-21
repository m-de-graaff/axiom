# ADR-0018: Cleaning semantics — what a segment is, and every rule that ends one

**Status:** Accepted (v0.3)

## Context

Kronos Appendix B, Algorithm 1 (`FilterLowQualitySegments`) is four paragraphs of pseudocode over
a corpus its authors describe but do not ship. Applying it to ours means answering questions the
paper leaves open: what counts as a missing bar when the market is legitimately shut, what
"near-zero volume" means as a number, and whether cleaning produces data or metadata.

Those answers have to be written down because they are not derivable from the algorithm. Somebody
reading `stages.py` in v0.7 needs to know that a choice was made, not guess that it was an
accident.

## Decision

### Cleaning emits a segment index, not a cleaned corpus

Raw is immutable (ADR-0010) and cleaning thresholds will be tuned. Materializing cleaned bars
would mean a full copy of the corpus per config change, and two places that both claim to say what
a series contains.

So the output of a clean run is a **table of intervals**: for each series, the contiguous
`[start_ts, end_ts]` spans that survive, with the rule that ended each one. Downstream — the v0.4
contract, the v0.5 tokenizer, the v0.6 shard job — reads raw bars *through* that index. Nothing
rewrites or filters a raw Parquet file, ever.

A segment row is bound to the exact bytes it was derived from: it carries `raw_artifact_sha256`
alongside `clean_config_hash` and `clean_version`. A raw file whose hash changed invalidates its
segments, and the clean run detects that rather than silently serving stale intervals.

### The five stages, in this order

The order is part of the config hash, and a test locks it by constructing a series whose segments
differ under any other ordering.

1. **Gap partition.** Split wherever the timestamp grid skips a step that the series' session says
   should have been there.
2. **Jump partition.** Split wherever `|open_t / close_{t−1} − 1| > θ_jump(freq)`.
3. **Illiquid excision.** Remove runs of `volume ≤ illiquid_eps` longer than `max_illiquid(freq)`.
4. **Stagnant excision.** Remove runs of exactly-equal `close` longer than `max_stagnant(freq)`.
5. **Min-length filter.** Drop what is left shorter than `min_bars(freq)`.

Partitioning before excision is what makes the run-length rules mean what they say. A run of zero
volume interrupted by a two-week outage is two runs, not one, and counting it as one would excise
bars on the strength of an absence.

### Jump rule

Cut between `t−1` and `t` when `|open_t / close_{t−1} − 1| > θ_jump(freq)`. Bar `t` opens a new
segment; bar `t−1` closes the old one. Both bars survive — a jump is a boundary, not a deletion.

The comparison is open-to-previous-close, so it fires on **overnight and cross-bar** discontinuity
only. An intrabar crash — a low that spikes down and a close that recovers — does not trigger it,
and should not: that is a real price path, fully contained in one bar.

**The inherited bias we keep and name.** The rule cannot distinguish an unadjusted corporate action
from a genuine 30 % overnight collapse, so genuine extreme moves get cut. Crypto delivers those
regularly. The consequence is that the model is systematically **under-exposed to extreme
discontinuities**: it will have seen the calm side of every crash boundary and not the boundary.
This is Kronos's behaviour and we keep it for comparability, but it is a known bias, not a
neutral choice. It goes in `docs/LIMITATIONS.md` and in the v0.9 model card.

### Gap rule — session-aware

This is the adaptation Kronos glosses over. A missing grid step is a hard boundary **only if it is
unexpected for the series' session** (`session_id`, ADR-0014):

- **`24x7`** (crypto): strict grid. Every step should hold a bar; any missing one is a boundary,
  because for a 24/7 exchange an absence is an outage.
- **`24x5`** (FX, commodities, index CFDs): the weekend window is expected and never a boundary.
  The window is Friday close to Sunday open, with tolerance for European DST — Dukascopy's week
  has opened at 19:00, 21:00 and 22:00 UTC across its history. Any gap that is not the weekend is
  a boundary.
- **`XNYS-regular`** (US equities, 1d): weekends and NYSE holidays are expected, resolved through
  the `exchange_calendars` package's XNYS calendar. Any non-calendar missing session — a trading
  suspension, a delisting hole, a vendor omission — is a boundary. The package is pinned; it was
  last released 2026-03-10 and is actively maintained.

**No imputation, anywhere.** Boundaries partition; they never fill. Kronos's Stage-1 step
"impute volume and amount to 0 when missing" is **N/A for our corpus**: volume is always present in
every source we carry, and amount is synthesized at parse time when a vendor does not publish it
(ADR-0010). There is nothing to impute.

### Illiquid run

Consecutive bars with `volume ≤ illiquid_eps`. The default `illiquid_eps` is **`0.0`** — exact
zero, meaning nothing traded.

The paper says "near-zero", which is not a number. Ours is a number, it is in the config, and it
is in the hash. Choosing exact zero rather than a percentile threshold keeps the rule about
*absence of trading* rather than about thinness, which the stagnant rule and the universe
liquidity screen already handle from different angles.

Runs strictly longer than `max_illiquid(freq)` are excised, and excision splits the surrounding
series into two segments. A run of exactly `max_illiquid` bars is kept.

### Stagnant run

Consecutive bars with exactly equal `close`. Float equality is deliberate and is meaningful here:
prices come from a stable CSV-or-binary → float64 parse with no arithmetic applied, so two bars
that printed the same price produce the same double. Two bars that printed different prices do
not accidentally compare equal.

Runs strictly longer than `max_stagnant(freq)` are excised.

**What this excises that is real.** US limit-up/limit-down halts and thin small-caps print
repeated closes for legitimate reasons, and this rule removes them. That is accepted: a model
that learns "the price is often exactly what it was" from halted tape learns a vendor artifact.
The cost is documented rather than avoided, and it is quantified in the drop-stats report.

### Min-length filter

A surviving segment shorter than `min_bars(freq)` is dropped, because a segment shorter than the
model's context window cannot produce a training window. `min_bars` is Kronos Table 4.

### Binding and invalidation

Every segment row carries `clean_config_hash`, `clean_version`, and `raw_artifact_sha256`.

- A changed threshold changes the config hash, and segments from the old hash are not valid for
  the new one. `axiom clean run` refuses `--incremental` across a config-hash change.
- A changed raw file changes its `sha256`, and the staleness guard re-cleans exactly those series.

## Consequences

- **Cheap to re-tune.** Changing `θ_jump` for 1h costs one clean run over metadata, not a corpus
  rewrite.
- **One source of truth.** There is no "cleaned corpus" that can drift from raw.
- **Every consumer pays a join.** v0.4 onward must read bars through the segment index rather than
  reading a file and trusting it. That is the price of not duplicating, and it is enforced by the
  segment table being the only thing that says which bars are usable.
- **Two named biases.** Extreme-move excision (jump rule) and halt excision (stagnant rule) are
  carried into `LIMITATIONS.md` with measured numbers, not left as caveats.

## Alternatives rejected

**Materialize cleaned Parquet.** Simplest for consumers, and wrong: it duplicates ~2 GB per config
change and creates a second thing that claims to be the corpus.

**Percentile-based illiquidity.** More adaptive, and unhashable in any useful sense — the
threshold would depend on the series, so the same config would mean different cuts on different
data.

**Impute missing bars.** Rejected in ADR-0010 and rejected again here. A filled bar is a fact the
market did not produce.
