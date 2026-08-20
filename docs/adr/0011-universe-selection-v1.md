# ADR-0011: Binance universe selection v1

**Status:** Accepted (v0.1)

## Context

"The 100 most liquid pairs" is not a specification. Liquid measured how, on what day, with which
of Binance's several thousand symbols excluded? Answered informally, the universe drifts every
time somebody re-runs the builder, and every downstream number — corpus size, eval coverage,
the survivorship claim in the model card — drifts with it.

## Decision

The universe is **code**: a deterministic function of a pinned selection month and an explicit
criteria list, emitted to `src/axiom/configs/universe_v1.yaml` and committed. Re-running the
builder on the same month must produce the same file, byte for byte.

### Metric

Rank by the summed `quote_asset_volume` of the selection month's **1d** bars, per symbol, per
market. Quote volume rather than base volume because base volume is not comparable across assets.
The selection month's daily zips are a few kilobytes each, so ranking the whole listing costs one
small download per candidate.

**Selection month: `2026-07`** — the last month that was complete when v0.1 was built. It is
pinned, not computed, so the builder is reproducible after the calendar moves on.

### Candidate filter

- Quote asset `USDT`, in both `spot` and `um`.
- The symbol must appear in the S3 listing for that market with a `1d` file for the selection
  month. Absence means delisted or not yet listed; either way it cannot be ranked.

### Exclusions

**Leveraged tokens** (`BTCUPUSDT`, `ETHBEARUSDT`, …). Their price is a rebalanced derivative of
the underlying, not a market. The rule is not a bare `(UP|DOWN|BULL|BEAR)$` suffix match — that
eats `JUPUSDT`, whose base is the Jupiter token. A symbol is leveraged only when stripping the
suffix leaves at least two characters *and* the stripped symbol is itself in the listing:
`BTCUPUSDT` → `BTCUSDT` exists → leveraged. `JUPUSDT` → `JUSDT` does not exist → kept.

**Stable-to-stable pairs** — `USDCUSDT`, `TUSDUSDT`, `FDUSDUSDT`, `DAIUSDT`, `BUSDUSDT`,
`USDPUSDT`, `SUSDUSDT`, `USDSUSDT`, `AEURUSDT`, `EURIUSDT`. A pair pinned to 1.0000 by
construction teaches a price model that prices do not move.

**Fiat-quoted pairs** — `EURUSDT`, `GBPUSDT`, `AUDUSDT`, `JPYUSDT`, `TRYUSDT`, `BRLUSDT`, and the
rest of the fiat bases. These are FX, and FX belongs to the Dukascopy loader in v0.2, where it
arrives with proper session metadata and a real tick history. Admitting them here would put the
same instrument in the corpus twice under two provenances.

Depegged former stables (`USTCUSDT`) are **kept**. Whatever they were designed to be, what they
print is real price action.

### Counts

`spot`: top 200. `um`: top 100. Both are headroom over the ≥ 100 exit gate, which is measured
after the pull on pairs that clear the history rule at *both* 1h and 1d.

### Minimum history

A pair counts toward the exit gate when its 1h series spans **at least 12 months** —
`last_ts - first_ts >= 365 days`. This is checked against the pulled manifests, not at selection
time, because the S3 listing tells you a file exists and not how many rows are in it. Short-history
symbols are still pulled; they are simply not counted.

### Identity

`universe_hash` is the config hash (ADR-0007's `canonical_json` + sha256, 12 hex characters) over
the criteria block **and** the emitted symbol lists. Every artifact manifest carries it, so any
Parquet file in `axiom-raw` can be traced to the universe definition that asked for it.

## Consequences

The universe is frozen until somebody deliberately bumps it to `universe_v2.yaml`. New listings do
not appear in the corpus on their own, which is the point: a corpus that grows silently is a corpus
whose eval results cannot be compared across time.

Selecting on a single month's volume biases toward whatever was in fashion in July 2026. A pair
that was dominant in 2021 and is now quiet ranks low and may be excluded despite years of good
history. This is accepted for v0.1 — a multi-month or history-weighted metric is a v0.2 refinement
if the corpus turns out to be too narrow.

Ranking by the selection month excludes anything listed after it. Survivorship bias in the other
direction — delisted pairs never enter — is inherited from ranking on a recent month at all, and is
recorded here so the model card's limitations section has something to cite.
