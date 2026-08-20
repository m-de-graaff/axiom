# ADR-0010: Canonical bar schema v1 and the raw tier

**Status:** Accepted (v0.1)

## Context

Every source downstream of here — Binance now, Dukascopy and Stooq in v0.2 — arrives with its own
column names, its own timestamp unit, its own idea of what "volume" counts. Something has to be
the shape everything else is translated into, and it has to exist before the first loader, or the
first loader becomes the shape by accident.

The second question is what "raw" means. A raw tier that quietly drops a suspicious bar is not a
raw tier; it is an undocumented cleaning pass, and v0.3's cleaning statistics would be measured
against an already-cleaned baseline without anybody knowing.

## Decision

### Bar schema v1

Six required columns, three optional ones, and nothing else:

| Column | Type | Meaning |
|---|---|---|
| `ts` | int64 | Bar **open** time, UTC, **milliseconds** since epoch |
| `open`, `high`, `low`, `close` | float64 | Prices in the quote asset |
| `volume` | float64 | **Base**-asset volume |
| `amount` | float64 | **Quote**-asset volume |
| `n_trades` | int64, nullable | Trade count, when the source has one |
| `taker_buy_volume` | float64, nullable | Taker buy base volume |
| `taker_buy_quote_volume` | float64, nullable | Taker buy quote volume |

`amount` is Binance's native `quote_asset_volume`. The Kronos synthesis rule
`amount = volume × mean(OHLC)` is reserved for sources that have no native quote volume, and when
it is applied the manifest says so in `amount_synthesized`. A synthesized amount and a measured
one must never be indistinguishable after the fact.

The three optional columns are kept because storage is cheap and taker flow may matter after 1.0.
Nothing downstream of v0.1 reads them; OHLCVA is the contract.

`close_time` and `ignore` are dropped. The first is derivable from `ts` and the frequency, the
second is a Binance placeholder.

### Identity lives in the path and the manifest, not in columns

`source`, `asset_class`, `market`, `symbol`, `frequency`, `exchange_tz`, `session_id` are constant
within a file. Storing them as columns would cost a column per row to say the same thing the
filename already says. They live in three places instead: the path, the sidecar manifest, and the
Parquet key-value metadata block (`axiom_schema_version`, `source`, `asset_class`, `market`,
`symbol`, `frequency`, `manifest_sha256`).

All Binance series are `exchange_tz = "UTC"`, `session_id = "24x7"`. These become real columns in
v0.2, when session-bound markets arrive and the values start to vary.

### Invariants, enforced at parse time

Violations fail the file rather than being repaired:

- `ts` strictly increasing
- `ts` aligned to the frequency grid — multiples of 3 600 000 ms for `1h`, of 86 400 000 ms for `1d`
- `high >= max(open, close)`, `low <= min(open, close)`, `high >= low`
- `volume >= 0`, `amount >= 0`
- no nulls in OHLCVA

Gaps in the grid are **allowed**. They are counted into the manifest and never filled.

### Raw is faithful

The raw tier reproduces the source. No filtering, no imputation, no outlier handling. Zero-volume
bars and stagnant runs are kept — they are exactly what v0.3's `FilterLowQualitySegments` is
measured on. The single exception is the monthly/daily seam, where the same `ts` legitimately
appears in two source files: the overlap must agree value-for-value, and one copy is kept. A
disagreement is a failure, not a merge.

### Layout and Parquet settings

```
raw/binance/{spot|um}/{1h|1d}/{SYMBOL}.parquet
raw/binance/{spot|um}/{1h|1d}/{SYMBOL}.parquet.manifest.json
```

One file per (market, frequency, symbol). zstd compression, row groups of 131 072 rows. One file
per series rather than per month keeps the file count in the low thousands, which is what the Hub
is comfortable with, and makes the sidecar manifest a per-series statement.

## Consequences

Timestamps are milliseconds everywhere, forever, and sources are normalized into that at parse
time rather than carrying their own units forward. Binance Vision has shifted between milliseconds
and microseconds on some datasets, so the parser detects the unit by magnitude rather than
trusting the documentation.

A schema change after this point is a `schema_version` bump and a re-pull, not an in-place edit.
That is the cost of putting identity in metadata: metadata is cheap to write and expensive to
migrate.
