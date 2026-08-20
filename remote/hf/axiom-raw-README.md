---
license: other
tags:
  - private
  - not-for-redistribution
---

# axiom-raw

The raw tier of the [`axiom`](https://github.com/m-de-graaff/axiom) corpus: source market data,
translated into one canonical schema, with a provenance manifest beside every file.

**This repository is private and its contents are never redistributed.** The bytes here are a
reproducibility cache. What `axiom` publishes, if it ever publishes anything, is the loader and
the manifests — enough for somebody else to fetch the same data from the same upstream and check
they got the same thing. Not the data itself.

## Layout

```
raw/binance/{spot|um}/{1h|1d}/{SYMBOL}.parquet
raw/binance/{spot|um}/{1h|1d}/{SYMBOL}.parquet.manifest.json
manifests/pulls/{pull_run_id}.json
```

One Parquet file per series, zstd-compressed, row groups of 131 072. Beside each one, a sidecar
manifest naming every source archive that went into it and the sha256 the upstream published for
each. The pull job has no other checkpoint state: a killed run resumes by comparing those
checksums against what the upstream publishes now.

## Schema

Bar schema v1, defined in ADR-0010 of the code repository.

| Column | Type | Meaning |
|---|---|---|
| `ts` | int64 | Bar **open** time, UTC, milliseconds |
| `open`, `high`, `low`, `close` | float64 | Prices in the quote asset |
| `volume` | float64 | **Base**-asset volume |
| `amount` | float64 | **Quote**-asset volume |
| `n_trades` | int64 | Trade count, retained raw |
| `taker_buy_volume` | float64 | Taker buy base volume, retained raw |
| `taker_buy_quote_volume` | float64 | Taker buy quote volume, retained raw |

Source, market, symbol, frequency, `exchange_tz` and `session_id` are constant within a file, so
they live in the path and in the Parquet key-value metadata rather than in a column.

## Raw means faithful

No filtering, no imputation, no outlier handling. Zero-volume bars and stagnant runs are kept:
they are what the v0.3 cleaning pass is measured against, and a raw tier that quietly dropped
them would be an undocumented cleaning pass wearing a different name.

Gaps in the timestamp grid are recorded in the manifest and never filled. The one exception to
"nothing is removed" is the monthly/daily seam, where the same bar is published in two archives.
The two copies must agree value for value; one is kept, and a disagreement fails the file.

## Provenance

Every manifest records the source URLs, the upstream checksums, the row count, the timestamp
range, the gap count, the volume convention, whether `amount` was synthesized (it is not, for
Binance — the exchange publishes quote volume natively), the adjustment policy, the loader
version, and the hash of the universe definition that asked for the symbol.
