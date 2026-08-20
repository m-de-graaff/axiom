# Data licensing and redistribution

What each source permits, and what that means for what this project may ship.

Every source here is free to fetch. None of them is free to republish. The distinction the corpus
is built on is between **the loader and the manifest**, which are ours and are publishable, and
**the bars**, which are the vendor's and stay private.

## The classes

| Class | What may be published | What may never be |
|---|---|---|
| `loader_manifest_private_cache` | The loader code, the provenance manifest, aggregate statistics | The bars themselves, in any repo, at any visibility |
| `loader_only_private` | The loader code | The manifest content and the data, both private-only |

Every sidecar manifest carries a `redistribution_class` field naming its row (ADR-0014). It is
recorded per artifact rather than inferred per source, so the question "may this file be
published" is answerable by a registry query rather than by remembering which loader wrote it.

## The sources

| Source | What it is | Terms | Class | Notes |
|---|---|---|---|---|
| **Binance Vision** (`binance_vision`) | Public S3 bucket of the exchange's own kline dumps | Public data dumps, no click-through licence, no stated redistribution grant | `loader_manifest_private_cache` | The most permissive of the four, and still not permissive enough to republish under. Checksums are vendor-published (`.CHECKSUM`), so reproducibility is verifiable against the vendor rather than against ourselves. |
| **Dukascopy** (`dukascopy`) | A Swiss broker's own historical quote feed | Broker datafeed offered for client use; no redistribution licence of any kind | `loader_manifest_private_cache` | Legally the greyest source in the corpus. Fetching is unrestricted in practice and republishing is plainly not granted. **Never redistribute, including in derived tokenized form.** |
| **Stooq** (`stooq`) | Bulk daily archive of the US market | Personal, non-commercial use per the site's terms | `loader_manifest_private_cache` | The corpus is personal research and stays inside that. No vendor checksum ships with the archive; the `sha256` recorded in the manifest is self-computed. |
| **Yahoo Finance** via yfinance (`yahoo_events`) | Scraped from an undocumented endpoint | No licence at all — not permissive, not restrictive, absent | `loader_only_private` | The strictest handling for the weakest claim: no licence means no grant, so even the event manifests stay private. Adjunct only; the corpus does not depend on it. |

## What this means downstream

**`axiom-raw` is private and stays private.** There is no version of the publish gate that flips
it. Its contents are four vendors' data in our schema, and putting our schema on it does not make
it ours.

**`axiom-tokenized` becomes public at the publish gate, and that is a decision this table does not
settle.** Tokenized shards are uint16 code pairs, not prices — a derived representation that
cannot be inverted back to the bars without the tokenizer, and lossily even then. Whether that is
far enough from redistribution to publish is a question to answer at the gate with the vendor terms
in hand, not one to assume now because the answer would be convenient. The Dukascopy row is the one
that decides it.

**`axiom-model` is publishable.** Model weights are not the data, by every reading of every row
above.

**Reproducibility is by recipe, not by copy.** Anyone with the repo can re-derive the corpus from
the loaders and the manifests: the manifest names every source URL and its checksum, so a third
party fetches from the vendor under their own terms rather than receiving the bytes from us. That
is what makes the private-cache class workable rather than merely restrictive — the work stays
reproducible without the data ever moving.

## Review

This table is checked at every version close and at the publish gate. If a vendor's terms change,
the class changes here first and the sidecars are regenerated to match — the field exists so that
regeneration is a targeted operation rather than a re-pull.
