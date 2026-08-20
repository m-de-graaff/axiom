# ADR-0012: A thin custom fetcher for Binance Vision

**Status:** Accepted (v0.1)

## Context

`data.binance.vision` publishes monthly and daily kline zips under a documented, stable URL
scheme, each with a `.CHECKSUM` sibling. Three ready-made ways to consume it exist:
`binance_historical_data` (the one the roadmap named), `binance-data-loader`, and Binance's own
`binance-public-data` scripts.

What v0.1 needs from a fetcher is a specific list: checksum verification *before* extraction,
control of the on-disk layout, a sidecar manifest written next to every artifact, and resume by
comparing remote manifests against source checksums. None of the three libraries offers the last
three, and wrapping one to get them means fighting its own layout and cache conventions.

## Decision

**Write the fetcher.** `src/axiom/sources/binance_vision.py` builds the URLs, enumerates available
months from the S3 XML listing, downloads with retry and a concurrency cap, verifies the
`.CHECKSUM`, and hands bytes to the parser. It is roughly two hundred lines because the URL scheme
is four format strings and the hard parts — checksums, idempotence, manifests — are ours either
way.

Enumeration comes from the S3 listing rather than a date range, because listing and delisting
leave real holes and a date range would turn every hole into a 404 that has to be guessed about.
A 404 on a month the listing named is a hard error; a 404 on a daily-tail probe is expected.

**`binance_historical_data` is retained as an independent cross-check.** It stays a dev
dependency, and Phase G3 runs it over three symbols and diffs the result against `axiom-raw`. A
custom fetcher's real risk is a systematic misunderstanding — an off-by-one on bar open time, a
column in the wrong position — and the way to catch that is a second implementation that made its
own mistakes, not more tests written by the same author against the same assumption.

## Consequences

We own the maintenance. If Binance changes the layout, nothing upstream fixes it for us. The
mitigation is that the layout has been stable for years and the fetcher's surface is small enough
to re-derive from the docs in an afternoon.

We also own the politeness. A global semaphore caps concurrent requests at 12 and retries back off
exponentially with jitter on 429 and 5xx. Binance Vision is a public S3 bucket with no published
rate limit, which is a reason to be conservative rather than a licence not to be.

Header handling is the one place the format actually bites: Binance shipped these files headerless
for years and started including a header row on newer ones, in the same directory tree. The parser
sniffs the first line and skips it when it is not numeric, rather than deciding by date.
