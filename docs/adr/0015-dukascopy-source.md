# ADR-0015: FX and commodities from Dukascopy

**Status:** Accepted (v0.2)

## Context

The corpus needs a second asset class with a long history and a session calendar, because a model
trained only on 24×7 crypto has never seen a market close. FX is the obvious candidate: two
decades of history, no licensing fee, and a weekend gap in every series that forces the cleaning
and contract layers to handle absence properly rather than assuming a dense grid.

Dukascopy is a Swiss broker that publishes its own historical feed openly. It is not an exchange.
Everything below follows from that one fact.

## Decision

### The library, not the bucket

Ingestion goes through **`dukascopy-python`** (≥ 4.0.1), using its interval API:

```python
dukascopy_python.fetch(instrument, interval, offer_side, start, end)
```

which returns a pandas DataFrame indexed by a tz-aware UTC `timestamp` with columns
`open, high, low, close, volume`. Verified live against `EUR/USD` at both `1HOUR` and `1DAY`
before this ADR was written.

**Tick-level `.bi5` assembly is out of scope.** Dukascopy's raw feed is one compressed file per
instrument-hour holding individual quotes, and reconstructing bars from it means owning the
decompression, the point-value scaling per instrument, and the bar-boundary rules. That is a
project (`tick-vault` is one). It would buy the ability to build 5m and 15m bars, which the
roadmap only wants at corpus M1, only for crypto, and only if G3 says so. The library's
pre-aggregated hourly and daily candles are what v0.2 needs, and they arrive already correct.

The famous trap in the raw scheme is that Dukascopy's URLs use **0-indexed months** — January is
`/00/`, December is `/11/` — and an off-by-one there silently shifts a whole series by a month.
This ADR's response is not to be careful about it: it is to never construct one of those URLs.
The library owns the scheme. A test that the library handles it correctly ships in
`tests/test_dukascopy.py` as a date-round-trip assertion on a known window, because "we use a
library that presumably gets it right" is not a check.

### Bid candles

`OFFER_SIDE_BID`, per ADR-0014's `price_side` field. Dukascopy publishes bid and ask separately;
taking one side consistently is what keeps the series free of an artificial spread-width step. The
manifest records `price_side = "bid"` on every file so nothing downstream has to guess.

### Tick volume is not volume

The `volume` column counts **quote updates in the bar**, not traded size. A broker has no exchange
tape to report against. The number is a genuine activity signal and it is not comparable to
Binance's base-asset volume in any units.

Recorded as `volume_convention = "dukascopy_tick_volume"`. `amount` has no native equivalent at
all, so it is synthesized as `volume × mean(OHLC)` with `amount_synthesized = true` — which means
the `amount` column for these series is a tick count multiplied by a price, an activity proxy
scaled into price units. It is retained for schema uniformity and flagged so v0.4 can decide
whether to feed it to anything.

### Sessions, weekends, and what a daily bar is

FX runs one continuous session from Sunday evening to Friday evening: `session_id = "24x5"`,
`exchange_tz = "UTC"`.

The session boundary follows Dukascopy's server clock, which observes European DST — the week
opens at 22:00 UTC on Sunday in winter and 21:00 UTC in summer, and closes at the matching hour on
Friday. So the loader's weekend assertion is not a sharp edge but a window that is safe in both
regimes: **no bar may fall between Saturday 00:00 UTC and Sunday 20:00 UTC.** A tighter rule would
fail twice a year on real data; a looser one would not catch anything.

Daily bars inherit the same boundary. The 1d series carries a bar stamped on **Sunday** — the
two-to-three-hour tail of the week's opening evening, with a volume an order of magnitude below a
weekday — and carries **no bar on Saturday**. Both were confirmed in the live probe. Neither is an
error, and neither is repaired: absence of a Saturday bar is counted by `count_gaps` exactly like
a crypto outage would be, which is the correct treatment because in both cases the honest statement
is "no bars exist here".

That does mean roughly one gap per week appears in every Dukascopy daily series' gap count. The QA
report says so rather than the number being read as damage.

### The universe holds the symbol map, not the code

`src/axiom/configs/universe_dukascopy_v1.yaml` pins 27 instruments — 7 FX majors, 14 crosses,
2 metals, 4 energy/industrial commodities — each with its canonical `symbol`, its `source_symbol`,
its `asset_class`, and the **measured** first date of its daily history.

The plan called for a Python map from canonical symbols to library instrument constants, and a
test proving the map total over the universe file. There is no such map. `dukascopy-python`'s
`INSTRUMENT_*` constants are plain strings (`INSTRUMENT_FX_MAJORS_EUR_USD == "EUR/USD"`), so the
map's whole content is a string per instrument, and a map in code plus a universe in YAML is two
files that can disagree about which instruments exist. Putting `source_symbol` in the universe file
makes them one file, deletes the totality test along with the thing it was testing, and puts the
value straight where the manifest needs it.

Start dates are measured, not assumed. Every instrument was fetched from 2003-01-01 to now and its
first bar recorded: FX majors reach back to 2003-05, most crosses to 2003-08, `AUDCAD` only to
2010-01, and the commodity CFDs to 2010–2014. Pinning a start year earlier than the data exists
would spend a request per empty year forever; pinning one later would silently truncate history.

Selection is by notoriety-liquidity — the instruments a human would name if asked for the liquid
FX pairs and the traded commodities. There is no measurement step, because for a hand-pinned set
of 27 a ranking procedure would be ceremony around a decision already made. This is the same
reasoning ADR-0011 rejected for Binance, and the difference is the size of the candidate pool:
choosing 300 of 3 000 needs a criterion, choosing 27 of 27 does not.

### Year-chunked work, immutable prior years

Work is enumerated per instrument × frequency × **calendar year**. A year of hourly bars is about
6 000 rows, which bounds a request and makes a killed run resumable at a useful granularity.

Prior years are **immutable by convention**: 2019's EURUSD bars are not going to change, so once a
year is in the artifact it is never re-fetched. Only the current calendar year is re-fetched on a
re-run; it is spliced onto the earlier years read back from the existing Parquet, and the file is
rewritten whole. The alternative — Parquet append — does not exist, and rewriting a 5 MB file is
cheaper than the machinery for pretending it does.

The v0.2 exit gate verifies this byte-wise on a sample: re-running a pull must leave every prior
year's rows identical.

### Index CFDs: off

Dukascopy also carries index CFDs (`USA500IDXUSD` and friends) which would give an equity-index
proxy with FX-like session behaviour. **Default off, not in the universe file.** The corpus already
gets equity exposure from Stooq with real share volume and a real exchange calendar, and a CFD
index tracks a spot index through a broker's quote with no volume information worth the name.
Adding it would be a fifth thing to explain in the model card in exchange for a series the corpus
already covers better. Reversing this is one block in the universe YAML if a later version wants it.

## Consequences

The corpus gains roughly 4 M bars — about 170 k daily and 4 M hourly across 27 instruments — for a
few hundred megabytes. That is a fifth of what the crypto tier holds and it is the only source with
history before 2017, so it carries disproportionate weight in anything that asks the model about a
regime crypto never saw.

### Reachability, measured (2026-08-20)

The fallback ladder was not hypothetical. Three hosts were asked directly, and they disagree:

| Host | Egress | `freeserv.dukascopy.com` | Full fetch |
|---|---|---|---|
| Laptop | Residential | 200 | 27 instruments, full history |
| GitHub Actions runner | Azure (`74.235.127.166`) | **403, 0 bytes** | 0 bars across 24 years |
| Kaggle CPU kernel | GCP (`34.80.255.184`) | 429 on a bare request | **315 daily bars for 2024** |

**So the Dukascopy pull runs on Kaggle, not on GitHub Actions.** This is the ladder's first rung
firing exactly as written, not a new decision -- but it does mean ADR-0013's "the data jobs run on
Actions" now holds for Binance and Stooq and not for this source.

Two details worth keeping, because both would otherwise be re-learned the hard way:

The endpoint the client reads is `freeserv.dukascopy.com/2.0/index.php`, a JSON chart service --
**not** the static `.bi5` datafeed. That is why the block bites: a CDN serving flat files has no
reason to fingerprint callers, and an interactive chart backend does.

A bare `curl` of that endpoint from Kaggle returns 429 while `dukascopy-python`'s own request path
returns data from the same host seconds later. The library gets through where a naive request does
not, so **the raw endpoint's status code is not a reliable proxy for whether the pull will work.**
The probe kernel checks both for that reason, and its verdict line is the fetch, not the curl.

Rung two of the ladder -- reduced concurrency with backoff -- was not needed and is not tuned. It
would not have helped anyway: Actions returns 403 on the very first request, which is a refusal to
serve this host rather than a complaint about how often it asks.

The corpus now contains one asset class whose volume column means something different from every
other asset class's. `volume_convention` is the field that keeps that recoverable, and v0.4 owns
the decision about what to do with it.
