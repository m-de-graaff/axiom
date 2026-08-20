# ADR-0014: Cross-source conventions for the raw tier

**Status:** Accepted (v0.2). Extends ADR-0010, and supersedes one paragraph of it.

## Context

ADR-0010 defined bar schema v1 against a single source. One source cannot disagree with itself,
so several questions never came up: what a price *is* when the source quotes a bid rather than a
trade, what `volume` counts when the venue does not publish one, and what the timestamp of a
daily bar means for a market that is closed most of the day.

v0.2 adds three sources with three different answers to each. The answers have to be recorded per
file, in the manifest, or the corpus becomes a pile of numbers whose meaning depends on which
loader happened to write them.

## Decision

### `price_side` — what the four prices are quotes of

New manifest field, one value per file:

| Value | Meaning | Sources |
|---|---|---|
| `trade` | Prices at which trades actually executed | Binance, Stooq |
| `bid` | The bid side of a broker's quote stream; no trade need have occurred | Dukascopy |

Dukascopy publishes bid and ask candles separately. Taking bid only is a decision, not an
oversight: mixing the two within a corpus would put a spread-width step into series that have no
spread information attached, and a model would learn the step. The spread is recoverable later by
pulling the ask side as a second series if it ever matters; it is not recoverable from a silently
blended one.

### `volume_convention` — what the volume column counts

| Value | Meaning | Sources |
|---|---|---|
| `base+quote_native` | `volume` is base asset, `amount` is the venue's own quote volume | Binance |
| `dukascopy_tick_volume` | `volume` counts **quote updates**, not traded size | Dukascopy |
| `shares` | `volume` is shares traded | Stooq |

`dukascopy_tick_volume` is the one that will mislead somebody eventually, so it gets said here as
well as in ADR-0015: an FX broker has no exchange to report size, and the number in that column is
an activity proxy. It is monotonically related to real volume and it is not real volume. Any
downstream feature that treats volume as size is comparing Binance's dollars against Dukascopy's
tick counts, and the manifest field is what makes that visible rather than plausible.

### `amount_synthesized` — the Kronos rule, always flagged

A source with no native quote volume gets `amount = volume × mean(open, high, low, close)` and
sets `amount_synthesized = true`. That applies to **Dukascopy and Stooq**, both of them, always.
Binance keeps its measured `quote_asset_volume` and the flag stays false.

The rule was already in ADR-0010. What is new is that it now fires on the majority of the corpus
by file count, so the flag stops being a footnote and becomes something v0.4's contract has to
read before it decides whether `amount` is a usable feature.

### `exchange_tz` and `session_id` stay metadata — superseding ADR-0010

ADR-0010 said, in its identity section: *"All Binance series are `exchange_tz = "UTC"`,
`session_id = "24x7"`. These become real columns in v0.2, when session-bound markets arrive and
the values start to vary."* `docs/ARCHITECTURE.md` repeated it.

**That is now rejected.** Both fields remain file-level metadata: Parquet key-value block, sidecar
manifest, path. No column is added. Schema stays **v1** — no migration, no re-pull, no version
bump.

The teaser's premise was that v0.2's markets would make the values vary. They do vary *between*
files — `America/New_York` for Stooq, `24x5` for Dukascopy — but they are still constant *within*
every file v0.2 writes, because identity is what selects a file. Promoting them to columns would
add sixteen bytes a row to store a string that the filename already determines, on ~50 M rows, to
answer a question no reader has: nobody opens `raw/stooq/us/1d/A/AAPL.parquet` and wonders which
timezone it is in.

The case that would justify columns is a *per-row* session label — bar 14 is regular-hours, bar 15
is after-hours — and that requires intraday bars in a session-bound market. v0.2 has no such
series: equities are daily-only, FX runs one continuous 24×5 session. When intraday equities
arrive (post-1.0 at the earliest), the column is added then, against real variation, and it will
be a `session` column carrying per-bar values rather than a constant repeated a million times.

Recording the deviation rather than quietly dropping it is the point of this section. The earlier
ADR made a prediction about a version that had not been planned yet, and the prediction was wrong
in a way worth being explicit about.

### Daily-bar timestamps for session-bound markets

For a 1d bar on a market with a trading calendar, `ts` is **00:00:00 UTC of the exchange calendar
date** — not the session open, not midnight local, not the close.

The alternative — timestamping at the local session open — would make `ts` jump an hour twice a
year at DST boundaries and would put US and European daily bars on grids that never align. The
calendar date is what the vendor publishes, what a human means by "AAPL on 2024-06-10", and what
makes `count_gaps` arithmetic work against an 86 400 000 ms step.

The real trading timezone is not lost: it is `exchange_tz` in the metadata
(`America/New_York` for US equities, `UTC` for crypto and FX). A consumer that needs the actual
session open reconstructs it from `exchange_tz` plus the calendar; a consumer that needs a
comparable daily grid uses `ts` directly. Both work, because the two facts are stored separately
rather than mashed into one field.

### `redistribution_class`

New manifest field naming the row in `docs/DATA_LICENSING.md` that governs the file. Values:
`loader_manifest_private_cache` (default, and what every v0.1 sidecar means) and
`loader_only_private`. It is recorded per artifact rather than per source because the licensing
table is the kind of document that gets updated after the data is already on disk, and a file that
carries its own class can be found by a query instead of by remembering which loader wrote it.

### `staging_exception_used`

Boolean on the pull-run manifest, default false. True when market-data bytes transited the laptop
under the ADR-0016 sanctioned fallback. The roadmap's "zero bytes on the laptop" rule has exactly
one permitted exception, and an exception nobody can count is a rule nobody is keeping.

## Consequences

Every new field is optional with a default equal to what v0.1 already meant. Existing crypto
sidecars stay valid unread and unrewritten; `is_current` compares source checksum lists and is
untouched, so no v0.1 artifact re-pulls because of this ADR.

`schema_version` stays 1 across v0.2. The only thing that would bump it is a column change, and
this ADR exists partly to say that none is happening.

The cost of keeping session metadata out of the columns is that a future intraday-equities version
pays a migration. That is the correct place to pay it: at the version where the data justifies the
column, rather than at the version that guessed it might.
