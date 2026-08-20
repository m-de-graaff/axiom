# ADR-0016: US daily equities from Stooq, with yfinance as an adjunct

**Status:** Accepted (v0.2)

## Context

Equities are the fourth asset class and the only one in the corpus with corporate actions. A split
is a discontinuity that is not a price move, and a corpus that contains unadjusted splits teaches
a model that stocks routinely fall 75 % in a day.

The free sources are all compromised in some way. Stooq publishes a bulk archive of the whole US
market but gates it behind a CAPTCHA. Yahoo has an unofficial API that yfinance scrapes, no
license, and an active habit of blocking datacenter IPs. Nothing free offers both full history and
an automated download.

The constraint that shapes the answer is the roadmap's: **no market-data bytes on the laptop.**

## Decision

### Stooq bulk is the corpus pillar

`https://stooq.com/db/h/` publishes `d_us_txt.zip` — every US ticker's full daily history as one
`.txt` per symbol, format `TICKER,PER,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT` with `PER=D` and
`DATE=YYYYMMDD`. Roughly 12–18 k series and tens of millions of bars for a couple of gigabytes.

Kept: the NASDAQ, NYSE and NYSE-MKT/AMEX **stocks and ETFs** directories. Skipped: indices,
futures, and the non-US trees. An index is not a tradeable instrument with a volume, and futures
belong to a contract-rollover problem this version does not open.

Stooq ships no checksum. The archive's `sha256` is computed on arrival and recorded in the
manifest, which makes it self-consistent rather than vendor-verified — a weaker guarantee than
Binance's `.CHECKSUM` and one worth naming as such.

### Ingestion is manual-assisted, with a URL handoff

The CAPTCHA has gated that page since December 2020 and there is no API key to ask for. So the
human does the one thing only a human can do, and nothing else:

1. Mark solves the CAPTCHA in the laptop browser.
2. He copies the resulting direct archive URL.
3. `axiom pull stooq --archive-url <url>` hands it to the cloud job, which downloads it there.

The archive bytes never touch the laptop. What crosses the machine is a URL, which is not market
data.

### The sanctioned fallback, and its accounting

If the URL turns out to be bound to the IP that solved the CAPTCHA, the cloud download 403s and the
handoff cannot work. Then, and only then:

download on the laptop → `hf upload` straight to `axiom-raw/staging/stooq/` → **delete the local
copy immediately** → the cloud job consumes from staging → `staging/` is pruned once the parse
succeeds.

This is the **single** permitted exception to the zero-bytes rule in the whole project. It is not
permitted quietly: `staging_exception_used = true` goes into the pull-run manifest (ADR-0014), the
local deletion is logged in `docs/RUNBOOK.md` terms, and the v0.2 exit checklist has a line asking
whether it fired. An exception that is invisible in the record is indistinguishable from the rule
having been abandoned.

### Parse tolerances

- **Fewer than 30 rows** — skipped, recorded as `skipped_short`. Not a failure: a ticker that
  listed last month is a fact about the market, and a 12-bar series is not usable by anything
  downstream.
- **Malformed lines** — counted per file. Above 0.1 % of a file's lines, the file fails. Below, the
  bad lines are dropped and the count is recorded. Vendor text dumps have occasional damage and
  failing a 9 000-row series over one truncated line would cost more than it protects.
- **Duplicate dates within a ticker** — hard fail, no tolerance. Every other defect here is
  *absence* of information; a duplicate date is a *contradiction*, and a raw tier that carries
  contradictions forward is not trustworthy for anything.

### Layout

`raw/stooq/us/1d/{first_char}/{TICKER}.parquet`, letter-bucketed.

The Hub degrades past roughly 10 000 files in one folder and 12–18 k series would land right on
that line. First-character bucketing splits them into 27-ish folders of a few hundred to a couple
of thousand, which no bucket exceeds even with the distribution's lumpiness. A guard test asserts
the planned layout keeps every folder under 9 000 files.

### Normalization

`.us` is stripped from the ticker into `symbol` and kept whole in `source_symbol`. `ts` is 00:00
UTC of the calendar date per ADR-0014. `volume` is shares (`volume_convention = "shares"`),
`amount` is synthesized and flagged, `OPENINT` is dropped — it is a futures field that is
identically zero in the equity dumps. `exchange_tz = "America/New_York"`,
`session_id = "XNYS-regular"`.

`adjustment_policy` starts at the placeholder `vendor_adjusted_unverified` and is replaced by an
evidence-based value once the audit below runs. A guess dressed as a recorded fact is worse than
an admitted unknown.

### yfinance is an adjunct and never a pillar

yfinance supplies two things and neither is load-bearing:

1. **Split and dividend event series** for a pinned ticker list, so v0.3 has the corporate actions
   as data rather than as an inference from price discontinuities.
2. **The cross-check** that classifies what Stooq's adjustment actually is.

It is scraped from an endpoint with no license and no stability promise, hard-rate-limited to
≤ 300 requests/hour on our side, and expected to be blocked from cloud IPs at least some of the
time. If it fails entirely, that is written down as a dated outcome and v0.2 proceeds — the split
probes in the adjustment audit stand on the Stooq data alone.

Its output is `redistribution_class = loader_only_private`: cached privately, never redistributed,
not even in the form the rest of the corpus is not redistributed in.

### yfinance reachability, measured (2026-08-20)

The expectation above -- that Yahoo would refuse cloud IPs at least some of the time -- was
tested rather than assumed, and it did not hold. A twenty-ticker smoke from a GitHub Actions
runner returned **20 of 20 tickers and 2085 events with zero failures**.

That is the opposite of the Dukascopy result on the same backend, and the contrast is the useful
part: Dukascopy fingerprints callers to an interactive chart endpoint and refuses datacenter
ranges outright, while Yahoo's action endpoint currently does not. Neither behaviour is a
promise. Yahoo has no licence and no stability guarantee, so this is recorded as a dated
observation, not as a property to depend on, and every failure path in the loader stays exactly
as written.

The consequence for v0.2 is only that the cross-check has both of its inputs available, and that
the audit will not have to open with an unavailability note. The adjunct remains non-load-bearing:
if this stops being true tomorrow, the split probes below still stand.

### The adjustment audit

Whether Stooq is split-adjusted, split-and-dividend-adjusted, or unadjusted decides what v0.3's
policy has to do, and the vendor does not document it. So it is measured:

- **Split probes.** AAPL 2020-08-31 (4:1), TSLA 2022-08-25 (3:1), NVDA 2024-06-10 (10:1). If the
  series has no ~N:1 discontinuity across those dates, it is split-adjusted. This test needs only
  Stooq data and a calendar, so it survives yfinance being unavailable.
- **Dividend probe.** One large special dividend and one ordinary-dividend blue chip, compared
  against yfinance's `auto_adjust=True` and `auto_adjust=False` closes. Which of the two the Stooq
  path tracks is the answer.

The verdict goes in `docs/reports/v0.2-adjustment-audit.md` and the `adjustment_policy` field is
regenerated for `raw/stooq/**` only.

### Survivorship: accepted and documented

Stooq's bulk dump skews heavily toward currently-listed tickers. Companies that went bankrupt or
were acquired are thinly represented or absent, so a model trained on this corpus has mostly seen
firms that survived.

This is **not** fixed in v0.2. Fixing it means a delisted-securities database, which is a paid
product. It is recorded here, it will be recorded again in the v0.9 model card, and any backtest
number this corpus ever produces is biased upward because of it. Naming a limitation precisely is
worth more than an unaffordable partial correction that makes it harder to reason about.

### The training universe is deferred, its criteria are not

The equities universe is not hand-pinned. Its **criteria** are pinned here — at least 5 years of
history, ranked by median daily dollar volume (`close × volume`), top ~3 000 — and the list itself
is generated in Phase F from the pulled data, because the ranking metric requires the data to
exist.

The pulled corpus is a superset of the training universe. Everything stays in `axiom-raw`; the
universe governs sampling from v0.5 onward, not what gets stored.

## Consequences

The equities tier is the only part of the corpus that cannot be refreshed by a cron job. Every
update needs a human and a CAPTCHA. That is acceptable for a corpus that is deliberately frozen
between versions, and it is the reason the roadmap's risk register lists equities as the fragile
asset class and Binance plus Dukascopy as the automated backbone.

Equities also dominate the corpus by file count — 12–18 k series against roughly 350 crypto and 54
Dukascopy — which is what forces the folder sharding and what makes the Phase F registry worth
building rather than a `list_repo_files` call.
