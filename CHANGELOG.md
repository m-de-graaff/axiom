# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses SemVer-style 0.x
versioning with git tags.

## [Unreleased]

## [0.2.0] - 2026-08-21

Corpus **M0 assembled**. The raw tier grew from one source to four: FX and commodities from
Dukascopy, the whole US equity market from Stooq, and split/dividend events from Yahoo, joined by
a queryable registry over every sidecar in `axiom-raw`.

**13,580 artifacts, 42,358,938 bars, 1.90 GB.** All five M0 slices present at their required
frequencies. The bar count is **below** the roadmap's ~50 M target and is recorded as a shortfall
with an analysis, not massaged into a pass — see `docs/reports/v0.2-raw-qa.md`.

The laptop still holds no market data. ADR-0016's staging exception was never used;
`staging_exception_used` is false in every pull run manifest.

### Verified

Three carried unverifieds were resolved by measurement, and **two came out opposite to
expectation**. Dukascopy returns 403 to GitHub Actions runners and answers a Kaggle kernel, so
that one pull moved backends permanently. A Stooq handoff URL returned 404 from both a runner and
the laptop within minutes — expired, not IP-bound, which matters because only the IP-bound case is
what the laptop staging exception exists for. Yahoo, expected to block datacenter IPs, answered
503 of 503 tickets with zero failures.

The adjustment audit returned **`split_and_dividend_adjusted`**. AAPL, NVDA and TSLA show close
ratios of 0.9672, 0.9926 and 1.0035 across splits of 4:1, 10:1 and 3:1, and across 20 sampled
tickers the median relative difference against Yahoo's dividend-adjusted closes is 0.0129. Stooq's
closes track a total-return path.

That inverts a v0.3 assumption: the plan expected to *build* a total-return series for eval labels
from a price path, and the measurement says the tokenization series is the one that has to be
derived instead.

Every v0.1 sidecar was re-read by v0.2 code and still verified its own recorded hash.

### Added

- **A source framework** (`sources/base.py`). One driver owning the skip test, validation, the
  Parquet write, the sidecar, the run manifest and the per-item blast wall; a source supplies
  `plan`, `build`, `manifest_extras` and `artifact_path`. Every v0.1 test passed unedited against
  it, which was the refactor's acceptance test.
- **Dukascopy loader** (ADR-0015). Bid candles, year-chunked, prior years immutable. The skip test
  is the shared one: a sealed year's token is constant, the current year's carries the run's
  as-of date.
- **Stooq loader** (ADR-0016). The bulk US archive via a human-solved CAPTCHA and a URL handed to
  a cloud job. Letter-sharded layout, and tolerances that differ on purpose — a short series is
  skipped, malformed lines are dropped up to 0.1% of a file, and a duplicate date fails outright
  because every other defect is absence of information and that one is a contradiction.
- **yfinance adjunct** (ADR-0016). Splits and dividends for a pinned cross-check population,
  rate-limited to 300 requests an hour, non-load-bearing by construction.
- **Corpus registry.** One table over every sidecar, answering what/from-where/pulled-when without
  touching the raw files. A rebuild from an unchanged tier reproduces the same `registry_hash`.
  A sidecar it cannot read is reported, never dropped.
- **Cross-source manifest conventions** (ADR-0014): `price_side`, `volume_convention`,
  `redistribution_class`, `staging_exception_used`, and per-series statistics for weekend padding
  and dollar volume. Fields holding their v0.1-equivalent default are dropped from the identity
  hash, so no existing sidecar was invalidated.
- **`docs/DATA_LICENSING.md`**, classifying all four sources. The loaders are publishable; the
  bars are not, from any of them.

### Changed

- **`exchange_tz` and `session_id` stay metadata**, superseding v0.1's stated plan to promote them
  to columns (ADR-0014). They vary between files and are constant within one, so a column would
  store per row what the path already fixes. Schema stays v1.
- **The source repository is public** (ADR-0017), taken to get unlimited Actions minutes after the
  private-repo allowance ran out mid-version. PyPI, `axiom-tokenized` and `axiom-model` still wait
  for the Publish Gate, and `axiom-raw` is now private permanently rather than "until the gate".
- **Hub commit batching, 50 to 2,000 files.** The Hub allows 128 commits an hour; the equities
  tier at the old size is roughly 500 and died partway. At the new one it is 13, and the pull
  finished in 49 minutes with zero Hub errors.
- **The equities universe ranks from the registry**, not from downloads. Its statistic is computed
  at pull time, when the bars are already in memory.

### Fixed

Four bugs, all found by real runs, and all the same shape: **an error being silently converted
into a value.**

- A rate-limited Parquet read returned `0.0`, which the ranking's `> 0` cut then discarded — so a
  Hub 429 became "this stock has no volume". One run measured 1,344 of 6,829 candidates and
  reported "978 of 978 kept". Failures are no longer representable as rankings, and the build
  refuses to emit a universe when more than 1% is unmeasured.
- A Hub 429 on `upload_folder` escaped `pull_ticker` and killed a 503-ticker run that had already
  landed 125. The blast wall covered only the fetch, not the store.
- `run_pull`'s final `store.flush()` sat outside its blast wall, so a failing last commit crashed
  the process and took the run manifest with it.
- The cross-check asked Yahoo for Stooq's spelling of the ticker (`SMXT.US`), which 404s. Every
  comparison would have failed and the report would have read as "Yahoo would not answer".

And one rule that was simply wrong about real data: **weekend bars are counted, not rejected.**
Asserting an empty weekend failed 24 of 27 hourly FX series. Measuring the feed across its own
history showed why — the week opened at 19:00 UTC in 2003, and some eras pad the weekend with
flat zero-volume bars carrying the Friday close forward. Rejecting a 155,000-bar series over 8,000
rows of vendor padding is the undocumented cleaning pass ADR-0010 exists to forbid.

## [0.1.0] - 2026-08-20

The canonical bar schema, the provenance-manifest system, and a checksum-verified, resumable
Binance loader that ran cloud-to-cloud and landed 600 series in a new private dataset.

The laptop never held a market-data byte. Every fixture in the test suite is a synthetic zip built
in-test, every real byte was fetched by a runner and written to the Hub, and `git status` is clean.

### Verified

600 series in `m-de-graaff/axiom-raw`: 200 spot and 100 USDT-M perpetual symbols at 1h and 1d,
10,885,159 bars, 0.57 GiB. **225 distinct symbols are present at both frequencies with at least a
year of history**; the exit gate asks for 100.

Re-pull reproducibility: **10/10** sampled series rebuilt byte-identically from the archives their
manifests name. Cross-check against `binance_historical_data`: **3/3** agree on row counts and on
every OHLCV value of a sampled day. The full pull finished `ok=440 skipped=160 failed=0`, with no
unexplained failures because there were no failures.

The kill drill was a real `gh run cancel`, which SIGKILLs the runner. 30 series had been built and
29 committed; the relaunch was the same dispatch with no resume flag, because there is no resume
flag, and reported `ok=131 skipped=29 failed=0`.

Zero Kaggle GPU-hours. Zero Modal spend — Modal still has no account (ADR-0013).

### Added

- **Bar schema v1** (ADR-0010). UTC epoch milliseconds, base volume alongside native quote
  amount, three retained raw columns, and identity carried by the path and the Parquet metadata
  rather than by a column repeating itself once per row. Invariants are enforced at parse time,
  so nothing downstream has to defend against a high below its own open.
- **Provenance manifests.** A sidecar beside every Parquet file naming every source archive and
  the checksum the upstream published for it, plus one manifest per pull run recording what
  landed, what was skipped, what failed, and whether the run was partial.
- **A Binance Vision loader** (ADR-0012). S3 enumeration rather than a date range, checksums
  verified before extraction, retries with jittered backoff behind a global concurrency cap, and
  a monthly/daily seam that must agree value-for-value or fail.
- **A pinned universe** (ADR-0011), `universe_v1.yaml`, hashed and committed. 200 spot and 100
  USDT-M symbols ranked on July 2026 quote volume, with leveraged tokens, stable-to-stable pairs,
  fiat pairs, and anything with less than 12 months of history excluded.
- **`axiom raw inspect | verify | stats`.** Reproduce a failing series and see the offending
  rows; re-derive a sample from the archives its manifest names and compare the bytes; reduce the
  sidecars into the committed QA report.
- **Two cloud jobs** (ADR-0013), `universe.yml` and `pull.yml` on GitHub Actions, plus a dry-run
  mode that runs the whole fetch-and-parse path against the real bucket and publishes nothing.

### Changed

- **Off-grid bars are a warning, not a violation.** The first real run failed spot 1h BTCUSDT on
  43 rows: consecutive hourly bars from 2018-02-09, phase-shifted by 28m14.789s after an exchange
  restart, each still exactly an hour after the last. They are real bars, so they are kept and
  counted into `off_grid_count`. Rejecting them would have cost the corpus its most important
  series; snapping them to the grid would have been imputation.
- **The minimum-history rule moved to selection time.** Ranking July 2026 volume alone put seven
  tokenized equities and metals — `NVDAUSDT`, `TSLAUSDT`, `SOXLUSDT` — in the top ten USDT-M
  perpetuals, and two recently launched stablecoins in the top ten spot pairs. A history rule
  removes all of it, and the next batch too, without anybody maintaining a list of which tickers
  are secretly stocks.

### Known limitations

- **Still not Modal.** The pull runs on GitHub Actions (ADR-0013), which shares an account and an
  outage with the code host, so backend #2 still does not deliver the vendor independence the
  roadmap wanted. Deferred again to v0.6, where the pre-tokenization map job will set requirements
  a data pull does not.
- **Byte-identity is conditional on the writer.** `artifact_sha256` is the hash of Parquet bytes,
  and pyarrow stamps its own version into every file it writes. The 10/10 result holds for a given
  pyarrow; a major upgrade will change the hashes without changing a single bar. The manifest's
  content identity (`manifest_sha256`) is unaffected, which is why it is the field the Parquet
  metadata carries.
- **Selection-month bias.** The universe is ranked on one month's volume, so a pair that was
  dominant in 2021 and is quiet now can miss the cut despite years of good history (ADR-0011).
- **No corpus registry.** Answering "what do we have" means reading 600 sidecars. That is fine at
  this size and is what v0.2 replaces.
- **The v0.1 manifests record `loader_version` `0.0.0+<commit>`.** The package version was bumped
  to `0.1.0` after the pull ran, so every sidecar written during v0.1 names the version that
  actually produced it, which was still `0.0.0`. Harmless by design — `loader_version` is outside
  the manifest identity hash precisely so that the build that wrote a file cannot change what the
  file is — and left as it stands rather than rewritten, because a manifest that claims a version
  it was not written by is worse than one that is merely surprising.

## [0.0.0] - 2026-08-20

First version. A private monorepo and a proven develop-local, execute-remote loop: one command
dispatches a dummy trainer to the cloud, which pulls the code, runs, checkpoints full state to
Hugging Face, survives a real mid-run kill, and resumes bit-identically.

No market data was touched and no GPU minutes were spent. Both were hard non-goals.

### Added

- **The Loop.** A dummy trainer whose state — step, accumulator, and the RNG position of every
  generator — round-trips through an atomic, sha256-verified checkpoint on a private Hugging Face
  dataset. Killing a run and resuming it produces the same final float, to the last bit.
- **Three execution backends running one CLI path.** The laptop, a Kaggle CPU kernel, and a
  GitHub Actions runner all call `axiom loop run` with no special cases between them.
- **Config identity.** A config hash that ignores `run_id` and `backend_tag`, so the same
  experiment hashes the same everywhere, and resuming a checkpoint whose config has changed is
  refused rather than silently blended.
- **Reproducibility core.** Seeding across `random`, NumPy, and torch; RNG capture and restore;
  run provenance (config hash, commit, package, Python, torch, backend) logged before step one.
- **CI** on GitHub Actions: lint, types, and tests across Python 3.11, 3.12, and 3.13.
- **Nine ADRs** closing the six open design decisions from the research document plus toolchain,
  repo topology, and the backend-#2 substitution.

### Verified

Kill-and-resume produces `acc=3018.7626345157623` at step 6000 whether the run was interrupted or
not, on both cloud backends and the laptop. Both kills were real — Stop Session on Kaggle,
`gh run cancel` on Actions — not fault injection.

Kaggle's image is Python 3.12.13 with torch 2.10.0+cpu, which settles the provisional floor in
ADR-0007 at `>=3.11` with no amendment needed.

### Known limitations

- **Modal is not the second backend.** Its account sits behind a review gate, so GitHub Actions
  stands in (ADR-0009). Actions shares a vendor with the code host, so v0.0 does not deliver the
  vendor independence the roadmap wanted from backend #2. `remote/modal/loop_test.py` is written
  and unrun.
- **Kaggle dispatch is two steps.** `kaggle kernels push` destroys the kernel's secret
  attachment, and the Kaggle API has no field to declare secrets, so a push must be followed by
  re-attaching them and clicking Save Version. `docs/RUNBOOK.md` carries the recipe.
