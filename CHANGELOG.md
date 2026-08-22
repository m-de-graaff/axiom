# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses SemVer-style 0.x
versioning with git tags.

## [Unreleased]

## [0.4.0] - 2026-08-22

The preprocessing contract, frozen. Two parameterizations of a bar sequence into six causal
features, behind four functions that training pre-tokenization and the inference Predictor will
both import — so a drift between what the model is trained on and what it is asked to predict
from becomes a test failure rather than a silent skew.

Causality is a property here, not a claim. `transform(bars[:t+1])` is exactly the first `t` rows
of `transform(bars)`, bit for bit, and everything else in this version is that requirement applied
somewhere specific.

### Added

- **`axiom.contract`** (ADR-0020). `load_spec`, `load_constants`, `transform`, `inverse`, and
  nothing else — a test asserts the package exports exactly those four plus `SCHEMA_VERSION`.

  Two specs are frozen at `schema_version = 1`. `geo-v1` is the primary: gap, body, upper wick,
  lower wick, plus two flow features. Its structural identities — `upper >= max(0, body)` and
  `lower <= min(0, body)` — are why it is primary; four independent log-returns spend model
  capacity relearning that high is above low. `ret-v1` is the A/B challenger, identical in every
  respect a consumer can see, so v0.5's reconstruction study measures the parameterization and
  nothing else.

  Volume and amount are `log1p` minus the median of `log1p` over the bars **strictly before** the
  one being scaled — expanding from the segment start until it reaches 256 bars, then rolling. No
  prior blending and no cold-start constant: the expanding phase is the warm-up, and a training
  segment passes through the same phase, so an inference context shorter than the window sees a
  distribution the model has genuinely trained on.

  Bar 0 of every segment is the **anchor** and produces no feature row. It seeds the gap feature,
  the first strictly-past median, and the Predictor's price inversion. Usable windows are
  therefore `Σ max(0, (n_bars − 1) − 511)`, one fewer per segment than v0.3 published.

- **The frozen scaling constants**, fitted over **31,102,283 pre-firewall bars** across 23,758
  segments, 0 artifacts failed. Per (asset class × frequency × feature), robust median and
  IQR/1.349, committed to the repo because constants are part of the contract and a constants file
  living somewhere else can drift from the transform that reads it.

- **The temporal firewall at 2025-01-01** (ADR-0021), pulled forward from v0.5 deliberately: a
  constant fitted before the boundary is chosen is a constant that chose the boundary. It leaves
  19.6 months of post-firewall history in the shortest M0 slice, against an 18-month floor.
  Enforcement is in code — the fit truncates before computing anything, records the `max(ts)` it
  consumed, and a constants file whose manifest says the assertion failed does not load at all.

- **The causality audit.** Prefix-consistency and perturbation-invariance, property-tested on
  generated series and tagged `@causality` so v0.8's leakage tripwires select on the marker rather
  than on file names. Two permanent tests prove the audit can fail: a forward-looking median
  window, and the leaky `kronos-zscore-v0` baseline, which exists only so v0.5 can put a number on
  what Kronos's per-window z-score buys in reconstruction. Production paths refuse it.

- **Six golden fixtures × two specs, checked twice.** Once against frozen output, and once against
  a second implementation written from ADR-0020's formulas rather than from `axiom.contract` —
  plain Python, `statistics.median`, one bar at a time. Frozen output alone would only prove the
  implementation has not changed, including if it was wrong the day it was frozen.

- **`axiom contract fit-constants | dryrun | show`**, a GitHub workflow and a Modal twin.

### Fixed

- **`inverse` produced bars the schema refuses.** Float32 emission rounds a zero-volume bar to a
  hair below zero, and `expm1` faithfully returns −1.5e−10; a wick of exactly zero can round to put
  `high` an ulp under `open`. Reconstructed bars are now projected onto the valid-candle set, which
  also matters for v0.9, where a model samples feature rows freely and nothing stops it sampling an
  upper wick below the body.

- **The quantile sketch could not resolve a log-price IQR.** Uniform bins over raw feature units
  cannot span ±20 for a flow feature and still resolve an hourly body's interquartile range of
  about 10⁻³. Bins are now uniform in `asinh(x / 1e-4)`, which holds relative resolution near
  0.05 % across the whole support.

- **The corpus fan-out queued every artifact up front**, so each completed result — about 6 MB of
  sketches — stayed alive inside its `Future` until the consumer reached it. The first corpus-wide
  fit was killed at 3,000 of 10,647 with nothing in the log but a cancellation. Submission is now
  bounded and futures are dropped as they are consumed.

### Changed

- `docs/LIMITATIONS.md` records the per-class scaling trade-off. One `scale` covers every
  instrument in an asset class, and the volatility spread inside "crypto" is wide. Per-series
  adaptive scaling is a named post-1.0 experiment, rejected here because a per-series statistic is
  a per-series fit, and a fit at inference time is the failure the contract exists to prevent.

### Deviations from the plan

Both recorded in ADR-0020. The rolling median is the numpy sliding-window path rather than a new
`bottleneck` dependency — the corpus arithmetic does not need it, and `bottleneck` stays the
documented upgrade path. And the plan's "non-contiguous `ts`" rejection reads as "strictly
increasing": a 24x5 series legitimately skips a weekend, the clean layer already adjudicated which
absences are boundaries (ADR-0018), and requiring grid contiguity would reject every FX segment in
the corpus.


## [0.3.1] - 2026-08-22

The v0.3 gate left one thing on the list: all 12,425 Stooq sidecars still recorded
`vendor_adjusted_unverified`, which is what the loader believed *before* the audit ran. It is
fixed, and fixing it flushed out three more instances of a bug already fixed three times.

### Added

- **`axiom raw stamp-verdict`** (ADR-0019 amendment). Writes a measured audit verdict into a
  source's sidecars. Ran over the corpus: **12,425 stamped, 0 failed**, in 100 seconds.

  Correcting `adjustment_policy` in place was never possible — it is inside `manifest_sha256`,
  which is stamped into every Parquet's own metadata, so editing it breaks the file-to-sidecar
  link on twelve thousand artifacts, moves `artifact_sha256`, and invalidates the entire segment
  index. So the verdict goes in a **second** field, `adjustment_policy_verified`, held outside the
  identity hash — the category `VOLATILE_MANIFEST_FIELDS` already existed for. Two fields rather
  than one correction, because what was believed at pull time and what was measured afterwards
  are two different facts and both are worth keeping.

  No Parquet was rewritten, no `artifact_sha256` moved, and `clean/v1/` stayed valid throughout.
  The registry carries the column, so `axiom derive tr` reads the verdict instead of falling back
  to a constant and printing a note about the divergence.

### Fixed

- **The per-file Hub download, in its fourth, fifth and sixth homes.** `list_manifests`,
  `build_registry` and `derive tr` each fetched all 13,580 sidecars one at a time. The registry
  rebuild lost 19 of them to `ReadTimeout` and published a registry with **13,561 artifacts**,
  which `clean run` reads — so nineteen series would have been skipped with nothing saying so.
  All three now take one snapshot. The registry rebuilt clean at 13,580, `registry_hash`
  `1ef9ebb7a4a3`.
- **Hub HTTP timeouts.** `huggingface_hub` reads with a ten-second timeout and
  `snapshot_download` abandons the whole batch on the first file that exceeds it, so every retry
  round died early and all eight were exhausted. Ten seconds is fine for a handful of files and
  not for thirteen thousand: 120s download, 60s etag, twelve rounds. The Hub is not slow here, it
  is rate-shaping, and waiting is the correct response.
- **A partial clean run no longer publishes over the full index.** `--limit` is for smoke runs,
  and a smoke run must not become the corpus — `clean/v1/` overwritten with fifty series looks
  exactly like a corpus that shrank rather than like a test, which is what happened the first
  time and why the drop-stats report showed 50 series for a while. `--force` overrides.


## [0.3.0] - 2026-08-21

The **cleaning pass**, delivered as metadata rather than as a corpus. Kronos Appendix B
Algorithm 1, with every interpretive choice the paper leaves open pinned in ADR-0018 instead of
buried in the code.

**13,077 series · 27,905 segments · 38,758,930 of 42,308,244 bars kept (8.39 % dropped) · 0
failed.** The usable number is smaller and it is the one that matters: **27,508,145 context-512
windows**, which is what v0.5 sizes the tokenizer corpus against rather than an estimate.

Raw Parquet was never rewritten. The whole output is three files under `clean/v1/` plus a
coverage manifest under `derived/tr_close/`, so re-tuning a threshold costs a rerun over intervals
instead of a copy of the corpus. Zero market-data bytes reached the laptop; every corpus run was
a GitHub runner.

### Added

- **A segment index** (ADR-0018). Per series, the contiguous `[start_ts, end_ts]` spans that
  survive, with the rule that ended each one, bound to the `raw_artifact_sha256` of the file they
  came from and to the config hash that produced them. Downstream reads bars *through* it.
- **Session-aware gap partitioning** — the adaptation Kronos glosses over. A missing bar is a
  boundary only if the session did not expect it: a strict grid for crypto, a DST-tolerant weekend
  for FX, a settlement break for commodity CFDs, the XNYS calendar for US equities. Nothing is
  ever imputed.
- **`axiom.testing.synth`**, fifteen generators with ground truth about where cuts belong, in
  timestamp space so pathologies compose. Library code, not test helpers — v0.4's contract tests
  and v0.8's leakage tripwires reuse it. It imports nothing from `axiom.clean`, and a test asserts
  that by parsing the imports: a toolkit sharing the engine's calendar would agree about weekends
  by construction.
- **`axiom clean probe`**, which explains *why* a series fragmented. The drop statistics report an
  instrument that lost everything to a thousand one-hour gaps identically to one that lost it to a
  single decade-long hole. The probe reports gap sizes, which hour of the day the holes fall in,
  and whether the dead bars sit inside the window where the market was shut. Both v0.3 definition
  bugs were found with it.
- **Post-clean views**: usable bars beside usable windows, per-rule drop rates, the top-20
  most-cut list, and three red-flag checks. `docs/reports/v0.3-clean-qa.md` carries a written
  investigation against every hit.
- **`docs/LIMITATIONS.md`**, feeding the v0.9 model card.
- **Total-return policy** (ADR-0019) behind one interface with both branches tested.

### Verified

**The audit inverted a v0.3 assumption and the plan followed the measurement.** Stooq is split
*and* dividend adjusted, so `tr_close` is an identity for every source. The plan called for a
letter-sharded Parquet tier; under the measured verdict that would have been twelve thousand
byte-for-byte copies of a column already in the file beside it, which is the duplication the plan
itself refuses for crypto. `axiom derive tr` writes a coverage manifest instead. Coverage is
100 %.

**Survivorship, quantified.** Across 12,425 Stooq series spanning 1962 to 2026, the earliest
*last* bar anywhere in the tier is 2026-02-06, and **not one ticker stopped trading more than a
year before the pull date**. The real market delists several percent of listings a year. The
archive is not a sample of market history; it is a snapshot of the currently-listed market with
each survivor's history extended backwards, and every equity number downstream is conditioned on
surviving to 2026.

### Fixed

Five bugs, every one found by a real run against the real corpus, and none by a test written in
advance.

- **Segment ids were not unique.** Binance lists the same ticker on spot and on USDT-M futures,
  both at 1d, beginning the same day: `ACEUSDT:1d:1702857600000` named two segments. Thirty-two
  collided. The corpus-wide invariant check caught it and refused to upload rather than publishing
  a table with duplicate keys.
- **Commodity CFDs were declared `24x5`.** All six show thousands of gaps exactly one slot wide in
  UTC hours 21 and 22, on every weekday — XAUUSD alone had 2,369. That is a settlement break, not
  an outage. Undeclared, it cost **100 % of Brent, copper, natural gas and silver, and 54 % of
  gold**. Fixed with a config override rather than a re-pull.
- **Weekend padding was excised as an illiquid run.** 8,177 of EURUSD's 8,235 zero-volume bars sit
  inside the weekend window. Excising them was right; *partitioning* there was not, because the
  hole they leave is the weekend the session already expects — and the resulting 120-bar weeks
  could not survive `min_bars = 256`. EURUSD lost 20.8 % of its history to it, USDJPY 21.4 %.
  A stage-zero filter now removes them before any rule sees them, and it tests volume as well as
  time: the weekend window is wider than any one DST regime, and deleting on the window alone
  threw away real Friday-evening bars.
- **Two of the three red-flag checks measured the wrong thing.** Major-series loss counted the
  session filter as damage, flagging every FX instrument for the cleaner working correctly. The
  weekend check could never fire, because the gap rule partitions and never drops a bar, so its
  drop count is zero by construction.
- **The corpus job kept being rate-limited off the Hub.** Thirteen thousand `hf_hub_download`
  calls is twenty-six thousand requests and earns a 429 however few threads make them.
  `snapshot_download` halves that and is resumable, so the retry that works wraps the whole
  snapshot rather than each file — and the runner cache now saves on failure, because the runs
  that most need their partial download kept are exactly the ones that did not finish.

### Changed

- **`axiom derive tr` reads the registry**, not every sidecar. The first version asked the store
  for all 13,580 and was rate-limited for it; everything it needs is already a registry column.
- **The adjustment verdict lives in code, pinned by ADR-0019**, not in the sidecars. All 12,425
  Stooq sidecars still record `vendor_adjusted_unverified`, which is what the loader honestly
  believed before the audit ran. Correcting them is a re-pull rather than an edit —
  `adjustment_policy` is inside `manifest_sha256`, which is stamped into each Parquet — so both
  values are kept and the command says when they differ.
- **`exchange-calendars` pinned** `>=4.13,<5`, in the `data` and `dev` extras and a new
  `calendars` one. NYSE holidays back to 1962, including Good Friday and the one-off closures, is
  not something to hand-roll.


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
