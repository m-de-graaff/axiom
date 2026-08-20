# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses SemVer-style 0.x
versioning with git tags.

## [Unreleased]

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
