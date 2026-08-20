# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses SemVer-style 0.x
versioning with git tags.

## [Unreleased]

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
