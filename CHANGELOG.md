# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses SemVer-style 0.x
versioning with git tags.

## [Unreleased]

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
