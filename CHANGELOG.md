# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses SemVer-style 0.x
versioning with git tags.

## [Unreleased]

### Added
- Repo skeleton: `uv` src layout, Ruff, `ty`, pytest with hypothesis, pre-commit, `justfile`.
- Config core: `AxiomSettings` and `LoopConfig` with unknown-key rejection, plus a config hash
  that ignores volatile fields so the same experiment hashes the same across runs.
- Determinism core: `seed_all`, and RNG capture/restore across `random`, `numpy`, and `torch`.
- The Loop: a dummy trainer that checkpoints full state to a private Hugging Face dataset,
  survives a mid-run kill, and resumes bit-identically.
- CI on GitHub Actions: lint, types, and tests across Python 3.11, 3.12, and 3.13.
- Eight ADRs closing the open design decisions from the research document.
