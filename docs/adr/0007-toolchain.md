# ADR-0007: Toolchain

**Status:** Accepted (v0.0)

## Context

A solo project run across a laptop, two free cloud backends, and a home inference box needs its
environment reproducible from a lockfile, and needs one CI configuration that is the authority on
whether the code is acceptable. Two linters or two type checkers with different opinions is a
standing tax.

The Python floor is constrained by whatever Kaggle's image ships, which is not knowable until a
kernel actually runs.

## Decision

- `uv` for environment and lockfile. `uv.lock` is committed.
- Ruff for both linting and formatting. No Black, no isort, no flake8.
- One type checker as the CI source of truth: `ty` to start. If it blocks on real code, we switch
  to `mypy` and amend this ADR rather than running both.
- pytest with hypothesis for property tests.
- typer for the CLI, pydantic-settings for config, trackio for experiment tracking.
- Python floor `>=3.11`, tested against 3.11, 3.12, and 3.13 in CI.

`torch` is pinned to the CPU wheel index so the laptop and CI never download CUDA. Cloud images
bring their own build.

## Consequences

The Python floor is provisional until a Kaggle kernel reports its actual interpreter version in
Phase F. If Kaggle ships something older than 3.11, this ADR is amended and the floor drops.

`ty` is pre-1.0 and may be wrong about valid code. The escape hatch is written into the decision
so hitting it is a planned switch rather than a mid-project argument.
