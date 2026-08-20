# Axiom

A foundation model for K-line (OHLCV bar) sequences, built in two stages: a quantizer that turns
continuous candles into discrete tokens, then a small autoregressive decoder trained over those
tokens. Private and unpublished. Working title `axiom`; the distribution name is chosen at the
Publish Gate.

## Honesty banner

Expected out-of-sample performance: directional accuracy 50–53%, RankIC 0.00–0.04. Volatility is
the genuinely forecastable target and the centerpiece of evaluation. The durable value of this
project is the reproducible tokenizer, the BSQ-vs-FSQ quantizer comparison, and an eval harness
that reports what it finds rather than what would look good.

This banner goes verbatim into the model card.

## Where things are

- `.artifacts/roadmap.md` — the v0.0 → v1.0 version ladder, gates, and compute budget.
- `docs/ARCHITECTURE.md` — the component map and which version delivers each piece.
- `docs/adr/` — the locked design decisions.
- `docs/GATES.md` — what each gate required and the evidence it passed on.
- `docs/REPOS.md` — every repo and account this project uses.
- `docs/RUNBOOK.md` — tokens, rotation, dispatch recipes, what to do when a session dies.

## Getting started

```sh
uv sync --all-extras
uv run axiom version
just check
```

## Operating rules

The laptop dispatches and develops. It never holds corpus bytes and never trains. Training runs
on free-tier cloud (Kaggle, Modal) and checkpoints to a private Hugging Face dataset. The home PC
runs inference on ROCm and receives exactly one artifact: the final `.safetensors` plus its
config.
