# ADR-0008: Repo topology

**Status:** Accepted (v0.0)

## Context

Cloud jobs need to install the package, and checkpoints need somewhere durable to live that is not
the ephemeral filesystem of a kernel about to be killed. Both need to happen without publishing
anything.

## Decision

A single private monorepo, `m-de-graaff/axiom` on GitHub. Cloud jobs install it with
`pip install git+https://…` using a fine-grained read-only PAT. If GitHub is ever unavailable or
undesirable, the fallback is `uv build` into a wheel attached as a private Kaggle Dataset — a
documented plan B, not an implemented one.

Hugging Face carries the data and artifacts, one private repo per tier, each created in the
version that first needs it:

| Repo | Type | Created in | Holds |
|---|---|---|---|
| `axiom-runs` | dataset | v0.0 | Checkpoints, run manifests, `latest.json` resume pointers |
| `axiom-trackio` | space | v0.0 (optional) | trackio dashboard sync |
| `axiom-raw` | dataset | v0.1 | Cleaned-source Parquet plus provenance manifests |
| `axiom-tokenized` | dataset | v0.6 | Pre-tokenized MDS shards |
| `axiom-model` | model | v0.9 | `model.safetensors`, `config.json`, model card |

Everything is private until the Publish Gate, at which point `axiom-tokenized` and `axiom-model`
flip public and the rest stay private.

## Consequences

`docs/REPOS.md` is the living version of this table and is updated in the same session any repo is
created. A repo that exists online but not in that file is a documentation bug.

Tokens are scoped to match: the GitHub PAT can read one repo, and the Hugging Face token can write
only to repos matching `axiom-*`. Rotation is every 90 days, tracked in `docs/RUNBOOK.md`.
