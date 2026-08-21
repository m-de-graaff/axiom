# Repos and accounts

The living version of roadmap §3 and ADR-0008. Anything that exists online belongs in this table
on the day it is created. A repo that exists but is not listed here is a documentation bug.

**Everything is private.** Nothing becomes public before the Publish Gate, which is a separate
decision after v1.0.

## Account names differ per service

They are not interchangeable, and assuming they are cost one debugging round already.

| Service | Username |
|---|---|
| GitHub | `m-de-graaff` |
| Hugging Face | `m-de-graaff` |
| Kaggle | `markdgraaff` |

`AxiomSettings.hf_namespace` defaults to `m-de-graaff` and is correct for Hugging Face. The Kaggle
kernel id in `remote/kaggle/loop_test/kernel-metadata.json` uses `markdgraaff`.

## Status legend

- **Live** — created and in use.
- **Pending** — this version needs it, and it has not been created yet.
- **Planned** — a later version creates it.

## GitHub (`m-de-graaff`)

| Repo | Visibility | Created in | Status | Purpose |
|---|---|---|---|---|
| `axiom` | **Public** (2026-08-21, ADR-0017) | v0.0 | **Live** | The monorepo. Cloud jobs `pip install` it over a read-only fine-grained PAT — still used, though a public repo no longer needs one. |

Default branch `main`, trunk-based, no branch protection (solo repo). CI runs on every push and
pull request to `main`; first run was green across lint, types, and Python 3.11/3.12/3.13.

**Fallback if GitHub is ever unavailable or unwanted:** `uv build` a wheel and attach it as a
private Kaggle Dataset instead of installing from git. Documented as plan B per ADR-0008; not
implemented.

## Hugging Face (`m-de-graaff`)

| Repo | Type | Visibility | Created in | Status | Holds |
|---|---|---|---|---|---|
| `axiom-runs` | dataset | Private | v0.0 | **Live** (2026-08-20) | Checkpoints, run manifests, `latest.json` resume pointers |
| `axiom-trackio` | space | Private | v0.0 (optional) | Pending | trackio dashboard sync. trackio may auto-create a backing dataset; if it does, add it to this table. |
| `axiom-raw` | dataset | Private | v0.1 | **Live** (2026-08-20) | Cleaned-source Parquet plus provenance manifests |
| `axiom-tokenized` | dataset | Private → public at Publish Gate | v0.6 | Planned | Pre-tokenized MDS shards |
| `axiom-model` | model | Private → public at Publish Gate | v0.9 | Planned | `model.safetensors`, `config.json`, model card |

### Creating `axiom-raw` took two token fixes

Resolved 2026-08-20. Recorded because the next private dataset — `axiom-tokenized` in v0.6 —
will hit both of these, and knowing that in advance is worth a paragraph.

Two separate limits of the same fine-grained `axiom-write` token, found in this order:

1. `create_repo` returns `403 Forbidden: You don't have the rights to create a dataset under the
   namespace "m-de-graaff"` — checked from the laptop and from an Actions runner, same token,
   same refusal. Worked around by creating the dataset by hand.
2. `repo_info` on the new dataset returns 404 while the same call on `axiom-runs` succeeds. On a
   private repo a 404 means "invisible to this token": the token is pinned to an explicit repo
   list fixed when it was minted, and `axiom-raw` is not on it.

The fix for the second was to add `m-de-graaff/axiom-raw` with **Write** to the token's
**Repository permissions** at <https://huggingface.co/settings/tokens>. The `AXIOM_HF_TOKEN`
repository secret is the same token, so Actions picked the change up with nothing to re-paste.

**For the next private dataset**, do it in this order and neither problem appears:

1. Create the dataset at <https://huggingface.co/new-dataset> — **Private**, owner `m-de-graaff`.
2. Add it to the `axiom-write` token's repository permissions with **Write**.
3. Run the bootstrap job, which seeds the front page from `remote/hf/{name}-README.md` — the
   versioned source of truth for it — and asserts the dataset is private before it exits. It is
   idempotent, so it is safe to run either way.

Both steps are deliberately human. Widening what an API token may reach is not something an agent
should do on the account's behalf, whatever the account holder says in the moment.

## Execution backends

| Service | What it is | Created in | Status | Notes |
|---|---|---|---|---|
| Kaggle | Execution backend #1 (GPU from v0.5); **the only backend Dukascopy will answer** (ADR-0015) | v0.0 | **Live** — phone-verified 2026-08-20 | v0.0 uses CPU kernels only. Secrets `GH_PAT` and `HF_TOKEN` are attached in the editor and are destroyed by every `kernels push`, so dispatch is two steps — see `RUNBOOK.md`. |
| GitHub Actions | Execution backend for the loop and for every pull except Dukascopy (ADR-0009, ADR-0013) | v0.0 | **Live** | `.github/workflows/loop.yml`, dispatched by hand, never on push. Reads `AXIOM_HF_TOKEN` from repository secrets. Needs no GitHub PAT: the job is already inside the repo. |
| Modal | Execution backend #2 per the roadmap | v0.0 | **Blocked** — account review gate; superseded by ADR-0009 for v0.0 and ADR-0013 for v0.1 | Free Starter plan, $30/month credits. Secrets: `axiom-gh`, `axiom-hf`. `remote/modal/loop_test.py` is written and unrun; it works when the gate clears. No Modal pull job is written until there is a Modal account to run it on. |
| GCP + TRC | Stretch TPU track | ≥ v0.6 | Not started | Only if pursuing the 102 M model. Needs billing. Apply about two weeks before the intended scale-up. Gated at G3/G4 per ADR-0004. |

## Deliberately not created

Recorded so a future session does not wonder whether they were forgotten:

- No PyPI project, and no name reservation (ADR-0001).
- No public GitHub repo, no public Hugging Face repo.
- No Cloudflare R2 bucket. It is an optional hot-shard cache from v0.6 at the earliest.
- No GCP project or TRC application.

## Kernel and job identifiers

| Backend | Identifier | Defined in |
|---|---|---|
| Kaggle kernel | `markdgraaff/axiom-loop-test` | `remote/kaggle/loop_test/kernel-metadata.json` |
| GitHub Actions workflow | `loop.yml` | `.github/workflows/loop.yml` |
| GitHub Actions workflow | `universe.yml` | `.github/workflows/universe.yml` |
| GitHub Actions workflow | `pull.yml` | `.github/workflows/pull.yml` |
| Modal app | `axiom-loop` | `remote/modal/loop_test.py` |
| trackio project | `axiom` | `src/axiom/ops/logx.py` |

## Checkpoint layout inside `axiom-runs`

```
loop-test/{run_id}/latest.json          {step, path_in_repo, sha256}
loop-test/{run_id}/step_00000200/state.pt
loop-test/{run_id}/step_00000200/meta.json
```

`loop-test/` is a prefix so v0.1 onward can share the repo without colliding with v0.0's drills.

## v0.2 created no new online infrastructure

Worth stating outright, because "the corpus grew from one source to four" sounds like it should
have. It did not. Every v0.2 artifact lands in the `axiom-raw` dataset v0.1 already created; no
new repo, no new service, no new account. The table above gains no row.

The one thing v0.2 changed about the estate is *which backend runs which job*, and that was
forced by measurement rather than chosen — see the execution-backend notes below and ADR-0015.

## Layout inside `axiom-raw`

```
raw/binance/{spot|um}/{1h|1d}/{SYMBOL}.parquet          # v0.1
raw/dukascopy/{fx|commodity}/{1h|1d}/{SYMBOL}.parquet   # v0.2
raw/stooq/us/1d/{A-Z0-9_}/{TICKER}.parquet              # v0.2, letter-sharded
raw/yahoo/adjustments/{A-Z0-9_}/{TICKER}.parquet        # v0.2, events not bars
registry/registry.parquet                               # v0.2
registry/summary.md
registry/bad_sidecars.json                              # only when something would not parse
staging/stooq/                                          # transient, see below
manifests/pulls/{pull_run_id}.json
```

Every `.parquet` has a `.parquet.manifest.json` sidecar beside it. The sidecars are the pull's
only resume state (ADR-0010) and the registry is built over them rather than instead of them.

**Letter sharding** applies to the two sources with thousands of series. The Hub degrades past
roughly 10 000 files in one folder and the equities tier is 12–18 k series at two files each, so
the first character of the ticker becomes a directory; anything not alphanumeric goes to `_`
(ADR-0016). Binance and Dukascopy stay flat — a few hundred files between them is nowhere near
the limit, and sharding them would churn every existing path to solve a problem they do not have.

**`raw/yahoo/adjustments/` is not bars.** Rows are `(ts, event_type, value)` with
`event_type ∈ {split, dividend}`, and the sidecars carry `frequency = "events"` precisely so
nothing downstream tries to index them on a bar grid. Its `redistribution_class` is
`loader_only_private`, the strictest class in `docs/DATA_LICENSING.md`.

**`staging/` is transient and has a pruning rule.** It exists only for ADR-0016's single
sanctioned exception: a Stooq archive that had to transit the laptop because its handed-over URL
was bound to the IP that solved the CAPTCHA. When it is used, the run manifest carries
`staging_exception_used = true`, and the directory is **pruned as soon as the parse succeeds**.
A `staging/` that still has contents after a successful pull is a bug, not a cache.
