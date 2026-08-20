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
| `axiom` | Private | v0.0 | **Live** (2026-08-20) | The monorepo. Cloud jobs `pip install` it over a read-only fine-grained PAT. |

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
| `axiom-raw` | dataset | Private | v0.1 | **Blocked** — see below | Cleaned-source Parquet plus provenance manifests |
| `axiom-tokenized` | dataset | Private → public at Publish Gate | v0.6 | Planned | Pre-tokenized MDS shards |
| `axiom-model` | model | Private → public at Publish Gate | v0.9 | Planned | `model.safetensors`, `config.json`, model card |

### `axiom-raw` is blocked on a token permission

`create_repo` returns:

```
403 Forbidden: You don't have the rights to create a dataset under the namespace "m-de-graaff"
```

Checked from both sides. The laptop's `.env` token and the `AXIOM_HF_TOKEN` repository secret are
the same fine-grained `axiom-write` token, both authenticate as `m-de-graaff`, and both are
refused. It has write access to the `axiom-*` repos that already exist and no permission to
create new ones under the namespace.

**This blocks the rest of v0.1**: the smoke run, the kill drill, the full pull, `raw verify`,
`raw stats`, the QA report, and the tag all need somewhere to write.

Two ways out, either is fine and neither needs code:

1. Add **Create repos** (write access to all repos under the namespace) to the `axiom-write`
   token at <https://huggingface.co/settings/tokens>.
2. Create the dataset by hand at <https://huggingface.co/new-dataset> — name `axiom-raw`, owner
   `m-de-graaff`, **Private** — and confirm the token's repo scope covers it.

Then run `just bootstrap-raw`, which creates the dataset if it is still missing and seeds its
front page from `remote/hf/axiom-raw-README.md` — the versioned source of truth for it. The job
is idempotent, so it is safe either way, and it asserts the dataset is private before it exits.

## Execution backends

| Service | What it is | Created in | Status | Notes |
|---|---|---|---|---|
| Kaggle | Execution backend #1 (GPU from v0.5) | v0.0 | **Live** — phone-verified 2026-08-20 | v0.0 uses CPU kernels only. Secrets `GH_PAT` and `HF_TOKEN` are attached in the editor and are destroyed by every `kernels push`, so dispatch is two steps — see `RUNBOOK.md`. |
| GitHub Actions | Execution backend #2 for v0.0 (ADR-0009) | v0.0 | **Live** | `.github/workflows/loop.yml`, dispatched by hand, never on push. Reads `AXIOM_HF_TOKEN` from repository secrets. Needs no GitHub PAT: the job is already inside the repo. |
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

## Layout inside `axiom-raw`

```
raw/binance/{spot|um}/{1h|1d}/{SYMBOL}.parquet
raw/binance/{spot|um}/{1h|1d}/{SYMBOL}.parquet.manifest.json
manifests/pulls/{pull_run_id}.json
```

One Parquet file per series with its sidecar beside it, and one manifest per pull run. The
sidecars are the pull's only resume state (ADR-0010).
