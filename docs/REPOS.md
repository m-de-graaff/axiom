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
| `axiom-raw` | dataset | Private | v0.1 | Planned | Cleaned-source Parquet plus provenance manifests |
| `axiom-tokenized` | dataset | Private → public at Publish Gate | v0.6 | Planned | Pre-tokenized MDS shards |
| `axiom-model` | model | Private → public at Publish Gate | v0.9 | Planned | `model.safetensors`, `config.json`, model card |

## Execution backends

| Service | What it is | Created in | Status | Notes |
|---|---|---|---|---|
| Kaggle | Execution backend #1 (GPU from v0.5) | v0.0 | **Live** — phone-verified 2026-08-20 | v0.0 uses CPU kernels only. Secrets `GH_PAT` and `HF_TOKEN` are attached in the editor and are destroyed by every `kernels push`, so dispatch is two steps — see `RUNBOOK.md`. |
| GitHub Actions | Execution backend #2 for v0.0 (ADR-0009) | v0.0 | **Live** | `.github/workflows/loop.yml`, dispatched by hand, never on push. Reads `AXIOM_HF_TOKEN` from repository secrets. Needs no GitHub PAT: the job is already inside the repo. |
| Modal | Execution backend #2 per the roadmap | v0.0 | **Blocked** — account review gate; superseded for v0.0 by ADR-0009 | Free Starter plan, $30/month credits. Secrets: `axiom-gh`, `axiom-hf`. `remote/modal/loop_test.py` is written and unrun; it works when the gate clears. |
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
| Modal app | `axiom-loop` | `remote/modal/loop_test.py` |
| trackio project | `axiom` | `src/axiom/ops/logx.py` |

## Checkpoint layout inside `axiom-runs`

```
loop-test/{run_id}/latest.json          {step, path_in_repo, sha256}
loop-test/{run_id}/step_00000200/state.pt
loop-test/{run_id}/step_00000200/meta.json
```

`loop-test/` is a prefix so v0.1 onward can share the repo without colliding with v0.0's drills.
