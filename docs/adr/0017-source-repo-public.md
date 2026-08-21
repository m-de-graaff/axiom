# ADR-0017: The source repo goes public; the data stays private

**Status:** Accepted (v0.2, 2026-08-21). Supersedes the GitHub half of ADR-0001.

## Context

ADR-0001 said nothing goes public before the Publish Gate: not PyPI, not GitHub, not Hugging Face.
It bundled three separate things — a distribution name, source code, and trained artifacts —
under one rule, and the rule was written before any of them existed.

v0.2 ran into the constraint from an unexpected direction. GitHub gives a **private** repository
2 000 Actions minutes a month and a **public** one unlimited minutes on standard runners. The
equities pull alone spent close to six hours of runner time before its commit batching was fixed,
and by the time the corpus was three sources deep the month's allowance was gone: every workflow
began failing in four to ten seconds with no log written at all, which is what an exhausted quota
looks like from the outside.

The alternatives were real and all worked: run the remaining CLI steps from a Kaggle kernel, buy
minutes at $0.008 each, or wait for the billing cycle. Going public was chosen deliberately over
those, not as the only way out.

## Decision

**`m-de-graaff/axiom` is public as of 2026-08-21.** The rest of ADR-0001 is untouched.

| Thing | Before | Now |
|---|---|---|
| Source repo | Private | **Public** |
| PyPI distribution | Nothing before the gate | Unchanged — nothing before the gate |
| `axiom-raw` | Private | Private, **permanently** — not "until the gate" |
| `axiom-tokenized`, `axiom-model` | Private until the gate | Unchanged |

The code was always publishable. `docs/DATA_LICENSING.md` classifies every source as
`loader_manifest_private_cache` or `loader_only_private`, and both classes permit publishing the
loader — what they forbid is republishing the vendors' bars. Nothing in this repository is a bar.

### What this exposes, deliberately

**Action run logs become world-readable.** Ours carry the Stooq CAPTCHA-token URLs that were
passed as workflow inputs. They expire in minutes and were already dead when this was written, so
the exposure is real but worthless. It is named here rather than discovered later, and it is the
reason those URLs are dispatch inputs rather than repository secrets: a secret would have outlived
its usefulness by months.

**Git history becomes world-readable.** Checked before the switch: no `.env`, `kaggle.json`, key
or credential file appears anywhere in the history, and `.gitignore` has covered all of them since
v0.0.

**Fork pull requests can run CI.** CI uses no secrets by construction — it says so in its own
`env` block — and GitHub does not expose repository secrets to workflows from forks. Every
workflow that does use a secret is `workflow_dispatch` only, which requires write access.

## Consequences

The v1.0 line "still private" in the roadmap's version ladder now means the *model*, not the
repository. The Publish Gate keeps its job: it decides the distribution name and flips
`axiom-tokenized` and `axiom-model`. It no longer decides anything about source visibility,
because that decision has been made.

Being public early has a cost ADR-0001 was right to name: the work is visible while it is still
wrong. This session alone shipped a weekend-window rule that rejected 24 of 27 FX series and a
commit batch size that could not survive its own corpus. Both are in the history, with the
reasoning that produced them and the measurements that corrected them. That is an acceptable
trade for a research repo whose value is meant to be its reproducibility, but it is a trade and
not a free win.

The unlimited-minutes benefit applies to standard runners only. Nothing here uses larger runners,
and if that ever changes the cost model changes with it.
