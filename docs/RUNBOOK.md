# Runbook

Operational procedures: tokens, dispatch, and what to do when a cloud session dies.

## Token inventory

| Token | Scope | Where it lives | Expiry | Rotation |
|---|---|---|---|---|
| GitHub fine-grained PAT `axiom-kaggle-read` | Repository access: only `m-de-graaff/axiom`. Permissions: Contents read-only, Metadata read-only | Laptop `.env` as `AXIOM_GH_PAT`; Kaggle secret `GH_PAT`; Modal secret `axiom-gh` | **2026-11-18** | Regenerate in GitHub, update `.env` and both cloud secrets, revoke the old one |
| Hugging Face fine-grained token `axiom-write` | Read + write on `datasets/m-de-graaff/axiom-runs` only. Every user-level and org-level permission left unchecked | Laptop `.env` as `AXIOM_HF_TOKEN`; Kaggle secret `HF_TOKEN`; Modal secret `axiom-hf` | 90 days from 2026-08-20 | Create the new token first, update all three consumers, then revoke |
| Kaggle API token (`kaggle.json`) | Full account API access | Laptop only, `~/.kaggle/kaggle.json`, chmod 600 | No expiry | Expire from Kaggle account settings, download a fresh one |
| Modal token | Full workspace access | Laptop only, via `modal token new` | No expiry | `modal token new` reissues; revoke the old one in the Modal dashboard |

### Creating the GitHub PAT

There is no API for this. GitHub removed the Authorizations API in 2020, and fine-grained tokens
are browser-only, so neither `gh` nor a script can mint one — the steps below are manual by
necessity, not by preference.

1. Go to Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate.
2. Name `axiom-kaggle-read`, resource owner `m-de-graaff`, expiry 90 days.
3. Repository access: **Only select repositories** → `axiom`.
4. Repository permissions: **Contents: Read-only**. Nothing else. Metadata is added automatically
   and is fine.
5. Copy it into the password manager, then into the Kaggle and Modal secret stores. It is shown
   once.

A token with write access, or with access to all repositories, is a compromise of every repo on
the account rather than of one drill kernel. The scoping is the point.

### The rules

**Tokens never appear in code, configs, notebook cell outputs, or git history.** The laptop reads
them from a gitignored `.env`; cloud kernels read them from the platform's secret store. The
Kaggle kernel builds its install URL as a list element passed to `subprocess`, so the PAT never
reaches a shell history or a printed command line.

**A token that was ever committed is compromised**, even after a history rewrite — assume it was
cloned and scraped. Rotate it; deleting it is not enough. The gitleaks pre-commit hook is what
stops this from happening in the first place; run `pre-commit install` once per clone.

**Revoking fast:** GitHub at Settings → Developer settings → Personal access tokens → Fine-grained
→ Delete. Hugging Face at Settings → Access Tokens → Invalidate. Both take effect immediately.
Kaggle and Modal tokens are revoked from their respective account pages.

## Local setup

```sh
uv sync --all-extras
uv run pre-commit install
```

`.env` (gitignored, never committed) holds two values:

```
AXIOM_HF_TOKEN=hf_...          # the axiom-write token
AXIOM_GH_PAT=github_pat_...    # the axiom-kaggle-read PAT, for pasting into cloud secret stores
```

Only `AXIOM_HF_TOKEN` is read by the package. `AXIOM_GH_PAT` is kept here purely so the value is
recoverable when a cloud secret store needs it — the laptop never uses it to clone, because `gh`
already has its own credentials.

Verify both:

```sh
uv run python -c "from huggingface_hub import HfApi; from axiom.config.settings import AxiomSettings; s=AxiomSettings(); print(HfApi(token=s.hf_token.get_secret_value()).whoami()['name'])"
```

Recorded toolchain versions on the laptop as of v0.0: git 2.55.0, uv 0.9.26, gh 2.97.0,
just 1.57.0, torch 2.13.0+cpu.

## Quality gates

`just check` runs exactly what CI runs, in the same order: `lint`, then `type`, then `test`. If it
is green locally it is green in CI, unless a Python version in the matrix disagrees.

CI finishes in about a minute on a warm uv cache, well inside the ten-minute budget. Watch a run
with `gh run watch <id> --exit-status`, or list them with `gh run list`.

The slow drills are marked: `uv run pytest -m slow` runs only the kill-and-resume tests,
`uv run pytest -m "not slow"` skips them.

## Dispatch

| Command | What it does |
|---|---|
| `just loop-local` | Runs the loop on the laptop with no network at all |
| `just loop-verify` | The determinism drill: clean run vs killed-and-resumed run, compared exactly. Exits non-zero on any drift |
| `just loop-hub` | Runs locally but pushes to `axiom-runs`, so there is a checkpoint tree to inspect |
| `just loop-kaggle` | Pushes and starts the Kaggle CPU kernel |
| `just loop-kaggle-status` | Polls the kernel's state |
| `just loop-kaggle-log` | Downloads the kernel's output log |
| `just loop-modal` | Runs the Modal CPU job |

### Before the first Kaggle dispatch

**The Kaggle username is `markdgraaff`, not `m-de-graaff`.** GitHub and Hugging Face both use
`m-de-graaff`; Kaggle does not. The kernel id in `kernel-metadata.json` reflects this.

1. **Phone-verify the account.** As of 2026-08-20 it is not verified, and without it Kaggle offers
   neither internet-enabled kernels nor user secrets — so every step below is blocked. Settings →
   Account → Phone verify. This needs a real phone and an SMS code, so it cannot be automated.
2. Settings → API Tokens → Generate New Token, saved to `~/.kaggle/kaggle.json`, chmod 600.
3. Add both secrets under Add-ons → Secrets in the kernel editor: `GH_PAT` (the GitHub PAT) and
   `HF_TOKEN` (the Hugging Face token). Both values are in the laptop's `.env`. Attach both to
   the kernel, not just create them — an unattached secret is invisible to the run.
4. `uv run kaggle kernels push -p remote/kaggle/loop_test` (or `just loop-kaggle`).

A run whose secrets are missing or unattached fails like this, about a second in:

```
client.get_secret("GH_PAT")
urllib.error.HTTPError: HTTP Error 400: Bad Request
ConnectionError: Connection error trying to communicate with service.
```

The `ConnectionError` is misleading — the network is fine, the secret is simply not attached.

The kernel prints its Python and torch versions on startup. **Record them in this file and amend
ADR-0007 if Kaggle's Python is below the 3.11 floor.**

Kaggle image versions observed (2026-08-20, first dispatch): **Python 3.12**, read from the
interpreter paths in the kernel log. That is above the `>=3.11` floor, so ADR-0007 stands and
needs no amendment. The torch version is still unrecorded — the first run died before
`report_image()`, which is why that call now happens before secrets are read.

### Before the first Modal dispatch

```sh
uv run modal token new
uv run modal secret create axiom-gh GH_PAT=github_pat_...
uv run modal secret create axiom-hf HF_TOKEN=hf_...
```

Expected cost for the v0.0 drill is cents against the $30 monthly credit. Check the actual figure
in the Modal dashboard after the first run and note it here.

Modal spend for v0.0: _not yet recorded._

## The kill-and-resume procedure

This is the loop v0.0 exists to prove, and it is the procedure every later version depends on.

**Locally**, `just loop-verify` does the whole drill in one command and exits non-zero on drift.
The same drill runs in CI as `tests/test_loop_determinism.py`.

**On Kaggle:**

1. `just loop-kaggle` and wait until the log shows the run past step 600.
2. Cancel the kernel from the Kaggle UI. This is a real SIGKILL, not fault injection.
3. `just loop-kaggle` again. The log must show `resumed loop-test-kaggle-001 from step N`, where
   N is the last multiple of `save_every` before the cancel.
4. Let it finish. The final `acc` must equal a local run of the same config:
   `uv run axiom loop run --config loop_test --run-id x --total-steps 2000 --save-every 200 --no-push`
5. Confirm on Hugging Face that `loop-test/loop-test-kaggle-001/latest.json` and the step
   directories exist in `axiom-runs`.

Nothing on Kaggle's side other than the log matters. The kernel's filesystem is disposable by
design; the checkpoint on the Hub is the only durable state.

## When a Kaggle session dies

Sessions die on the 12-hour cap, on quota exhaustion, and occasionally for no stated reason. None
of these need investigation — they are the expected case, which is why the resume path exists.

1. Check the last `latest.json` in `axiom-runs` for the step it reached.
2. Re-push the kernel. It resumes from that step with no arguments changed.
3. If the resume refuses with a config-hash mismatch, the config changed between the checkpoint
   and the resume. That is the guard working: either restore the old config, or start a new
   `run_id`. Do not delete the guard.

A run that resumes but produces a different final number is a real bug, not a session problem.
That is `systematic-debugging` territory, and the first thing to check is whether RNG state
survived the round trip.

## Deferred to later versions

- A CPU tiny-model smoke-train job is added to the CI workflow at v0.7.
- GPU quota tracking starts at v0.5, when the first GPU minutes are spent.
