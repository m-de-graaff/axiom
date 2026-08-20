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
| `just loop-kaggle` | Uploads the kernel code. **Does not run it with secrets** — see the two-step note below |
| `just loop-kaggle-status` | Polls the kernel's state |
| `just loop-kaggle-log` | Downloads the kernel's output log |
| `just loop-github` | Dispatches backend #2 on GitHub Actions. One command, no follow-up |
| `just loop-github-watch` | Follows the newest Actions loop run to completion |
| `just loop-github-kill` | Cancels the newest run mid-flight — a real SIGKILL, for the resume drill |
| `just loop-modal` | Runs the Modal CPU job. Blocked on the account review gate; see ADR-0009 |

### v0.1 data jobs

| Command | What it does |
|---|---|
| `just universe-build 2026-07` | Ranks the bucket in the cloud and writes `universe_v1.yaml` as a workflow artifact |
| `just universe-fetch` | Downloads that artifact into `src/axiom/configs/` for review and commit |
| `just universe-show` | Prints a universe's criteria and counts, verifying its hash |
| `just pull-smoke` | Two majors, spot only, both frequencies. The first thing to run against a fresh `axiom-raw` |
| `just pull-binance` | Dispatches the full pull. Extra `-f key=value` flags pass through as workflow inputs |
| `just pull-watch` | Follows the newest pull to completion |
| `just pull-log` | Downloads the newest pull's log |
| `just pull-kill` | Cancels the newest pull mid-flight — a real SIGKILL, for the resume drill |
| `just pull-dryrun` | The whole fetch, checksum, parse and validate path against the real bucket, writing to the runner and publishing nothing |
| `just pull-local` | Pulls into `.artifacts/raw-local`. **Development only** — this writes market data to the machine it runs on, which the laptop must never do |
| `just bootstrap-raw` | Creates the private `axiom-raw` dataset and seeds its front page. Idempotent |
| `just raw-inspect SYMBOL` | Fetches one series and prints what failed validation. Writes nothing |
| `just raw-verify` | Re-derives a sample and compares the bytes |
| `just raw-stats` | Regenerates the QA report from the sidecars |

The narrowing flags are for smoke runs:

```sh
just pull-binance -f markets=spot -f symbols=BTCUSDT,ETHUSDT
just pull-binance -f limit=40
```

Both are recorded in the pull manifest, and a run that used either is marked `(PARTIAL)` in its
summary line. A partial pull must never be mistaken for a full one.

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

### `kaggle kernels push` destroys the secret attachment

Verified three times on 2026-08-20. Push the kernel, and the secrets that were attached to it are
gone; the next run dies at `get_secret` with the error above. The Kaggle API has no field for
secrets — `kernel-metadata.json` carries `enable_gpu`, `enable_internet`, `dataset_sources` and
friends, and nothing else — so there is no way to declare them and no way to re-attach them from
the CLI.

**Kaggle dispatch is therefore two steps, and the order matters:**

1. `just loop-kaggle` uploads the code. This wipes the secrets.
2. In the editor, re-attach `GH_PAT` and `HF_TOKEN`, then click **Save Version**. That runs it.

Once the code is uploaded, further runs of the same code need no push: **Save Version** alone
re-runs and resumes, and the attachment survives because nothing overwrote it. Only a code change
costs you the re-attach.

This is why `just loop-kaggle` is not sufficient on its own, and why the v0.0 plan's Phase F5 —
which assumed a push was the whole dispatch — was wrong on this point.

Backend #2 has no such problem: GitHub Actions reads `AXIOM_HF_TOKEN` from repository secrets,
which nothing resets. See ADR-0009.

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

**On GitHub Actions**, the whole drill is scriptable from the laptop:

```sh
gh workflow run loop.yml -f run_id=drill -f total_steps=6000 -f save_every=500 -f resume=false
just loop-github-kill                     # once it is past a checkpoint
just loop-github run_id=drill             # resume=true is the default
```

Recorded result, 2026-08-20: killed at step 2000 of 6000, resumed, finished at
`acc=3018.7626345157623` — identical to an uninterrupted local run of the same config.

**On Kaggle**, every step is a UI action, because a push would wipe the secrets:

1. Re-attach `GH_PAT` and `HF_TOKEN` in the editor.
2. **Save Version.** Wait until the run is past a checkpoint.
3. Cancel it from the Kaggle UI. This is a real SIGKILL, not fault injection.
4. **Save Version** again — no push in between, or the secrets go and the resume dies at
   `get_secret`. The log must show `resumed <run_id> from step N`, where N is the last multiple
   of `save_every` before the cancel.
5. The final `acc` must equal a local run of the same config:
   `uv run axiom loop run --config loop_test --run-id x --total-steps 6000 --save-every 500 --no-push`
6. Confirm the step directories and `latest.json` are in `axiom-runs`.

Nothing on Kaggle's side other than the log matters. The kernel's filesystem is disposable by
design; the checkpoint on the Hub is the only durable state.

Recorded results, 2026-08-20, all at seed 1337:

| Backend | Run | Final `acc` |
|---|---|---|
| Laptop | 2000 steps, uninterrupted | 996.4922949671745 |
| Kaggle | 2000 steps, uninterrupted | 996.4922949671745 |
| GitHub Actions | 2000 steps, uninterrupted | 996.4922949671745 |
| Laptop | 6000 steps, uninterrupted | 3018.7626345157623 |
| Kaggle | **killed at 3000**, resumed to 6000 | 3018.7626345157623 |
| GitHub Actions | **killed at 2000**, resumed to 6000 | 3018.7626345157623 |

Both kills were real: Stop Session on Kaggle, `gh run cancel` on Actions. Neither used the
`AXIOM_KILL_AT_STEP` fault injection, which exists for the local test only.

## Running a pull

A pull is dispatched, watched, and forgotten. It has no checkpoint, so there is nothing to clean
up if it dies and nothing to configure if it is restarted.

```sh
just pull-binance                 # the whole universe, both markets, 1h and 1d
just pull-watch                   # follow it
```

The job downloads roughly a gigabyte of small archives across some six hundred series and takes
on the order of an hour. It writes into `axiom-raw` in batched commits of about fifty files, so
progress is visible in the dataset's commit history while the run is still going.

### Reading a pull manifest

Every run writes `manifests/pulls/{pull_run_id}.json` into `axiom-raw`. The fields worth looking
at first:

| Field | Means |
|---|---|
| `ok` / `skipped` / `failed` | Series written, already current, and not landed |
| `limit`, `symbols_filter` | Non-empty means this was a partial pull |
| `failures[]` | One entry per failed series, with the exception that stopped it |
| `total_rows`, `total_bytes` | What this run added, not what the corpus holds |
| `universe_hash` | Which universe definition asked for this work |

`skipped` is the number that says resume worked. On a second run of the same work list with
nothing new upstream, every series should be skipped and `ok` should be zero.

A `failed` entry is not automatically a bug. A symbol in the universe that has since been
delisted, or one whose archives have a gap the bucket never filled, fails honestly and is
recorded. What matters at the exit gate is that every failure has an explanation.

### The kill-and-resume drill for pulls

Same shape as the loop drill, and simpler, because there is no state to compare — the evidence is
in the skip count.

```sh
just pull-binance -f limit=40     # start a partial pull
just pull-kill                    # once the log shows a dozen symbols done
just pull-binance -f limit=40     # relaunch, unchanged
```

The relaunch must report a non-zero `skipped` matching roughly what the first run finished, and
must finish the rest. No flag is passed to make this happen: resuming is what the pull does on
every start. `gh run cancel` SIGKILLs the runner, so this is a real interruption.

The one thing to check in the second run's log is that skipped symbols cost no archive
downloads — a skip is a listing plus a handful of tiny `.CHECKSUM` fetches, nothing more.

**Recorded result, 2026-08-20.** A `--limit 40` run was cancelled after 7 minutes with
`gh run cancel`, which SIGKILLs the runner. At that moment 30 series had been built in-process and
**29 were committed** to `axiom-raw` — the thirtieth was still in the staging directory and died
with the container, which is the intended trade: staged-but-uncommitted work is lost, committed
work survives, and nothing lands half-written.

The relaunch was byte-for-byte the same dispatch, with no resume flag, because there is no resume
flag:

```
kill-drill-resume: ok=131 skipped=29 failed=0 rows=2517297 bytes=143162567 (PARTIAL)
```

29 skipped, matching the 29 committed, and the remaining 131 finished. That is the whole
mechanism: the sidecars are the state.

### When a pull fails partway

Dispatch it again. That is the whole procedure. The finished series skip, the unfinished ones are
retried, and the second run's manifest records the split.

If the same symbol fails twice with the same error, that is a real failure worth reading. The
likely causes, in order: the symbol was delisted between the universe build and the pull, the
bucket published an archive whose checksum does not match, or two source archives disagree about
a bar at the monthly/daily seam. All three fail loudly and name the symbol.

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
