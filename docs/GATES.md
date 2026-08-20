# Gate records

One section per gate, written when the gate is passed. Each claim names the evidence, so a number
in the eventual model card can be traced back to the run that produced it.

## v0.1 "Schema & First Bars" — exit checklist passed 2026-08-20

**The gate:** roadmap §4/v0.1 — at least 100 liquid Binance pairs at both 1h and 1d in a private
`axiom-raw` with manifests, a re-pull that is byte-identical or produces a documented manifest
diff, and zero bytes on the laptop.

### Checklist

| Item | Status | Evidence |
|---|---|---|
| ≥ 100 pairs at **both** 1h and 1d with ≥ 365 days of history | Pass | 225 distinct symbols (200 spot, 100 um, 75 in both markets) |
| `universe_v1.yaml` committed with a hash; every manifest references it | Pass | `universe_hash=2de32d7d4f27` in the config and in sampled sidecars across both markets |
| Re-pull sample byte-identical on monthly content | Pass | `axiom raw verify --sample 10 --seed 1337`: 10/10 |
| Daily-tail divergence reported as a manifest diff, not a failure | Pass | `VerifyResult.status="drifted"`; no sampled series had drifted at verify time |
| Pull kill drill passed, resume via sidecars | Pass | `gh run cancel` at 30 built / 29 committed; relaunch `ok=131 skipped=29 failed=0` |
| All loader/schema/manifest tests green in CI; no live network in CI | Pass | 188 tests, lint and types green across Python 3.11–3.13 |
| Cross-check against an independent implementation recorded | Pass | `binance_historical_data`, 3/3 agree on rows and on every OHLCV value of 2024-02-14 |
| QA report committed; invariant violations = 0; storage < 2 GB | Pass | `docs/reports/v0.1-raw-qa.md`; 0 violations by construction; 0.57 GiB |
| Zero market-data bytes on laptop or home PC | Pass | No Parquet outside the runners; `.artifacts/` holds three markdown files |
| Zero Kaggle GPU-hours; Modal spend < $5 | Pass | Kaggle never ran; Modal has no account, $0 |
| `docs/REPOS.md` lists `axiom-raw` (private); nothing public | Pass | Created private 2026-08-20, `repo_info` reports `private=True` |

### The numbers

| Run | Result |
|---|---|
| Smoke (`--symbols BTCUSDT,ETHUSDT --markets spot`) | `ok=4 skipped=0 failed=0`, 164,238 bars |
| Kill drill, killed | 30 series built, 29 committed, cancelled at 7m01s |
| Kill drill, relaunched | `ok=131 skipped=29 failed=0`, 2,517,297 bars |
| Full pull | `ok=440 skipped=160 failed=0`, 7,447,699 bars, 406,897,096 bytes |
| Corpus total | 600 series, 10,885,159 bars, 0.57 GiB |

### Deviations from the v0.1 plan

1. **The pull runs on GitHub Actions, not Modal** (ADR-0013). Modal's account is still behind the
   review gate that ADR-0009 recorded at v0.0, so there is no Modal token to run anything with.
   The vendor-independence half of backend #2 is still undelivered and is deferred again to v0.6.
2. **ADRs are numbered 0010–0012**, not 0009–0011. The v0.0 backend substitution took 0009.
3. **Off-grid timestamps are a warning, not a violation.** The first run against the real bucket
   failed spot 1h BTCUSDT on 43 phase-shifted bars from an exchange restart. Rejecting them would
   have cost the corpus its most important series; snapping them to the grid would have been
   imputation. They are counted into `off_grid_count` instead (ADR-0010).
4. **The minimum-history rule is applied at selection time**, not after the pull. ADR-0011
   originally deferred it on reasoning that turned out to be wrong: the listing's earliest month
   *is* the start of the series. Applying it early also removed seven tokenized equities that had
   taken the top of the USDT-M volume ranking.
5. **`configs/universe_v1.yaml` lives at `src/axiom/configs/`**, inheriting the v0.0 deviation
   that moved configs inside the package so cloud kernels can reach them from a wheel.

### Settled by this gate

`data.binance.vision` publishes phase-shifted bars after an exchange restart, and both the
timestamp grid and the CSV header presence vary within one bucket. Detection beats assumption for
all three.

Byte-identity is conditional on the Parquet writer's version, which pyarrow stamps into every
file. The manifest's own content hash is not, which is why that is the field the Parquet metadata
carries and the field the idempotence test compares.

## G1 — v0.0 "Spine & Loop" — passed 2026-08-20

**The gate:** kill-and-resume produces a final state bit-identical to an uninterrupted run, on a
real cloud backend, with checkpoints on Hugging Face. CI green. ADRs merged. `docs/REPOS.md`
current. No market data, no GPU minutes.

### Checklist

| Item | Status | Evidence |
|---|---|---|
| CI green on `main` — lint, types, tests × 3 Python versions | Pass | Run 32366649686, all 5 jobs success |
| ADRs merged; the open design decisions closed | Pass | `docs/adr/0001`–`0009`, nine files |
| Kill→resume bit-identity proven locally | Pass | `tests/test_loop_determinism.py`, 5 tests including a two-kill case |
| Kill→resume bit-identity proven on Kaggle | Pass | Killed at step 3000 of 6000 via Stop Session, resumed, `acc=3018.7626345157623` |
| Checkpoints + `latest.json` for ≥ 3 runs in private `axiom-runs` | Pass | 5 runs, 54 checkpoints, `latest.json` on each |
| Both backends executed the identical CLI path | Pass, with deviation | Kaggle and GitHub Actions. **Not Modal** — see below |
| Zero market data touched | Pass | No loader exists; no network call outside GitHub and Hugging Face |
| Zero Kaggle GPU-hours | Pass | `enable_gpu: false` in `kernel-metadata.json`; every run CPU |
| Modal spend < $2 | Pass | $0. Modal never ran |
| No secret in git history | Pass | `gitleaks` over all files clean; pattern grep over `git log -p --all` returns 0; `.env` untracked |
| `docs/REPOS.md` documents every repo created; nothing public | Pass | GitHub `axiom`, HF `axiom-runs`, both private |

### The numbers

Every run at seed 1337, config hash `d2ff0be80933` for the 2000-step arm.

| Backend | Run | Final `acc` |
|---|---|---|
| Laptop | 2000 steps, uninterrupted | 996.4922949671745 |
| Kaggle | 2000 steps, uninterrupted | 996.4922949671745 |
| GitHub Actions | 2000 steps, uninterrupted | 996.4922949671745 |
| Laptop | 6000 steps, uninterrupted | 3018.7626345157623 |
| Kaggle | killed at 3000, resumed to 6000 | 3018.7626345157623 |
| GitHub Actions | killed at 2000, resumed to 6000 | 3018.7626345157623 |

Neither cloud kill used the `AXIOM_KILL_AT_STEP` fault injection. Kaggle was killed with Stop
Session, Actions with `gh run cancel`, which SIGKILLs the runner.

### Deviations from the v0.0 plan

Recorded because a gate passed with an asterisk is worth less than one that says where the
asterisk is.

1. **Backend #2 is GitHub Actions, not Modal** (ADR-0009). Modal's account is behind a review
   gate. Actions shares a vendor with the code host, so the vendor-independence half of what
   backend #2 was for is **not** delivered by v0.0. Deferred to v0.6 with the real vendor choice.
2. **Kaggle dispatch is two steps, not one.** Phase F5 assumed `kaggle kernels push` was the whole
   dispatch. It destroys the kernel's secret attachment, and the API has no field to declare
   secrets, so a push must be followed by re-attaching and Save Version.
3. **`configs/` moved inside the package** to `src/axiom/configs/`. Cloud kernels install a wheel
   and have no checkout, so a repo-root config could not be reached.

### Settled by this gate

Kaggle's image is Python 3.12.13 with torch 2.10.0+cpu. ADR-0007's provisional `>=3.11` floor
stands with no amendment.
