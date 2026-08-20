# Gate records

One section per gate, written when the gate is passed. Each claim names the evidence, so a number
in the eventual model card can be traced back to the run that produced it.

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
