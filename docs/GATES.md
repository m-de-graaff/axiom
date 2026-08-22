# Gate records

One section per gate, written when the gate is passed. Each claim names the evidence, so a number
in the eventual model card can be traced back to the run that produced it.

## G2 — v0.4 "Contract" — passed 2026-08-22

**The gate:** roadmap §4/v0.4 — golden vectors + property tests green in CI; contract frozen at
`schema_version = 1`; a causality audit that fails loudly if any future-window statistic sneaks in.
**No tokenizer work may begin before G2.**

### Checklist

| Item | Status | Evidence |
|---|---|---|
| Golden vectors + full property battery green across the 3.11/3.12/3.13 matrix | Pass | 6 fixtures × 2 specs; 637 tests total, 155 of them contract tests |
| Golden vectors verified by a second, independent implementation | Pass | `reference_features` in `tests/test_contract_golden.py` — plain Python from the ADR-0020 formulas, `statistics.median`, one bar at a time |
| `@causality` marker wired | Pass | Registered in `pyproject.toml`; CI runs `pytest -m causality` as its own step so a marker that stops matching is a failure, not a silent pass |
| Prefix-consistency and perturbation-invariance pass on synthetic data | Pass | `test_a_prefix_transforms_to_a_prefix_of_the_transform`, `test_perturbing_a_bar_leaves_every_earlier_row_bit_identical`, both specs, windows 4 and 256 |
| Prefix-consistency passes on real segments, cloud-side | Pass | **300/300** split points, 50 random segments × 3 splits × 2 specs, bit-exact (run 32539968069) |
| A deliberately introduced leak fails the audit | Pass | Two permanent tests rather than one manual drill: `test_a_forward_looking_median_window_fails_prefix_consistency` and `test_the_leaky_spec_fails_prefix_consistency_by_construction` |
| Both specs frozen at `schema_version = 1` | Pass | `geo-v1` `bdb9bee03866`, `ret-v1` `d9bf2231a5e9`; a spec declaring any other version is refused at load |
| Constants committed with a firewall-respecting generation manifest | Pass | `contract_constants_v1.yaml`, hash `e3929409d26e`, `firewall_respected: true` over 31,102,283 bars; a manifest saying otherwise does not load |
| `firewall.yaml` hash-recorded in ADR-0021 | Pass | `94dd8b5072b01f746c03537450b6559180f21e87e3031fe22daad6c04719e871`, asserted by `test_the_firewall_config_hashes_to_what_adr_0021_records` |
| Regression snapshot hashes committed, 5 series × 2 specs | Pass | `tests/snapshots/contract_v1.json` — BTCUSDT-1h, EURUSD-1h, AAPL-1d, XAUUSD-1d, ETHUSDT-1d |
| Clip rates within bounds or investigated in writing | Pass | 36 combinations between 0.513 % and 0.896 %, one written investigation covering all of them; the first fit's 76 combinations up to 19.2 % were a real defect and were fixed |
| Feature-distribution report available for v0.5 | Pass | `docs/reports/v0.4-contract-qa.md`, per (spec × class × frequency × feature) quantiles |
| Usable-windows table corrected for the anchor-bar rule | Pass | **27,492,792**, superseding v0.3's 27,508,145 — exactly one window per segment |
| Public API surface = exactly four functions | Pass | `test_the_package_exports_exactly_the_four_contract_functions`, plus a namespace test |
| Leaky spec provably rejected by production paths | Pass | `transform` refuses it without `allow_leaky=True`; `inverse` refuses it outright; both tested |
| Contract is pure — no import-time network or IO | Pass | AST pass over every top-level import in `axiom/contract/`, driver and config loader exempt and named |
| **No tokenizer or quantizer code exists in the repo** | Pass | `grep -rniE 'quantiz|bsq|fsq|codebook|tokenizer' src/` returns eight hits, every one of them a docstring naming a future version |
| Zero market-data bytes on laptop or home PC | Pass | Every corpus pass on a GitHub runner; the laptop read the segment index, the constants YAML and the report, all metadata |
| Zero Kaggle GPU-hours; Modal spend < $5 | Pass | Kaggle never ran; Modal still behind the ADR-0009 gate, $0. GitHub Actions is unlimited on a public repo (ADR-0017) |
| Nothing public | Pass | Source repo was already public; no data or model artifact was published |

### The numbers

| | |
|---|---|
| Segment-passes | 55,786 (27,893 segments × 2 specs) |
| Bars streamed | 77,877,850 |
| Feature rows produced | 77,822,064 |
| NaN or Inf in output | **0** |
| Prefix-consistency on real bars | **300/300** |
| Context-512 windows | 27,492,792 |
| Pre-firewall bars fitted | 31,102,283 across 23,758 segments |
| Throughput | 6,950 bars/sec/core |
| `schema_version` | 1 |
| Constants hash | `e3929409d26e` |
| Firewall | 2025-01-01T00:00:00Z, 19.6 months of post-firewall history |
| Wall time | ~22 min fit, ~25 min dryrun |
| Cost | $0 |

### Deviations from the v0.4 plan

1. **ADRs are 0020 and 0021**, not 0017/0018 — those numbers were taken by v0.2 and v0.3.
2. **Configs live in `src/axiom/configs/`**, where every other config lives; there is no top-level
   `configs/`. Same deviation v0.3 recorded.
3. **The rolling median is numpy, not `bottleneck`.** The plan named a fallback ladder with
   `bottleneck` at the top. The measured cost of the numpy path — 6,950 bars/sec/core, both corpus
   passes inside an hour — did not justify a dependency. `bottleneck` stays the documented upgrade
   path and ADR-0020 carries the arithmetic to re-check it against at corpus M1.
4. **"Non-contiguous `ts`" is read as "strictly increasing".** The plan asks the contract to reject
   a segment whose timestamps skip a grid step. Every FX segment in the corpus skips weekends, and
   the clean layer already adjudicated which absences are boundaries (ADR-0018), so the literal
   reading would reject a quarter of the corpus for being correct.
5. **`scale` is `max(IQR/1.349, (q99 − q1)/4.6527)`, not IQR/1.349.** Forced by measurement: the
   first fit clipped 19.2 % of the FX daily gap feature because IQR was measuring a spike at zero.
   The Phase E gate asks for exactly this investigation, and this is its outcome.
6. **The live-fire drill is two permanent tests rather than a one-off.** The plan asks for an
   injected leak, once, referenced in a PR. Freezing it as a test costs the same and cannot be
   forgotten — and the exercise found something the plan's version would have missed: a median
   window that includes bar `t` itself is still prefix-consistent, so the drill has to inject a
   *forward-looking* window. Same-bar self-normalization is caught by a separate boundary test.
7. **No `api.py`.** The plan named a module holding the four-function surface; `__init__.py` is
   that module. A second file re-exporting from a third would be the indirection the
   single-implementation rule exists to prevent.

### What G2 hands to v0.5

The wick features are the finding. `upper` is never negative and `lower` never positive, so
scaling a half-distribution by a symmetric spread puts the whole short side inside half a unit and
spends the entire clip budget on one tail — which is where 36 of the residual clip flags come from.
A symmetric quantizer range would spend most of its levels on a side of the axis the feature cannot
reach. Quantizer range selection is a v0.4 non-goal by design; this is the measurement it was
deferred to.

The geo-v1 against ret-v1 comparison separates the two by nothing: clip rates agree within 0.11
percentage points in every slice. That is the intended result — the A/B is decided by tokenizer
reconstruction, not by which distribution reads more tidily.

---

## v0.3 "Clean" — exit checklist passed 2026-08-21

**The gate:** roadmap §4/v0.3 — all cleaning tests green; per-rule drop statistics reported per
source and frequency, and eyeballed sane.

### Checklist

| Item | Status | Evidence |
|---|---|---|
| Full edge-case suite green: splits, gaps, flash crash, limit-lock, DST, holidays, rollover, delisting, suspension, min-length | Pass | `tests/test_clean_engine.py`, 473 tests total across the suite |
| Hypothesis invariants: non-overlap, full accounting, determinism, idempotence, config-hash sensitivity | Pass | `test_invariants_hold_over_random_compositions`, `test_cleaning_is_deterministic`, `test_cleaning_is_idempotent` |
| Order-of-operations locked | Pass | `test_stage_order_changes_the_answer` — 11 dead bars split by an outage clean differently under either ordering |
| Engine pure; no import-time network or IO | Pass | AST pass over every top-level import in `axiom/clean/`, driver exempt and named |
| Toolkit independent of the engine | Pass | `test_synthetic_toolkit_shares_no_code_with_the_engine`, by parsing imports |
| Segment index covers every bar artifact in the registry | Pass | 13,077 of 13,077; registry holds 13,580 rows, of which 503 are Yahoo event series and not bars |
| Every segment bound to a `raw_artifact_sha256` | Pass | Column is non-nullable in `SEGMENTS_SCHEMA`; verified per artifact in `test_a_full_run_produces_three_files_that_read_back` |
| Invariants verified corpus-wide, not only in tests | Pass | `verify_corpus_invariants` runs inside `write_outputs`, so no path can upload an index that overlaps itself. It fired for real: 32 duplicate segment ids, refused |
| **Determinism at corpus scale** | Pass | Two independent full runs, `segments_hash` **474ebff75c51** both times (runs 32509742564 and 32513845062) |
| Drop-stats report committed; every red flag investigated | Pass | `docs/reports/v0.3-clean-qa.md`; 3 hits, one written investigation each |
| Top-20 most-cut inspected with a verdict each | Pass | All 20 are thin US equities; verdicts in the report |
| Usable-bars / usable-windows tables exist | Pass | **27,508,145** context-512 windows from 38,758,930 usable bars |
| TR tier built per the real verdict; arithmetic tested; coverage documented | Pass | 12,425/12,425 `tr_available` (100 %), 491 with captured corporate actions, 0 failed |
| `LIMITATIONS.md` started with quantified biases | Pass | Survivorship measured: zero of 12,425 tickers stopped trading more than a year before the pull |
| Staleness guard works | Pass | `test_a_changed_raw_file_is_detected_and_recleaned`; the config-hash refusal fired for real when the session filter changed the hash |
| RUNBOOK rules written | Pass | `docs/RUNBOOK.md` §"The v0.3 cleaning pass" |
| Zero market-data bytes on laptop or home PC | Pass | Every corpus run on a GitHub runner; the laptop read only the registry, the segment index and the drop stats, all of which are metadata |
| Zero Kaggle GPU-hours; Modal spend < $5 | Pass | Kaggle never ran; Modal still behind the ADR-0009 gate, $0 |
| Nothing public | Pass | `axiom-raw` private; `clean/` and `derived/` are inside it |

### The numbers

| | |
|---|---|
| Series cleaned | 13,077 |
| Segments | 27,905 |
| Bars in | 42,308,244 |
| Bars kept | 38,758,930 (**91.61 %**) |
| Context-512 windows | 27,508,145 |
| Failures | 0 |
| `clean_config_hash` | `98d62d99d8f6` |
| `segments_hash` | `474ebff75c51` |
| Wall time | ~26 min clean, ~20 min snapshot |

### Deviations from the v0.3 plan

1. **ADRs are 0018 and 0019**, not 0015/0016 — those numbers were taken by v0.2.
2. **The clean config lives in `src/axiom/configs/`**, where every other config lives; there is no
   top-level `configs/`.
3. **The TR tier is a coverage manifest, not Parquet.** The audit measured
   `split_and_dividend_adjusted`, so `tr_close` is an identity for every source and the planned
   letter-sharded tier would have been twelve thousand copies of a column already in the file
   beside it — the duplication the plan itself refuses for crypto. Both branches are implemented
   and tested; a re-audit flips it without a rewrite.
4. **A sixth stage.** The plan named five. `session_filter` was added after the first full corpus
   run measured what happens without it: Dukascopy's synthetic weekend padding was excised as an
   illiquid run, partitioning every padded weekend and costing EURUSD 20.8 % of its history. See
   ADR-0018 *Amendments*.
5. **A new session, `24x5-cfd`.** v0.2 stamped every Dukascopy artifact `24x5`; the commodity CFDs
   have a daily settlement break, and undeclared it cost 100 % of Brent, copper, natural gas and
   silver. Fixed with a config override rather than a re-pull.
6. **Segment ids carry source and market.** The plan's `symbol:frequency:start_ts` collided 32
   times — Binance lists the same ticker on spot and USDT-M futures, both at 1d, starting the same
   day.
7. **Two of the three red-flag checks were rewritten** to measure what they meant. See the report.
8. **Modal is still not the backend.** GitHub Actions ran every corpus job; `remote/modal/clean_run.py`
   exists and runs the identical CLI when the ADR-0009 account gate clears.

### Follow-up, 2026-08-22 (v0.3.1)

The one item the gate recorded as a deviation — Stooq sidecars still reading
`vendor_adjusted_unverified` — is closed. `axiom raw stamp-verdict` wrote the measured verdict
into 12,425 sidecars with 0 failures, in a field held outside the identity hash, so no Parquet
moved and this gate's evidence still stands unchanged. `segments_hash` is still `474ebff75c51`.

Doing it exposed three more copies of the per-file Hub download and a registry that had silently
published 19 artifacts short. Both fixed; the registry rebuilt clean at 13,580.

### What the corpus run found that no test would have

Every one of the five bugs fixed in v0.3 came from a real run, not from a test written in advance:
duplicate segment ids, the undeclared CFD break, the weekend-padding partition, two red-flag
checks measuring the wrong thing, and the Hub rate-limiting a 26,000-request burst. The tests
came after, and each one now fails without its fix.

The invariant check is the reason none of them shipped. It refused to upload a segment index with
duplicate keys, which is what turned a silent corruption into a failed job.

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
