# Axiom v0.4 — "Contract" — Phased Development TODO

**Goal:** the versioned preprocessing contract exists as a single, pure, frozen implementation that training pre-tokenization (v0.6) and the inference Predictor (v0.9) will both import — no alternate paths. Two parameterizations are frozen (candle-geometry as designated primary, per-field log-returns as the A/B challenger; v0.5's tokenizer reconstruction decides the survivor). All scaling is **causal by construction**, proven by a prefix-consistency audit that fails loudly if any future statistic sneaks in. Gate **G2** closes this version; **no tokenizer work may begin before it.**

**Starting state:** v0.3 complete (segment index `clean/v1/` over the full M0 corpus, usable-windows table, TR tier, LIMITATIONS.md started).
**Exit gate:** **G2** — golden vectors + property tests green in CI; contract frozen at `schema_version=1`; causality audit in CI and passed corpus-wide.
**Total budget:** ~6–7 focused days (≈ 1–1.5 calendar weeks). Modal spend target < $5. Kaggle GPU-hours: 0.

**Non-goals for v0.4 (scope fence):** no tokenizer or quantizer code (hard gate — scope discipline is itself an exit-checklist item); no feature storage (v0.6 stores; v0.4 only streams for stats); no quantizer level/range decisions (v0.5 consumes this version's distribution report); no per-series *adaptive* price scaling (documented trade-off, post-1.0 experiment); no training-loop concerns (Kronos's 5 % volume dropout stays a v0.5 item); nothing public; no GPU.

**Repos/services this version creates: none.** Outputs are code, committed config/constants files, and a stats report. State explicitly in Phase G.

**Design stance (read before building):** the contract must be a **pure function of (provided bar sequence, frozen constants)** — nothing else. That single rule resolves every hard question in this version: it forces frozen (not fitted-at-runtime) scale constants, forbids per-window statistics (Kronos's leaky z-score), makes pre-tokenization-per-segment and streaming inference automatically consistent, and turns "causality" into a testable property: **prefix-consistency** — `transform(bars[0..t])` must equal the first rows of `transform(bars[0..n])`, exactly.

---

## Phase A — ADRs, firewall, spec freeze (budget: 1 day)

### A1. `docs/adr/0017-preprocessing-contract.md` — the load-bearing ADR
- [ ] **Anchor-bar rule:** feature rows exist for bars `t ≥ 1` of each segment; bar 0 is consumed as the anchor (`close_0` seeds the gap feature and the Predictor's price inversion). Consequence: usable-windows math becomes `Σ max(0, (n_bars − 1) − 511)` — recompute and supersede the v0.3 table (explicit task, Phase E).
- [ ] **Spec `geo-v1` (primary), 6 features:** `gap g_t = log(open_t/close_{t−1})`, `body b_t = log(close_t/open_t)`, `upper u_t = log(high_t/open_t)`, `lower l_t = log(low_t/open_t)`, `vol v_t`, `amt a_t` (below). Structural identities to encode as invariants: `u_t ≥ max(0, b_t)` and `l_t ≤ min(0, b_t)` always (follow from high ≥ max(o,c), low ≤ min(o,c)).
- [ ] **Spec `ret-v1` (challenger), 6 features:** `r_o, r_h, r_l, r_c = log(x_t/close_{t−1})` for x ∈ {open, high, low, close}; same `v_t, a_t`. Invariants: `r_h ≥ max(r_o, r_c)`, `r_l ≤ min(r_o, r_c)`.
- [ ] **Price-feature scaling:** frozen affine constants per (asset_class × frequency × feature): `center` = robust median, `scale` = IQR/1.349, computed on **pre-firewall data only** (Phase B) and committed as a versioned constants file. No runtime fitting, no per-series adaptivity — record the trade-off honestly: cross-asset vol spread within a class is absorbed by clip + quantizer range + the conditioning embeddings (v0.7); per-series adaptive scaling is a named post-1.0 experiment.
- [ ] **Volume/amount:** `log1p` → **relative-to-own-past**: `v_t = log1p(vol_t) − RM_t`, where `RM_t` = median of `log1p(vol)` over the strictly-past window `[max(0, t−L), t−1]`, `L = 256` — i.e., expanding from segment start until L is reached, then rolling L. No prior-blending (the expanding phase *is* the warm-up; `RM_1` = median of one value, well-defined). Then frozen class×frequency scale, clip. Same for `amount` (kept for Kronos 6-dim parity; near-redundancy with `v_t` + price features noted in the ADR). Early-bar distribution note: segment starts in *training* also pass through the expanding phase, so inference contexts shorter than L see feature distributions the model has genuinely trained on — a principled, documented alignment, not a leak.
- [ ] **Clip:** `[−5, +5]` in scaled units for all six features (the Kronos clip, now applied to causal values). Clip events counted, reported (Phase E), never silently saturating > 0.5 % of any feature without investigation.
- [ ] **Validity & NaN policy:** contract **rejects loudly** — any non-positive price (a new guard beyond v0.1's invariants), any NaN/Inf input, any non-contiguous `ts` within the provided segment → typed exception, no silent fill, no coercion. NaN/Inf in *output* on valid input = bug, property-tested to zero.
- [ ] **Dtypes & determinism:** compute in float64, emit float32; same-platform runs are bit-identical (CI-enforced); cross-platform golden comparisons use atol 1e-9 (documented).
- [ ] **Single-implementation rule:** public API surface is exactly `{load_spec, load_constants, transform, inverse}`; v0.6 and v0.9 must import these — recorded here, enforced by an API-surface test (Phase D) and a RUNBOOK rule.
- [ ] **Leaky baseline (optional, Phase F):** Kronos's per-window z-score+clip may exist only as `kronos-zscore-v0`, marked leaky, refused by all production paths.

### A2. `docs/adr/0018-temporal-firewall.md` — pulled forward from v0.5, deliberately
- [ ] The roadmap places the firewall declaration in v0.5, but Phase B's constants must be computed on pre-firewall data — so the **date** is fixed now; v0.5 still owns the full sealed-holdout *governance* (embargo width, holdout hashing). Record this as an explicit schedule deviation with reason.
- [ ] Choose `firewall_ts` such that the post-firewall span is ≥ 18 months across all five M0 slices (check against the registry's max `last_ts`); write `configs/firewall.yaml` (`firewall_ts`, rationale, registry hash consulted); commit; record the file's sha256 inside the ADR itself (hash-commit).
- [ ] Standing rule stated: *no statistic, constant, threshold, or model parameter may be derived from bars at or after `firewall_ts` until v0.8's sealed evaluation.*

### A3. Spec config files
- [ ] `configs/contract_geo_v1.yaml` and `configs/contract_ret_v1.yaml`: L, clip bounds, dtype, feature order, `schema_version: 1`, `spec_id`; config-hash logged by every consumer.

**Phase A acceptance:** ADRs 0017–0018 merged; `firewall.yaml` committed and hash-recorded; both spec configs committed; the v0.3 usable-windows correction is queued as a Phase E task, not forgotten.

---

## Phase B — Frozen constants job (budget: 0.5–1 day)

- [ ] `remote/modal/contract_constants.py` + `axiom contract fit-constants --spec geo-v1|ret-v1`: registry- and segment-driven streaming pass over **pre-firewall** bars only → per (asset_class × frequency × feature) robust `center`/`scale` → `configs/contract_constants_v1.yaml` (one file, both specs' tables) with a generation manifest block (registry hash, `clean_config_hash`, firewall hash, git commit, row/bar counts consumed).
- [ ] Firewall enforcement in-code, not by convention: the job asserts `max(ts consumed) < firewall_ts` and writes that assertion's result into the manifest.
- [ ] Determinism: run twice → byte-identical YAML (stable ordering, fixed float formatting).
- [ ] Sanity bounds test: every `scale` > 0 and finite; `u`'s center > 0 and `l`'s center < 0 for geo-v1 (one-sidedness respected); crypto scales > FX scales at the same frequency (spot-check expectation, logged not hard-asserted).
- [ ] Constants file committed to git — constants are part of the contract, versioned with it.

**Phase B acceptance:** constants committed with manifest; determinism + firewall assertions green; file size trivial (a few KB).

---

## Phase C — Contract implementation (budget: 1.5–2 days)

### C1. `src/axiom/contract/`
- [ ] `spec.py`: frozen pydantic models for specs + constants (`extra="forbid"`, immutable); loaders that verify config-hash ↔ file integrity.
- [ ] `rolling.py`: the strictly-past rolling/expanding median. **Performance decision with fallback ladder:** default `bottleneck.move_median` over numpy (C-speed; verify 2026 maintenance status and pin — flag in ADR-0017 if adopted); fallback #1 pandas `rolling(L).median()` (accepting the pandas dependency); fallback #2 hand-rolled two-heap streaming median (correct but slow — acceptable only if the Phase E throughput check still fits Modal budgets). Whichever wins: wrap behind one function; property-test against a brute-force reference on small inputs; the *strictly-past* offset (window ends at t−1, never t) is the #1 bug risk — test it explicitly at every boundary (t = 1, t = L, t = L+1).
- [ ] `transform.py`: `transform(bars, spec, constants) -> FeatureBlock` (float32 array [n−1, 6] + `feature_names` + anchor metadata + clip-event counts). Both specs behind the same signature; zero I/O, zero globals, import-time-network forbidden (reuse the v0.3 purity test pattern).
- [ ] `inverse.py`: `inverse(features, anchor_close, spec, constants) -> bars` — exact algebraic inversion (un-clip is impossible; inversion is defined on unclipped-range features, and the round-trip property is asserted on inputs whose features fall inside clip bounds; clipped features invert to the clip boundary — documented). Reconstructed bars must pass `validate_bars` and the structural identities (geo: high = open·e^u ≥ max(open, close) by construction — assert anyway).
- [ ] `api.py`: the four-function public surface; everything else underscore-private.

### C2. Micro-benchmarks (informs v0.6 sizing)
- [ ] `pytest-benchmark` (dev extra) micro-bench: bars/sec through `transform` for a 100 k-bar segment, both specs, recorded in the Phase E report. Target: ≥ 1 M bars/sec/core with bottleneck (order-of-magnitude check, not a hard gate).

**Phase C acceptance:** both specs transform + invert; purity and API-surface constraints hold; rolling-median reference tests green including all boundary offsets.

---

## Phase D — Test battery: golden vectors + the causality audit (budget: 1–1.5 days)

### D1. Golden vectors (synthetic — the no-market-bytes-on-laptop rule stands)
- [ ] Build 6 golden fixtures from the v0.3 synth toolkit (plain walk, walk-with-gap-feature edge, high==open bar, low==open bar, zero-volume run, L-boundary-length segment); for each: input table + expected feature rows for **both specs**, with 3–5 rows per fixture hand-verified against an independent spreadsheet/manual calculation (note the verification method in the fixture header).
- [ ] Golden test: exact match same-platform; atol 1e-9 comparison path for cross-platform CI.

### D2. Property battery (hypothesis, both specs, warm-up region always included)
- [ ] **Prefix-consistency (the causality audit, headline of G2):** for random valid synth segments and random split points t: `transform(bars[:t+1])` rows == first t rows of `transform(bars)` — **exactly** (float64 pipeline, same platform). This single property is the contract's definition of causal.
- [ ] **Perturbation-invariance (second, independent causal probe):** perturb bar j (valid mutation of close/high/low/volume); features at all rows < j−? — precisely: rows with bar-index < j — unchanged bit-exactly; rows ≥ j may change.
- [ ] No-NaN/Inf on valid input; clip bounds honored ([−5, 5] inclusive); float32 output dtype.
- [ ] Round-trip: `inverse(transform(bars)) == bars` within atol on in-clip-range inputs; structural identities on reconstructed bars.
- [ ] Chunked-equals-whole: transforming a segment in one call == concatenation of prefix-extension calls (streaming-safety, corollary of prefix-consistency but tested separately for the API path).
- [ ] Invalid-input rejection: non-positive price, NaN, Inf, non-contiguous `ts`, empty/1-bar segment → typed exception, correct error code, never partial output.
- [ ] API-surface test: `axiom.contract` exports exactly the four public names.
- [ ] Constants-sensitivity: changing any constant changes outputs (guards against dead config paths).

### D3. CI wiring
- [ ] The full battery runs in the standard CI test job (fast: synth-sized inputs); the causality properties are additionally tagged `@causality` so v0.8's leakage-tripwire suite can re-run them by marker.

**Phase D acceptance:** everything green across the 3.11/3.12/3.13 matrix; a deliberately-introduced leak (temporary mutation: rolling window including index t) makes prefix-consistency fail loudly — do this once as a live-fire drill, then revert (note the drill in the PR description).

---

## Phase E — Corpus dry-run, distributions & stats (budget: 0.5–1 day)

- [ ] `remote/modal/contract_dryrun.py` + `axiom contract dryrun --spec both`: stream every cleaned segment through both specs (features discarded, stats retained): per (asset_class × frequency × feature) — quantiles (0.1/1/5/25/50/75/95/99/99.9 %), clip-event rate, NaN count (must be 0), throughput (bars/sec).
- [ ] **Corpus-level causality spot-audit:** for 50 random real segments, run the prefix-consistency check cloud-side at 3 random split points each — 150/150 must pass (the CI property uses synth data; this closes the loop on real data without moving bytes to the laptop).
- [ ] **Regression snapshots:** for 5 pinned series (e.g., BTCUSDT-1h, EURUSD-1h, AAPL-1d, XAUUSD-1d, ETHUSDT-1d), compute the sha256 of the full feature block per spec and commit the *hashes* (not the data) to `tests/snapshots/contract_v1.json` — any future contract change that alters outputs trips this immediately.
- [ ] **Usable-windows correction:** recompute the v0.3 usable-windows table under the anchor-bar rule (`n_bars − 1`) and publish it in this version's report as the superseding table (cross-reference in the v0.3 report via a one-line addendum).
- [ ] Red-flag review: clip rate > 0.5 % on any (class × frequency × feature) → investigate (constants wrong? pathological slice?); geo-v1 vs ret-v1 distribution comparison narrated (input for the v0.5 A/B and quantizer range planning).
- [ ] `docs/reports/v0.4-contract-qa.md`: all of the above + benchmark numbers + Modal cost; trackio summary logged.

**Phase E acceptance:** report committed; 150/150 spot-audit; snapshots committed; clip rates pass or carry written investigations; the v0.5 quantizer now has real feature distributions to design against.

---

## Phase F — Optional: leaky Kronos baseline spec (budget: 0.5 day, skippable)

- [ ] `kronos-zscore-v0`: per-window per-feature z-score + clip [−5, 5], faithful to `KronosPredictor` — implemented **only** to let v0.5 quantify what the leaky normalization buys in reconstruction A/Bs.
- [ ] Guards: spec flagged `leaky=True`; `transform` accepts it only with `allow_leaky=True`; the Predictor (v0.9) and pre-tokenization (v0.6) reject leaky specs by construction — test both refusals now (the guard is cheap; the future mistake it prevents is not).
- [ ] Excluded from prefix-consistency (it fails it by design — assert that it *does* fail, as documentation-by-test).

**Phase F acceptance (if taken):** baseline spec exists, guarded, and demonstrably fails the causality audit — the whole point, captured in a test name.

---

## Phase G — Freeze, docs, tag, exit review (budget: 0.5 day)

- [ ] Freeze declaration: `SCHEMA_VERSION = 1` as a code constant re-exported by `axiom.contract`; ADR-0017 stamped "frozen — changes require schema_version bump + new constants + new snapshots."
- [ ] `docs/ARCHITECTURE.md`: C5 **complete**; C6 (tokenizer) unblocked pending G2 sign-off. `docs/REPOS.md`: "v0.4 created no new online infrastructure."
- [ ] `docs/RUNBOOK.md`: constants regeneration rules (allowed only with unchanged firewall + explicit version bump), the single-implementation rule, how to read the dryrun report, the live-fire-drill practice.
- [ ] `CHANGELOG.md` → `[0.4.0] - <date>`; `git tag v0.4.0`; push.
- [ ] **Gate G2 exit checklist** (all must be true):
  - [ ] Golden vectors + full property battery green in CI across the Python matrix; `@causality` marker wired.
  - [ ] Prefix-consistency and perturbation-invariance pass on synth (CI) **and** 150/150 on real segments (cloud spot-audit).
  - [ ] Live-fire drill performed once: an injected leak failed the audit loudly (referenced in a PR).
  - [ ] Both specs frozen at `schema_version=1`; constants committed with a firewall-respecting generation manifest; `firewall.yaml` hash-recorded in ADR-0018.
  - [ ] Regression snapshot hashes committed for 5 pinned series × 2 specs.
  - [ ] Clip rates within bounds or investigated in writing; feature-distribution report available for v0.5.
  - [ ] Usable-windows table corrected for the anchor-bar rule and published.
  - [ ] Public API surface = exactly four functions; leaky spec (if built) provably rejected by production paths.
  - [ ] **No tokenizer/quantizer code exists in the repo** (scope discipline, checked by inspection).
  - [ ] Zero market-data bytes on laptop or home PC; zero Kaggle GPU-hours; Modal spend < $5; nothing public.

**Then:** v0.5 "Tokenizer" TODO (next session) — vendor Kronos's `BSQuantizer` + tokenizer encoder/decoder under the NOTICE, wire Huber reconstruction in contract space, run the BSQ-default vs flat-FSQ ablation and the geo-v1 vs ret-v1 A/B on Kaggle (first GPU hours of the project), codebook-health metrics, sealed-holdout governance completed, and Gate **G3** — which also decides corpus M1 and the TRC filing.
