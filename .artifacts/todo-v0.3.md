# Axiom v0.3 — "Clean" — Phased Development TODO

**Goal:** the Kronos cleaning pipeline (Appendix B, Algorithm 1 `FilterLowQualitySegments`) exists as pure, config-driven, fully-tested code and has run over the entire M0 corpus. Its output is a **segment index** — a versioned metadata table of valid contiguous sub-sequences per series, with per-rule cut attribution — plus drop statistics per source × frequency, reviewed and eyeballed sane. The split/dividend policy is *applied*: tokenization consumes the audited vendor-adjusted series, and a derived total-return close series exists for future eval labels. Raw bars are never rewritten.

**Starting state:** v0.2 complete (five M0 slices in `axiom-raw`, registry built, adjustment audit verdict written in `docs/reports/v0.2-adjustment-audit.md`).
**Exit gate:** roadmap §4/v0.3 gate — all cleaning tests green; per-rule drop statistics reported per source/frequency and eyeballed sane (expanded in Phase G).
**Total budget:** ~6–7 focused days (≈ 1–1.5 calendar weeks). Modal spend target < $5. Kaggle GPU-hours: 0.

**Non-goals for v0.3 (scope fence):** no normalization, feature transforms, or log-return work (v0.4 — the contract); no training-time volume/amount dropout (Kronos's 5 % trick is a v0.5 training-loop concern, noted here only so it isn't forgotten); no rewriting or filtering of raw Parquet (cleaning output is metadata; the only new *data* is the small derived TR series); no intraday-equities session calendars beyond design notes; no tokenizer/model work; no GPU; nothing public.

**Repos/services this version creates: none.** Outputs land in `axiom-raw` under `clean/{clean_version}/` and `derived/tr_close/`. State this explicitly in Phase G.

**Key design stance (why a segment index):** raw is immutable and cleaning thresholds will be tuned; representing cleaning as data would mean duplicating the corpus per config change. A segment table (`[start_ts, end_ts]` + cut reasons, bound to the exact raw file hash) is cheap to regenerate, versionable, and keeps a single source of truth. Downstream (v0.4 contract, v0.5 tokenizer, v0.6 shards) reads raw bars *through* the segment index.

---

## Phase A — ADRs & clean config (budget: 1 day)

### A1. `docs/adr/0015-cleaning-semantics.md` — pin every interpretive choice
- [ ] **Order of operations (fixed):** (1) Stage-1 gap partition → (2) `PartitionByPriceJumps` → (3) illiquid-run excision → (4) stagnant-run excision → (5) min-length filter. Order is part of the config hash; a test asserts order-dependence is covered.
- [ ] **Jump rule:** cut between t−1 and t when `|open_t / close_{t−1} − 1| > θ_jump(freq)`; bar t starts a new segment. Note the inherited Kronos behavior we *keep*: genuine extreme market moves (e.g., a crypto collapse crossing a bar boundary at > 20 %) get cut, so the model is systematically under-exposed to extreme discontinuities — recorded as a known bias for `docs/LIMITATIONS.md` and the model card. Intrabar crashes (low spikes with recovering close) correctly do **not** trigger.
- [ ] **Gap rule (session-aware — the adaptation Kronos glosses over):** a missing grid step is a hard boundary *only if unexpected for the series' session*. Crypto (`24x7`): strict grid, any missing bar = boundary. FX/commodities (`24x5`): the weekend window (Fri ~22:00 UTC → Sun ~22:00 UTC, ± DST tolerance) is expected, never a boundary; any other gap is. US equities 1d (`XNYS-regular`): weekends + exchange holidays expected via the `exchange_calendars` package (XNYS calendar; verify the package's 2026 maintenance status at implementation and pin the version); any non-calendar gap (suspension, delisting hole) is a boundary. No imputation anywhere; boundaries partition, period. (Kronos's Stage-1 "impute volume/amount = 0 when missing" is recorded as N/A for our sources — volume always present, amount synthesized upstream.)
- [ ] **Illiquid run:** consecutive bars with `volume ≤ illiquid_eps` (default `0.0`, exact zero — the "near-zero" in the paper is an interpretive choice; ours is pinned and configurable). Runs longer than `max_illiquid(freq)` are excised; excision splits the series.
- [ ] **Stagnant run:** consecutive bars with exactly equal `close` (float equality is meaningful — values come from a stable CSV→float64 parse). Runs longer than `max_stagnant(freq)` are excised. Note: US LULD halts and thin small-caps will be excised by this rule — acceptable, documented.
- [ ] **Min-length filter:** surviving segments shorter than `min_bars(freq)` are dropped.
- [ ] **Binding:** every segment row carries `clean_config_hash`, `clean_version`, and the `raw_artifact_sha256` of the file it was derived from — a changed raw file invalidates its segments (staleness check in Phase E).

### A2. `docs/adr/0016-adjustment-policy.md` — branch on the v0.2 audit verdict
- [ ] Record the verdict verbatim, then pin the policy: **tokenization consumes the vendor-adjusted Stooq OHLC as-is** (split-adjusted confirmed by the audit; whether dividends are also adjusted determines the TR branch below). Dividend adjustment is *not* separately applied to tokenization inputs; the resulting small systematic ex-dividend gaps are documented (Kronos ignores them too — we say so out loud).
- [ ] **TR (total-return) close for eval labels:** if the audit found Stooq split-only → build `tr_close` from dividend events: `tr_t = tr_{t−1} × (close_t + div_t) / close_{t−1}`, anchored `tr_first = close_first`. If the audit found split+dividend-adjusted → `tr_close = close` (identity), still materialized for a uniform downstream interface. Crypto/FX/commodities: `adjustment_policy = none`, `tr_close = close`.
- [ ] Tickers without dividend-event coverage (yfinance gaps or total unavailability from v0.2): `tr_available = false` in the derived manifest; they are excluded from dividend-sensitive eval slices later, never silently approximated.

### A3. `configs/clean_v1.yaml`
- [ ] Full Kronos Table 4 (all rows, future-proofing M1): 1m 2048/0.10/15/45 · 5m 1024/0.15/3/10 · 10m 512/0.15/3/6 · 15m 512/0.15/2/5 · 20m 512/0.15/2/5 · 30m 512/0.20/2/3 · 40m 256/0.20/1/3 · **1h 256/0.20/1/3** · 2H 128/0.25/1/3 · 4H 128/0.25/1/3 · **1d 128/0.30/1/3** · 1w 16/0.50/0/2. Columns: `min_bars / θ_jump / max_illiquid / max_stagnant`. Footnote in-file: only the 1h and 1d rows are exercised in v1.0; the 10m/20m/40m/2H rows come from an earlier extraction and **must be re-verified against the paper PDF before first use**.
- [ ] Plus: `illiquid_eps`, stagnant definition ref, per-`session_id` expected-gap rules, stage order, `clean_version: 1`. Config-hash logged into every output.

**Phase A acceptance:** ADRs 0015–0016 merged (0016 contains the real audit verdict, not a placeholder); `clean_v1.yaml` committed; `exchange_calendars` added to the `data` extra and pinned.

---

## Phase B — Synthetic series toolkit (budget: 0.5–1 day)

> Built as library code, not test-local helpers — v0.4's contract tests and v0.8's leakage tripwires will reuse it.

- [ ] `src/axiom/testing/synth.py`: parameterized generators, each returning schema-v1-valid tables with a ground-truth annotation of where cuts *should* land: `walk(freq, n, seed)` base series; `with_split(ratio, at)` (unadjusted jump) and `with_adjusted_split(...)` (no jump); `with_gap(at, n_bars, kind={expected_weekend, expected_holiday, unexpected})`; `with_flash_crash(at, intrabar: bool)`; `with_limit_lock(at, n)` (constant close + zero volume); `with_stagnant(at, n)`; `with_illiquid(at, n)`; `with_rollover_jump(at)`; `with_dst_weekend(...)` (FX weekend edges shifted ± 1 h); `truncate_tail(n)` (short-segment bait); `ends_at(ts)` (delisting); `with_suspension(at, days)`.
- [ ] Generators for XNYS-calendar-shaped 1d series (real holiday dates for a chosen historical year, via `exchange_calendars`).
- [ ] Self-tests: every generator's output passes `validate_bars`; annotations are internally consistent (cut positions exist on the grid).

**Phase B acceptance:** toolkit merged with its own tests; no cleaning logic imported (independence — the toolkit must not share code with the thing it tests).

---

## Phase C — Cleaning engine (budget: 1.5–2 days)

### C1. `src/axiom/clean/`
- [ ] `calendars.py`: expected-gap predicates per `session_id` — `24x7` (strict), `24x5` (weekend window + DST tolerance), `XNYS-regular` (via `exchange_calendars`, holidays + weekends; half-days are irrelevant at 1d but the predicate interface takes frequency for future intraday use).
- [ ] `stages.py`: five pure, vectorized functions (numpy over pyarrow columns), each `(bars, config, session) -> list[SegmentSpan] + StageStats`; no I/O, no HF, no globals.
- [ ] `engine.py`: `clean_series(bars, identity, config) -> (segments_table, dropstats_rows)` composing the stages in the ADR-0015 order; segment rows: `source, market, asset_class, symbol, frequency, segment_id (= f"{symbol}:{freq}:{start_ts}"), start_ts, end_ts, n_bars, cut_reason_start, cut_reason_end, clean_config_hash, clean_version, raw_artifact_sha256`; dropstats rows: per (series, rule) → bars dropped, runs excised, segments created/dropped.
- [ ] Invariant guarantees encoded as asserts: segments non-overlapping, strictly ordered, every kept bar in exactly one segment, `kept + dropped == total`.

### C2. Tests — the roadmap's named edge cases, wired to Phase B
- [ ] Unadjusted split → jump cut at the annotated bar; adjusted split → no cut.
- [ ] Expected weekend/holiday gap (FX, XNYS) → no cut; unexpected gap (crypto missing hour; equity suspension) → cut; DST-shifted weekend → no false cut.
- [ ] Cross-bar flash crash → cut; intrabar flash crash → no cut (documented Kronos-consistent behavior).
- [ ] Limit-lock → excised by stagnant and/or illiquid rules exactly per Table 4 run limits (boundary cases: run == max kept, run == max+1 excised).
- [ ] Rollover jump → cut; delisting → clean `series_end`; suspension → boundaries both sides; short tail → dropped by min-length.
- [ ] Hypothesis properties over random `synth` compositions: invariants above always hold; **determinism** (same input + config → byte-identical segment table, stable hash); **idempotence** (cleaning the bars restricted to output segments yields the same segments); config-hash sensitivity (any threshold change changes the hash).
- [ ] Order-of-operations test: a crafted series where a different stage order would yield different segments — locks ADR-0015's order.

**Phase C acceptance:** full edge-case suite + properties green in CI; engine is pure (import-time network/IO forbidden, enforced by a test).

---

## Phase D — Adjustment layer + TR series (budget: 1 day)

- [ ] `src/axiom/adjust/policy.py`: the ADR-0016 branch implemented behind one interface: `tr_close(bars, events | None, verdict) -> table`; exact-arithmetic tests with synthetic splits/dividends (including: dividend on a gap day, consecutive dividends, zero-dividend identity, split+dividend same day).
- [ ] Build job `axiom derive tr --universe configs/universe_equities_v1.yaml` (Modal): joins `raw/stooq/us/1d/**` with `raw/yahoo/adjustments/**` → `derived/tr_close/us/1d/{first_char}/{TICKER}.parquet` (`ts`, `tr_close`) + sidecar manifests (`derived_from` raw + events hashes, `tr_available`, policy verdict, config hash). Letter-sharded (B1-v0.2 helper), tiny footprint (< 200 MB).
- [ ] Coverage report: % of the equities universe with `tr_available = true`; if yfinance was unavailable in v0.2, the whole tier is `tr_close = close` with `tr_available = false` where dividends mattered — stated plainly, feeding v0.8's eval-slice definitions.
- [ ] Crypto/FX/commodities: no materialization — `tr_close ≡ close` is a documented identity handled at read time (don't duplicate 20 M rows to store a copy of `close`).

**Phase D acceptance:** TR tier built for equities per the real verdict; arithmetic unit-tested; coverage stated in the QA report draft.

---

## Phase E — Corpus clean run on Modal (budget: 0.5–1 day)

- [ ] `remote/modal/clean_run.py` + `axiom clean run --config configs/clean_v1.yaml`: registry-driven fan-out over every **bar** artifact (registry filter excludes `raw/yahoo/adjustments/**` and `derived/**`), `.map()` per series, driver concatenates → `clean/v1/segments.parquet`, `clean/v1/dropstats.parquet`, `clean/v1/run_manifest.json` (config hash, registry hash consumed, per-source coverage, wall time) → uploaded to `axiom-raw`.
- [ ] **Staleness guard:** before running, join registry vs any existing segment index; raw files whose `sha256` changed since the last clean are flagged and re-cleaned; `axiom clean run` refuses `--incremental` if the config hash changed (full rerun required). RUNBOOK rule: *clean runs after every pull; segments are never trusted across a config change.*
- [ ] Registry integration: `axiom registry query` gains canned post-clean views — **usable bars** per (source × frequency) after min-length, usable-window counts at context 512 (`Σ max(0, n_bars − 511)` per segment for 1h; the same at 1d) — this table is the direct input to v0.5 corpus sizing and belongs in the QA report.
- [ ] Scale check: ~50 M bars, vectorized → expect minutes-to-an-hour of Modal CPU; record actual cost.

**Phase E acceptance:** segment index + dropstats live in `axiom-raw/clean/v1/`; staleness guard demonstrated (touch one raw file's manifest in a scratch copy → detected); usable-bars view renders.

---

## Phase F — Drop-stats review & QA report (budget: 0.5 day)

- [ ] `docs/reports/v0.3-clean-qa.md` — the "eyeballed sane" gate made concrete:
  - Expected patterns (assert-in-prose, with numbers): crypto majors (BTC/ETH pairs) lose ≈ 0 % to cleaning; losses concentrate in small-cap equities and exotic pairs; FX weekend gaps contribute **zero** to drop counts (they're expected gaps); stagnant/illiquid excisions cluster in the illiquid tail of the equities universe.
  - Red-flag checks (any hit → investigate before gating): a major losing > 1 % of bars; jump cuts firing on vendor-adjusted equities at a rate ≫ plausible corporate-action frequency; any source × frequency losing > 15 % overall without explanation.
  - Top-20 most-cut series manually inspected (one-line verdict each: data problem vs real market pathology vs rule artifact).
  - The usable-bars and usable-windows tables (Phase E) — the corpus that actually exists for v0.5.
  - TR coverage numbers from Phase D.
- [ ] `docs/LIMITATIONS.md` (new, feeds the v0.9 model card): survivorship bias with a **quantification attempt** — share of Stooq tickers whose `last_ts` predates pull-date − 30 d (proxy for included-but-dead series) vs the known reality that free bulk data skews to survivors; extreme-event excision bias (A1 jump-rule note); LULD/halt excision; ex-dividend gap note.
- [ ] Trackio: log the drop-stat summary + usable-bars totals for the run.

**Phase F acceptance:** report committed; every red-flag check explicitly ticked pass/investigated; LIMITATIONS.md exists with real numbers.

---

## Phase G — Docs, tag, exit review (budget: 0.5 day)

- [ ] `docs/ARCHITECTURE.md`: C4 (cleaning) **complete**; C5 next. `docs/REPOS.md`: "v0.3 created no new online infrastructure"; `axiom-raw` layout section gains `clean/{version}/` and `derived/tr_close/`.
- [ ] `docs/RUNBOOK.md`: clean-after-pull rule, incremental vs full rerun semantics, how to read dropstats, how to bump `clean_version`.
- [ ] `CHANGELOG.md` → `[0.3.0] - <date>`; `git tag v0.3.0`; push.
- [ ] **v0.3 exit checklist** (all must be true):
  - [ ] Entire edge-case suite (splits, gaps, flash crash, limit-lock, DST, holidays, rollover, delisting, suspension, min-length) + hypothesis invariants green in CI; engine pure; toolkit independent.
  - [ ] `clean/v1/segments.parquet` + `dropstats.parquet` cover every bar artifact in the registry; every segment bound to a `raw_artifact_sha256`; invariants (non-overlap, full accounting) verified corpus-wide, not just in tests.
  - [ ] Determinism proven: a second full clean run produces a byte-identical segment table (same hash).
  - [ ] Drop-stats report committed; all red-flag checks pass or carry a written investigation; top-20 inspected.
  - [ ] Usable-bars / usable-windows tables exist — v0.5's corpus size is now a known number, not an estimate.
  - [ ] TR tier built per the audit verdict; arithmetic tests green; coverage documented.
  - [ ] `LIMITATIONS.md` started with quantified survivorship + excision biases.
  - [ ] Staleness guard works; RUNBOOK rules written.
  - [ ] Zero market-data bytes on laptop or home PC; zero Kaggle GPU-hours; Modal spend < $5; nothing public.

**Then:** v0.4 "Contract" TODO (next session) — the versioned preprocessing contract: candle-geometry parameterization vs per-field log-returns A/B, causal volume/amount scaling (expanding median/IQR, strictly past), clip + NaN policy, `schema_version=1` freeze, golden vectors, and the causality audit that must fail loudly before any tokenizer work (Gate **G2**).
