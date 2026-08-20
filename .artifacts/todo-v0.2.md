# Axiom v0.2 — "Corpus Breadth" — Phased Development TODO

**Goal:** the raw tier grows from one source to four. A Dukascopy loader lands FX + commodities (1h + 1d), a manual-assisted Stooq flow lands full-history US daily equities, and a rate-disciplined yfinance adjunct captures split/dividend events for cross-checking — all cloud-to-cloud into the existing `axiom-raw`. A corpus registry is built over every sidecar manifest so one query answers *"what do we have, from where, pulled when."* Corpus **M0** is assembled.

**Starting state:** v0.1 complete (Binance spot + um at 1h + 1d in `axiom-raw`, ≥ 100 pairs, sidecar-manifest idempotence proven, registry not yet built).
**Exit gate:** roadmap §4/v0.2 exit checklist (expanded in Phase G).
**Total budget:** ~8–9 focused days (≈ 1.5–2 calendar weeks). Modal spend target < $8. Kaggle GPU-hours: 0.

**Non-goals for v0.2 (scope fence):** no cleaning or filtering (v0.3 — raw stays faithful); no *application* of adjustment policy (v0.2 captures adjustment facts, v0.3 applies policy); no intraday equities; no CN equities (Qlib/Baostock deferred indefinitely per research); no COIN-M futures; no index CFDs (optional stretch only, off by default); no tokenizer/model work; no GPU kernels; nothing public.

**Repos/services this version creates: none.** All data lands in the existing private `m-de-graaff/axiom-raw`; `docs/REPOS.md` gets a layout update, not a new row. (State this explicitly in Phase G — "no new online infrastructure" is a deliberate, documented outcome.)

**Carried unverifieds (resolve by smoke test before building on them):** Dukascopy datafeed reachability from Modal IPs (C1); Stooq post-CAPTCHA URL validity from a different IP (D1); Yahoo blocking of cloud IPs (E1 — yfinance is explicitly non-load-bearing, failure is an acceptable outcome).

---

## Phase A — ADRs & universes (budget: 1 day)

### A1. ADRs (extend the set)
- [x] `docs/adr/0012-multi-source-raw-conventions.md` — cross-source conventions: per-source `price_side` (`trade` for Binance, `bid` for Dukascopy, `trade` for Stooq), `volume_convention` values (`base+quote_native`, `dukascopy_tick_volume`, `shares`), `amount_synthesized` rule (Kronos synthesis `amount = volume × mean(OHLC)` applies to Dukascopy + Stooq, always flagged); **`exchange_tz`/`session_id` stay file-level metadata + manifest fields, not columns** — this supersedes the v0.1 closing teaser ("promoted to real columns") and the ADR must say so and why (constant per file for all current sources; per-row session columns deferred until intraday session-bound markets exist, post-1.0); daily-bar timestamp convention: `ts` = 00:00 UTC of the exchange calendar date, actual trading timezone recorded in metadata (`America/New_York` for US equities); schema stays **v1** — no column changes, no migration.
- [x] `docs/adr/0013-dukascopy-source.md` — instrument set (A2), **bid candles** via the `dukascopy-python` interval API (pre-aggregated hourly/daily; not tick-level `.bi5` assembly — `tick-vault`-style tick ingestion is explicitly out of scope), tick-volume caveat (FX volume is indicative, not exchange volume), the **0-indexed month** gotcha in Dukascopy's URL scheme (verify the library handles it; never hand-build URLs without a test), canonical-symbol map (`EURUSD`, `XAUUSD`, `WTICOUSD`, … ↔ library instrument constants, with `source_symbol` recorded in the manifest), weekend/holiday absence-of-bars is expected (24×5 sessions).
- [x] `docs/adr/0014-equities-source.md` — Stooq bulk (`stooq.com/db/h/`, CAPTCHA-gated since Dec 2020) is the primary; ingestion is **manual-assisted with a URL handoff**: Mark solves the CAPTCHA in his laptop browser, copies the resulting direct archive URL, and hands it to a Modal job that downloads cloud-side — the archive bytes never persist on the laptop. **Sanctioned fallback** (only if the URL is IP-bound): download on laptop → immediately `hf upload` to `axiom-raw/staging/stooq/` → delete locally → Modal consumes from staging → staging pruned; this is the *single* permitted transient-laptop exception, logged in the pull manifest when used. yfinance is adjunct-only (events + cross-check, never a corpus pillar). Redistribution classes per source (A3 table). Survivorship bias: Stooq bulk skews to currently-listed tickers → accept-and-document (feeds the v0.9 model card).
- [x] Optional-stretch flag documented (default off): Dukascopy index CFDs (`USA500IDXUSD` etc.) as an equity-index proxy series.

### A2. Pinned universes
- [x] `configs/universe_dukascopy_v1.yaml` — hand-pinned, committed, with per-instrument start year: FX majors (EURUSD, USDJPY, GBPUSD, USDCHF, AUDUSD, USDCAD, NZDUSD), ~14 liquid minors/crosses (EURGBP, EURJPY, GBPJPY, EURCHF, AUDJPY, EURAUD, CADJPY, CHFJPY, AUDNZD, EURCAD, GBPCHF, AUDCAD, NZDJPY, GBPAUD), commodities (XAUUSD, XAGUSD, WTI, Brent, NatGas, Copper — mapped to Dukascopy CMD instruments). ~27 instruments total; selection rationale = notoriety-liquidity, recorded in ADR-0013 (no measurement step for a fixed small set).
- [x] Equities universe is **not** pinned by hand — criteria pinned now in ADR-0014 (≥ 5 y history, top-N by median dollar volume, N ≈ 3000, computed from the pulled data), generation deferred to Phase F where the data exists.

### A3. Redistribution classification table
- [x] `docs/DATA_LICENSING.md`: one row per source — Binance (public dumps; loader+manifest; private cache only), Dukascopy (broker datafeed; loader+manifest; gray, never redistribute), Stooq (personal/non-commercial ToS; loader+manifest; never redistribute), Yahoo/yfinance (scraped, no license; loader-only; events cached privately, never redistributed). Manifest gains a `redistribution_class` field (B2) referencing this table.

**Phase A acceptance:** ADRs 0012–0014 merged; Dukascopy universe committed; licensing table committed; the v0.1-teaser deviation is explicitly recorded, not silently dropped.

---

## Phase B — Source framework refactor + manifest evolution (budget: 1 day)

> One source is a script; four sources are a framework. Extract the generic machinery *before* writing loader #2, so Dukascopy and Stooq are implementations, not copies.

### B1. `src/axiom/sources/base.py`
- [x] `Source` protocol: `enumerate_work(universe, remote_state) -> list[WorkItem]`, `fetch(item) -> RawPayload`, `parse(payload) -> pa.Table`, `identity(item) -> (market, asset_class, frequency, symbol)`. Shared driver owns: retry/backoff/jitter, concurrency semaphore, `validate_bars`, Parquet+metadata write, sidecar emission, `is_current` idempotence skip, batched HF upload, `PullRunManifest` aggregation.
- [x] Refactor `binance_vision.py` onto the protocol. **All v0.1 tests stay green unchanged** — that is the refactor's acceptance test.
- [x] Folder-sharding helper for HF's <10 k-files-per-folder limit: equities land as `raw/stooq/us/1d/{first_char}/{TICKER}.parquet` (letter buckets); guard test asserts no planned layout exceeds 9 000 files/folder and total repo files stay ≪ 100 k.

### B2. Manifest model (backward-compatible extension — no sidecar rewrites)
- [x] Add optional fields with v0.1-equivalent defaults: `price_side` (default `trade`), `source_symbol` (default = `symbol`), `redistribution_class` (default `loader_manifest_private_cache`), `staging_exception_used` (default false). Existing crypto sidecars remain valid as-is; `is_current` untouched.
- [x] `validate_bars` gains a session-aware grid mode: 1h series may have absent weekend/holiday bars (already legal — gaps allowed), 1d equities must be date-aligned at 00:00 UTC with weekday-gap tolerance; crypto keeps the strict 24×7 expectation as a *QA statistic*, never a hard failure (raw = faithful).

**Phase B acceptance:** `just check` green; Binance pull smoke re-run (`--symbols BTCUSDT --limit 1`) on Modal produces sidecars byte-compatible with v0.1 (modulo `loader_version`); framework has zero source-specific code.

---

## Phase C — Dukascopy loader (budget: 1.5–2 days)

### C1. Reachability smoke (do this first — carried unverified)
- [ ] Minimal Modal job: `dukascopy-python` fetch of EURUSD hourly bid candles for one recent month. If Modal IPs are blocked/throttled → fallback ladder documented in RUNBOOK: (1) Kaggle CPU kernel as pull backend, (2) reduced concurrency + backoff. Record the outcome in ADR-0013.

### C2. `src/axiom/sources/dukascopy.py` (on the B1 protocol)
- [ ] Dependency: `dukascopy-python` (≥ v4.0.1) added to the `data` extra; pin exact version in lock.
- [ ] `enumerate_work`: per instrument × {1h, 1d}, chunked **by calendar year** from the pinned start year to now (bounds memory, makes work items idempotently small); remote-state skip = year-chunk granularity via sidecar `last_ts` (a symbol's file is re-extended, not re-pulled, for the current year only — document the append-rewrite: the whole per-symbol Parquet is rewritten from cached year frames? **Simplest correct rule:** re-fetch only the current calendar year, re-parse, splice onto prior years' data re-read from the existing artifact, rewrite the file, update sidecar. Prior years are immutable by convention).
- [ ] `fetch`/`parse`: bid candles, UTC; normalize to schema v1 — `volume` = Dukascopy tick volume (float), `amount` = synthesized (flagged), `price_side = "bid"`, `exchange_tz = "UTC"`, `session_id = "24x5"`.
- [ ] Sanity assertions: no bars inside the weekend close window (Fri ~22:00 UTC → Sun ~22:00 UTC) beyond a small tolerance; monotone `ts`; grid alignment.
- [ ] Canonical-symbol map table with `source_symbol` in every manifest; unit test the map is total over `universe_dukascopy_v1.yaml`.

### C3. Tests (offline) + pull
- [ ] Monkeypatched fetcher returning synthetic frames: year-chunk splice logic, weekend-window assertion, volume/amount synthesis flags, idempotent re-run (immutable prior years untouched byte-wise).
- [ ] Full pull on Modal: ~27 instruments × 1h (from start years, many since 2003–2010) + 1d ≈ **3–5 M bars**, ~150–300 MB. Kill drill mid-pull → relaunch → sidecar skip works (inherited from framework, but verify once on this source).

**Phase C acceptance:** FX + commodities present at 1h + 1d with valid sidecars; QA spot-check (EURUSD 1h count vs years elapsed × ~24×5) sane; reachability outcome recorded.

---

## Phase D — Stooq US daily equities (budget: 1.5–2 days)

### D1. URL-handoff smoke (carried unverified)
- [ ] Mark solves the CAPTCHA for `d_us_txt.zip` on the laptop browser, copies the direct URL, runs `axiom pull stooq --archive-url <url>` → Modal downloads. If the URL is IP-bound and 403s from Modal → execute the ADR-0014 sanctioned fallback (laptop → `axiom-raw/staging/stooq/` → local delete → continue), set `staging_exception_used = true` in the pull manifest, log the local deletion in RUNBOOK terms. Either path: archive `sha256` self-computed and recorded (Stooq ships no vendor checksum).

### D2. `src/axiom/sources/stooq.py` (on the B1 protocol; "fetch" = read the staged/downloaded archive)
- [ ] Parse the bulk layout (per-ticker `.txt` under `data/daily/us/...`), format `TICKER,PER,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT` with `PER=D`, `DATE=YYYYMMDD`; keep NASDAQ/NYSE/AMEX **stocks + ETFs** directories; skip indices/futures/other directories (list the kept paths in ADR-0014).
- [ ] Normalize: strip the `.us` suffix into `symbol`, keep it in `source_symbol`; `ts` = 00:00 UTC of `DATE`; `volume` = shares; `amount` synthesized (flagged); drop `OPENINT`; `exchange_tz = "America/New_York"`, `session_id = "XNYS-regular"`, `adjustment_policy` = value determined by D3 (placeholder `vendor_adjusted_unverified` until then).
- [ ] Tolerances: tickers with < 30 rows skipped (recorded as skipped-short, not failed); malformed lines counted per file, file fails above a threshold (e.g., > 0.1 %); duplicate dates within a ticker → hard fail (raw must be trustworthy).
- [ ] Letter-sharded layout `raw/stooq/us/1d/{first_char}/{TICKER}.parquet` (B1 helper); expect ~12–18 k series, **~30–60 M bars**, ~1.5–3 GB — comfortably inside the 100 GB tier; the B1 folder-guard test covers the real file plan.
- [ ] Offline tests: synthetic Stooq txt fixtures for every rule above; archive-to-parquet end-to-end against a miniature fake zip.

### D3. Adjustment audit (empirical — feeds v0.3's policy and the model card)
- [ ] Known-split probes: AAPL 2020-08-31 (4:1), TSLA 2022-08-25 (3:1), NVDA 2024-06-10 (10:1) — assert no ~N:1 price discontinuity across the split date in the Stooq series ⇒ split-adjusted confirmed.
- [ ] Dividend probe: one large special-dividend case + one ordinary-dividend blue chip — compare Stooq close path against yfinance `auto_adjust=True` and `False` (E-phase data) to classify Stooq as split-only vs split+dividend adjusted.
- [ ] Record the verdict in `docs/reports/v0.2-adjustment-audit.md`; update `adjustment_policy` manifests accordingly (targeted sidecar regeneration for `raw/stooq/**` only).

**Phase D acceptance:** US daily equities ingested cloud-side; parse-failure rate < 0.1 % of files; adjustment verdict written; handoff-vs-fallback outcome recorded with the exception flag set truthfully.

---

## Phase E — yfinance adjunct: adjustment events + cross-check (budget: 1 day)

> Non-load-bearing by design. Partial success is success; total failure is a documented outcome, not a blocker.

- [ ] Pinned ticker list `configs/yahoo_events_v1.yaml`: current S&P 500 + Nasdaq-100 constituents (deduped, ~550), source and retrieval date recorded in the file header (manual copy is acceptable; this list is a cross-check population, not a survivorship-safe universe — say so in the header).
- [ ] `src/axiom/sources/yahoo_events.py`: per ticker fetch **splits + dividends event series** (and, for the D3 probes only, short adjusted/unadjusted daily windows); hard client-side rate limit ≤ 300 req/h with jittered pacing, resume via sidecars, failures tolerated and listed.
- [ ] Storage: `raw/yahoo/adjustments/{first_char}/{TICKER}.parquet` (rows: `ts`, `event_type` ∈ {split, dividend}, `value`), sidecars with `redistribution_class = loader_only_private`, tiny footprint (< 50 MB).
- [ ] Cross-check job `axiom raw crosscheck-equities --sample 25`: for 25 random tickers present in both sources, compare recent-2 y Stooq closes vs yfinance adjusted closes — report max-abs-relative-diff and correlation into `docs/reports/v0.2-adjustment-audit.md`.
- [ ] Run from Modal; if Yahoo blocks Modal IPs, retry once from a Kaggle CPU kernel; if both blocked, record "yfinance unavailable from cloud backends as of <date>" in the audit report and proceed — D3's split probes alone still stand.

**Phase E acceptance:** events captured for the majority of the pinned list *or* the unavailability is documented; cross-check numbers (or their absence) are in the audit report.

---

## Phase F — Corpus registry + equities universe (budget: 1 day)

### F1. Registry (`src/axiom/registry/`)
- [ ] `axiom registry build` (Modal): `HfApi.list_repo_files("axiom-raw")` → threaded download of every `*.manifest.json` → one row per artifact (all scalar manifest fields + path) → `registry/registry.parquet` (zstd) + `registry/summary.md` uploaded back into `axiom-raw`; deterministic ordering, `registry_hash` recorded; idempotent rebuild.
- [ ] `axiom registry query "<sql>"` — DuckDB over the registry parquet (downloaded to a temp path or read via httpfs), plus canned reports: **coverage matrix** (source × asset_class × market × frequency → series count, bar count, min/max `ts`), storage-by-source, gappiest-series, staleness (days since `last_ts`).
- [ ] Tests: registry build over a synthetic sidecar tree; coverage matrix golden output; a corrupted sidecar is reported, never silently dropped.

### F2. Equities training universe (criteria from ADR-0014, data-driven)
- [ ] `axiom universe build-equities` (Modal): from `raw/stooq/us/1d/**` — filter ≥ 5 y history, rank by median daily dollar volume (`close × volume`), take top ~3000 → `configs/universe_equities_v1.yaml` (criteria echoed, `universe_hash`, generation date, registry hash it was derived from). Committed to git.
- [ ] Note in the file header: pulled corpus ⊃ training universe — everything stays in `axiom-raw`; the universe governs *sampling* from v0.5 onward.

### F3. M0 assembly check
- [ ] Coverage matrix must show all five slices at required frequencies: crypto-spot (1h+1d), crypto-um (1h+1d), fx (1h+1d), commodities (1h+1d), us-equities (1d).
- [ ] Total raw bar count computed and stated **honestly** against the ~50 M M0 target (expected: ~15 M crypto + ~4 M Dukascopy + ~30–60 M equities ⇒ target met; if short, the shortfall and reason go in the QA report, not a silent pass).

**Phase F acceptance:** one command answers what/where/when; coverage matrix committed; `universe_equities_v1.yaml` committed; M0 verdict written.

---

## Phase G — QA, docs, tag, exit review (budget: 0.5–1 day)

- [ ] `docs/reports/v0.2-raw-qa.md`: per-source stats (series, bars, history depth, gap distributions), coverage matrix, adjustment-audit verdict, cross-check numbers, reachability outcomes (Dukascopy/Modal, Stooq URL handoff, Yahoo), Modal spend, storage total.
- [ ] `docs/REPOS.md`: explicit "v0.2 created no new online infrastructure"; `axiom-raw` layout section updated (dukascopy, stooq letter-sharded, yahoo/adjustments, staging policy + pruning rule, registry/).
- [ ] `docs/RUNBOOK.md`: Stooq CAPTCHA→URL handoff procedure step-by-step (+ the sanctioned fallback and its deletion log line), Dukascopy quirks (0-indexed months, bid convention, tick volume), yfinance rate discipline, registry rebuild-after-every-pull rule.
- [ ] `docs/ARCHITECTURE.md`: C2 (data acquisition) marked **complete for v1.0 scope**, C3 (schema + corpus infra) marked complete; next up C4.
- [ ] `CHANGELOG.md` → `[0.2.0] - <date>`; `git tag v0.2.0`; push.
- [ ] **v0.2 exit checklist** (all must be true):
  - [ ] Registry built; canned queries answer what/from-where/pulled-when without touching raw files.
  - [ ] Coverage matrix shows all five M0 slices at their required frequencies, every artifact with a valid sidecar.
  - [ ] Total raw bars ≥ ~50 M, or a written shortfall analysis exists.
  - [ ] Adjustment audit verdict recorded; Stooq manifests carry the evidence-based `adjustment_policy`.
  - [ ] Dukascopy kill-drill + idempotent resume verified; immutable-prior-years rule holds byte-wise on a sample.
  - [ ] Stooq ingestion ran cloud-side; if the staging fallback fired, `staging_exception_used` is true in the pull manifest, local deletion is logged, and `staging/` is pruned.
  - [ ] `universe_equities_v1.yaml` + `universe_dukascopy_v1.yaml` committed with hashes.
  - [ ] All new loaders fully tested offline; CI green; no live network in CI.
  - [ ] Storage < 10 GB total (headroom check vs the 100 GB tier passes trivially).
  - [ ] Zero market-data bytes resident on laptop or home PC (transient exception, if used, documented and cleaned); zero Kaggle GPU-hours; Modal spend < $8.
  - [ ] Nothing public anywhere; `DATA_LICENSING.md` current.

**Then:** v0.3 "Clean" TODO (next session) — Kronos Algorithm 1 (`PartitionByPriceJumps`, illiquid/stagnant runs) with Table 4 thresholds wired per frequency, the split/dividend policy *applied* (tokenization series vs total-return eval series, powered by this version's adjustment audit), survivorship documentation, and the synthetic edge-case test suite (splits, gaps, flash crashes, limit-up/down, DST, holidays, rollover).
