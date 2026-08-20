# Axiom v0.1 — "Schema & First Bars" — Phased Development TODO

**Goal:** the canonical bar schema and provenance-manifest system exist as code with tests; a checksum-verified, resumable Binance loader runs **cloud-to-cloud on Modal** and lands ≥ 100 liquid pairs × (1h + 1d), spot + USDT-M perpetuals, as zstd Parquet with sidecar manifests in a new private HF dataset `m-de-graaff/axiom-raw`. The laptop never holds a single market-data byte.

**Starting state:** v0.0 complete, Gate **G1** passed (repo + CI + config/RNG core + proven dispatch→checkpoint→kill→resume loop on Kaggle and Modal).
**Exit gate:** roadmap §4/v0.1 exit checklist (expanded in Phase G below).
**Total budget:** ~6–7 focused days (≈ 1–1.5 calendar weeks). Modal spend target < $5 of the monthly $30. Kaggle GPU-hours: 0.

**Non-goals for v0.1 (scope fence):** no Dukascopy / Stooq / yfinance (v0.2); no corpus registry (v0.2 — per-file + per-pull manifests only); no cleaning/filtering (v0.3 — raw tier stays faithful, zero-volume and stagnant bars are *kept*); no COIN-M (`cm`) futures; no 1m/5m/15m frequencies; no tokenizer/model work; no GPU kernels; no R2; nothing public.

**Repos/services this version creates (documented in `docs/REPOS.md` in Phase E):**
- HF dataset `m-de-graaff/axiom-raw` (private) — raw-cache tier: cleaned-source Parquet + manifests. Loader-and-manifest policy: these bytes are a private reproducibility cache and are never redistributed.

---

## Phase A — Design specs & symbol universe (budget: 1 day)

### A1. ADRs (extend the v0.0 set)
- [ ] `docs/adr/0009-bar-schema-and-raw-tier.md` — canonical bar schema v1 (fields, units, conventions below), raw-tier layout, "raw = faithful" rule (no filtering, no imputation, no dedup beyond exact-duplicate `ts` resolution at the monthly/daily seam), Parquet settings (zstd, row-group 131 072), identity carried by path + sidecar manifest + Parquet key-value metadata (not columns).
- [ ] `docs/adr/0010-universe-selection-v1.md` — deterministic, pinned universe policy (A3): selection month, ranking metric (summed `quote_asset_volume` of that month's 1d bars), top-N counts, exclusion lists (leveraged tokens `*UP/*DOWN/*BULL/*BEAR`, stable-stable pairs: `USDCUSDT`, `TUSDUSDT`, `FDUSDUSDT`, `DAIUSDT`, `BUSDUSDT`, `EURUSDT`-style fiat-stables kept? → decide and list), min-history requirement (≥ 12 months of 1h) for a pair to count toward the ≥ 100 gate.
- [ ] `docs/adr/0011-binance-fetcher.md` — **decision: thin custom fetcher** over the documented `data.binance.vision` URL scheme (full control of CHECKSUM verification, layout, manifests, resume), with `binance_historical_data` retained as an independent cross-check tool only (Phase G3). Record the alternative (`binance-data-loader`, official `binance-public-data` scripts) and why not.

### A2. Bar schema v1 (spec, written into ADR-0009 before any code)
- [ ] Columns (Parquet): `ts` int64 — **bar open time, UTC, milliseconds**; `open, high, low, close` float64; `volume` float64 — **base-asset** volume; `amount` float64 — **quote-asset** volume (Binance `quote_asset_volume` used natively; the Kronos synthesis rule `amount = volume × mean(OHLC)` is reserved for sources without a native amount and must be flagged in the manifest when applied).
- [ ] Optional retained raw columns (nullable): `n_trades` int64, `taker_buy_volume` float64, `taker_buy_quote_volume` float64. Downstream consumes OHLCVA only; raw keeps these because storage is cheap and taker flow may matter post-1.0.
- [ ] Dropped source columns: `close_time`, `ignore`.
- [ ] Conventions: 1d bars open at 00:00 UTC; `exchange_tz = "UTC"`, `session_id = "24x7"` for all Binance series (fields live in the manifest + Parquet metadata, not as columns; they become real columns when session-bound markets arrive in v0.2).
- [ ] Parquet key-value metadata per file: `axiom_schema_version=1`, `source`, `asset_class`, `market` (`spot`|`um`), `symbol`, `frequency`, `manifest_sha256`.
- [ ] Invariants (enforced at parse time, violations fail the file): strictly increasing `ts`; `high ≥ max(open, close)`; `low ≤ min(open, close)`; `high ≥ low`; `volume ≥ 0`; `amount ≥ 0`; no nulls in OHLCVA; `ts` grid-aligned to the frequency (multiples of 3 600 000 ms for 1h; 00:00 UTC for 1d). Gaps are allowed (recorded, not filled).

### A3. Universe builder (deterministic, cloud-run)
- [ ] `src/axiom/universe/binance.py`: (1) enumerate symbols via the S3 XML listing (`https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/spot/monthly/klines/`, paginate with `marker`; same with `prefix=data/futures/um/monthly/klines/`); (2) filter to `*USDT` quote + exclusions; (3) for each candidate, download only the **selection month's 1d monthly zip** (tiny), sum `quote_asset_volume`; (4) rank; (5) emit `configs/universe_v1.yaml`.
- [ ] `configs/universe_v1.yaml` contents: `selection_month`, criteria + exclusions (echoed), `spot: top 200`, `um: top 100` symbol lists, min-history rule, and a `universe_hash` (config-hash of the criteria + lists). Committed to git — the universe is code.
- [ ] CLI: `axiom universe build --market spot --market um --month <YYYY-MM> --out configs/universe_v1.yaml` (runs on Modal via a thin wrapper; laptop invocation only dispatches).
- [ ] Unit tests with a faked listing + synthetic 1d zips: ranking is deterministic, exclusions applied, hash stable.

**Phase A acceptance:** ADRs 0009–0011 merged; `universe_v1.yaml` generated in the cloud, reviewed, committed; ≥ 150 spot + ≥ 60 um symbols pass the min-history rule on paper (headroom over the ≥ 100 gate).

---

## Phase B — Schema implementation (budget: 1 day)

### B1. `src/axiom/schema/bars.py`
- [ ] `BARS_SCHEMA_V1`: the `pyarrow.Schema` for A2 (fields, nullability) + helper `bars_metadata(...) -> dict` for the key-value metadata block.
- [ ] `validate_bars(table: pa.Table, frequency: str) -> ValidationReport`: every A2 invariant as a vectorized check; report carries per-invariant violation counts and first-offending row; `raise_on_error=True` path for the loader.
- [ ] Timestamp normalization helper: accept source epoch in ms **or** µs, detect by magnitude (µs if `ts > 10^14`), normalize to ms int64. (Binance vision has shifted timestamp units on some datasets; never trust, always detect.)
- [ ] Grid-alignment helper per frequency (`1h`, `1d`), UTC-only.

### B2. Tests (synthetic only — no real market data on the laptop, ever)
- [ ] Hand-built golden fixture: 10 valid bars → passes; each invariant violated once in 10 mutated fixtures → each fails with the right code.
- [ ] Hypothesis strategies generating random valid bar tables (respecting invariants by construction) → `validate_bars` always passes; random single-field corruption → always caught.
- [ ] µs-vs-ms detection: same logical series in both units normalizes to identical output.
- [ ] Round-trip: table → Parquet (zstd, row-group 131 072, metadata) → read → schema, metadata, and values identical.

**Phase B acceptance:** `just check` green including new tests; schema module has zero I/O or network code (pure).

---

## Phase C — Provenance manifests (budget: 0.5 day)

### C1. `src/axiom/provenance/manifest.py`
- [ ] `FileManifest` (pydantic, `extra="forbid"`): `schema_version`, `source` (`binance_vision`), `market`, `asset_class` (`crypto`), `symbol`, `frequency`, `pull_run_id`, `pulled_at` (UTC ISO), `source_urls` (list of every zip consumed), `source_sha256s` (from the `.CHECKSUM` files, verified), `artifact_path` (path in `axiom-raw`), `artifact_sha256`, `row_count`, `first_ts`, `last_ts`, `gap_count`, `volume_convention` (`base+quote_native`), `amount_synthesized` (bool, false for Binance), `adjustment_policy` (`none`), `loader_version` (package version + git commit), `universe_hash`.
- [ ] Canonical JSON serialization (sorted keys) + `manifest_sha256`; writer emits the sidecar `*.parquet.manifest.json` **adjacent to the Parquet file**.
- [ ] `PullRunManifest`: `pull_run_id`, config hash, universe hash, started/finished, counts (files ok / skipped / failed), failure list, total rows, total bytes — one per pull run under `manifests/pulls/`.
- [ ] Idempotence primitive: `is_current(remote_sidecar, expected_source_sha256s) -> bool` — the loader's skip test.

### C2. Tests
- [ ] Serialization round-trip byte-stable; hash changes iff a substantive field changes; `pulled_at`/`pull_run_id` excluded from the idempotence comparison.

**Phase C acceptance:** manifests are pure data + hashing, fully unit-tested, no network.

---

## Phase D — Binance loader (budget: 2 days)

### D1. Fetch layer (`src/axiom/sources/binance_vision.py`)
- [ ] URL builders for the four families: spot/monthly `data/spot/monthly/klines/{S}/{f}/{S}-{f}-{YYYY-MM}.zip`, spot/daily `.../daily/klines/{S}/{f}/{S}-{f}-{YYYY-MM-DD}.zip`, um/monthly `data/futures/um/monthly/klines/...`, um/daily — each with its `.CHECKSUM` sibling.
- [ ] Month enumeration per symbol from the S3 XML listing (authoritative, handles listing/delisting gaps); daily-tail enumeration = days after the last complete month.
- [ ] `httpx` client: timeout, retry with exponential backoff + jitter on 429/5xx/connection errors, global concurrency semaphore (≤ 12), streaming download to container-local tmp, **404 on a listed month = hard error; 404 on an unlisted probe = expected**.
- [ ] CHECKSUM verification: parse `<sha256>  <filename>`, verify the zip **before extraction**; mismatch → delete, retry once, then fail the file loudly.

### D2. Parse layer
- [ ] Zip → CSV via `pyarrow.csv` with an explicit 12-column schema: `open_time, open, high, low, close, volume, close_time, quote_asset_volume, n_trades, taker_buy_volume, taker_buy_quote_volume, ignore`.
- [ ] **Header auto-detect** (Binance files exist both with and without header rows) — sniff first line, skip if non-numeric.
- [ ] Timestamp unit detection + normalization (B1 helper) on `open_time` → `ts`.
- [ ] Column mapping to schema v1; drop `close_time`, `ignore`; cast float64; sort by `ts`.
- [ ] **Monthly/daily seam dedup:** exact-duplicate `ts` across the last monthly file and daily tail → assert value-equality on the overlap (mismatch = fail file), keep one.
- [ ] `validate_bars(..., raise_on_error=True)`; compute `gap_count` (missing grid steps) for the manifest — gaps recorded, never filled.

### D3. Write + upload layer
- [ ] One Parquet per (market, frequency, symbol): `raw/binance/{spot|um}/{1h|1d}/{SYMBOL}.parquet`, zstd, row-group 131 072, metadata block from B1, plus the C1 sidecar manifest.
- [ ] Upload to `m-de-graaff/axiom-raw` via `HfApi.upload_folder` in batched commits (≤ ~50 files/commit to stay friendly to Hub rate limits; `run_as_future` + bounded queue; final barrier). Full-corpus re-syncs use `upload_large_folder`.
- [ ] Idempotence: before processing a symbol, fetch its remote sidecar (if any) and `is_current(...)` against the enumerated source checksums → skip completed work. **This is the resume mechanism** — the pull job has no other checkpoint state.

### D4. Orchestration + CLI
- [ ] `pull_symbol(cfg, market, symbol, frequency) -> FileManifest | Skip | Failure` — the unit of work; pure function of (universe, remote state, source).
- [ ] `axiom pull binance --universe configs/universe_v1.yaml --frequencies 1h,1d --markets spot,um [--limit N] [--symbols BTCUSDT,ETHUSDT]` — builds the work list, runs it, writes the `PullRunManifest`.
- [ ] `--limit`/`--symbols` exist for smoke runs only; the flags are recorded in the pull manifest so partial pulls are never mistaken for full ones.

### D5. Tests (all offline; fixtures are synthetic zips built in-test)
- [ ] Fetch layer against a monkeypatched transport: retry/backoff, checksum-mismatch path, 404 semantics, concurrency cap honored.
- [ ] Parse: header/headerless, ms/µs, seam dedup (equal + conflicting overlap), out-of-order rows sorted, scientific-notation floats, empty file → clean failure.
- [ ] End-to-end `pull_symbol` against a fake source + local `axiom-raw` stand-in dir: produces Parquet + sidecar; second run skips; corrupting the remote sidecar forces a re-pull.

**Phase D acceptance:** loader wholly unit-tested offline; CI green; no live network in tests.

---

## Phase E — `axiom-raw` + Modal pull job (budget: 1 day)

### E1. Repo creation
- [ ] `create_repo("m-de-graaff/axiom-raw", repo_type="dataset", private=True)`; seed `README.md` inside the dataset (private-facing: layout, schema version, loader-and-manifest policy, "never redistributed").
- [ ] Update `docs/REPOS.md` (visibility, purpose, created-in v0.1) — same-day rule from v0.0 stands.

### E2. `remote/modal/pull_binance.py`
- [ ] Modal app `axiom-pull`: image = slim + `pip install git+https://…axiom.git@main` (build-time `axiom-gh` secret), runtime `axiom-hf` secret; CPU-only.
- [ ] Driver function builds the work list, then `.map()` over symbols with `max_containers` ≈ 10 (politeness cap is the client semaphore; container cap bounds cost), per-symbol timeout 15 min, Modal retries = 1 (idempotence makes retries safe).
- [ ] Driver aggregates results → uploads `manifests/pulls/{pull_run_id}.json`.
- [ ] `just pull-binance` recipe wrapping `modal run remote/modal/pull_binance.py -- --universe configs/universe_v1.yaml`.

### E3. Smoke → full
- [ ] Smoke: `--symbols BTCUSDT,ETHUSDT --markets spot --frequencies 1h,1d` → verify files + sidecars on HF, spot-check values against the Binance web UI chart for one date (eyeball sanity, note in RUNBOOK).
- [ ] **Kill drill:** launch a `--limit 40` run, terminate the Modal app mid-flight, relaunch → completed symbols skip via sidecar idempotence, run finishes; pull manifest shows correct ok/skipped split.
- [ ] Full pull: entire `universe_v1.yaml`, both markets, 1h + 1d. Expected order of magnitude: ~250–300 series/frequency, ~15 M bars total, ~0.5–1 GB in `axiom-raw`, a few $ of Modal credit, well under the HF 100 GB tier.

**Phase E acceptance:** full pull completes; `PullRunManifest` shows 0 unexplained failures (delisted/short-history symbols may legitimately fail min-history and are listed as such); laptop `git status` clean and no local Parquet/zips outside `/tmp` of cloud containers.

---

## Phase F — Verification & data QA (budget: 0.5–1 day)

### F1. Re-pull reproducibility (the roadmap's headline gate condition)
- [ ] `axiom raw verify --sample 10` (runs on Modal): for 10 random (symbol, frequency) picks, re-download all **monthly** source zips, re-verify CHECKSUMs, re-parse, re-write Parquet in-memory → compare `artifact_sha256` against the stored sidecar. Monthly-derived content must be **byte-identical**.
- [ ] Daily-tail divergence (new days since the pull) is expected: verify tool reports it as a *documented manifest diff* (old vs new `last_ts`/`row_count`), not a failure. Wire both outcomes into the report format.

### F2. QA report
- [ ] `axiom raw stats` (Modal): per market × frequency — series count, total rows, min/median/max history length, gap-count distribution, top-10 gappiest symbols, invariant-violation count (must be 0 by construction), storage bytes. Output: markdown report committed to `docs/reports/v0.1-raw-qa.md` + logged to trackio as a table/summary.
- [ ] Eyeball pass recorded in the report: BTCUSDT/ETHUSDT 1d row counts match listing age; 24/7 grid continuity for majors (gap_count ≈ 0); um perps start dates plausible.

**Phase F acceptance:** verify-sample = 10/10 byte-identical on monthly content; QA report committed and sane.

---

## Phase G — Docs, cross-check, tag, exit review (budget: 0.5 day)

- [ ] **G1 doc pass:** `docs/ARCHITECTURE.md` marks C3 partially delivered (schema + raw tier) and C2 partially delivered (Binance); `docs/RUNBOOK.md` gains: running a pull, reading a pull manifest, the kill/resume story for pulls, token use by the Modal image build.
- [ ] **G2 CHANGELOG + tag:** move `[Unreleased]` → `[0.1.0] - <date>`; `git tag v0.1.0`; push.
- [ ] **G3 independent cross-check:** on Modal, run `binance_historical_data` for 3 symbols × 1h and diff row counts + a sampled day of OHLCV values against `axiom-raw` — record agreement (or explained differences, e.g., timestamp conventions) in the QA report. This is the ADR-0011 safety net.
- [ ] **v0.1 exit checklist** (all must be true):
  - [ ] ≥ 100 pairs meeting the min-history rule present in `axiom-raw` at **both** 1h and 1d, each with a valid sidecar manifest (`ok_pairs_1h ∩ ok_pairs_1d ≥ 100`).
  - [ ] `universe_v1.yaml` committed with hash; every artifact manifest references it.
  - [ ] Re-pull sample: 10/10 monthly byte-identity; daily-tail diffs reported as manifest diffs.
  - [ ] Pull kill-drill passed (idempotent resume via sidecars).
  - [ ] All loader/schema/manifest tests green in CI; no live network in CI.
  - [ ] Cross-check vs `binance_historical_data` recorded.
  - [ ] QA report committed; invariant violations = 0; storage < 2 GB.
  - [ ] Zero market-data bytes on laptop or home PC; zero Kaggle GPU-hours; Modal spend < $5.
  - [ ] `docs/REPOS.md` lists `axiom-raw` (private); nothing public anywhere.

**Then:** v0.2 "Corpus Breadth" TODO (next session) — Dukascopy FX/commodities loader, Stooq bulk (manual-assisted) + yfinance gap-fill for daily equities with adjustment factors, session-bound schema fields (`exchange_tz`, `session_id`) promoted to real columns, and the corpus registry over `axiom-raw`.
