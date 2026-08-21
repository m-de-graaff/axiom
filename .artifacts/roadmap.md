# Axiom — Model Line Roadmap: v0.0 → v1.0

**Status:** source public since 2026-08-21 (ADR-0017); data and model artifacts unpublished. Working title `axiom` (import name `axiom`; distribution name deferred — see Publish Gate).
**Owner:** Mark (`m-de-graaff`), solo, building with Claude Code.
**Scope of this roadmap:** V1 = the trained K-line foundation model delivered as `.safetensors`, plus everything required to produce it (data acquisition, cleaning, preprocessing contract, tokenizer, pre-tokenized corpus, AR decoder training, evaluation harness, export, local ROCm inference). Chart predictor, backtester, and the trading agent are **post-1.0 product lines** and intentionally absent here.
**Source of truth for requirements:** "Axiom V1: Component Inventory and Build Requirements on the Free-First Stack (2026)" (components C1–C11).

---

## 1. Operating constraints (fixed)

| Constraint | Rule |
|---|---|
| Laptop role | Dispatch + development only. No training, no corpus bytes, ever. |
| Home PC (RX 7900 XTX, ROCm, Linux) | Inference only. Receives exactly one thing: the final `.safetensors` + config. |
| Compute | Free-first: Kaggle (~30 GPU-h/wk, P100 or 2×T4, 12 h sessions; TPU v5e-8 ~20 h/wk, 9 h sessions), Modal ($30/mo credits, resets monthly), optional TRC (stretch). Paid fallback only by explicit decision: Vast.ai/RunPod RTX 4090 ≈ $0.30–0.34/h. |
| Storage | HF Hub free tier: 100 GB private. Optional Cloudflare R2 (10 GB, zero egress) as hot shard cache. |
| Publishing | **No PyPI and no public data or model artifacts until the Publish Gate (post-v1.0).** The source repo `m-de-graaff/axiom` went **public on 2026-08-21** to get unlimited Actions minutes (ADR-0017); the code was always publishable under `DATA_LICENSING.md`. `axiom-raw` stays private permanently. `axiom-tokenized` and `axiom-model` stay private until the gate. |
| Architecture (locked by ADR in v0.0) | Two-stage discretize-then-autoregress. Stage 1: BSQ tokenizer (vendored from Kronos, MIT) as hierarchy default; flat FSQ (`vector-quantize-pytorch`) as ablation. Stage 2: decoder-only AR, Kronos-small config (8L/512d/1024ff/8h, ~24.7 M params), ctx 512, coarse/fine dual head with sampled-coarse, + asset-class & frequency conditioning embeddings. GPU-only for v1.0; TPU/JAX is stretch. |
| Corpus (locked by ADR in v0.0) | **M0 (mandatory):** mixed-market 1h + 1d — Binance crypto spot + USDT-M futures, Dukascopy FX/commodities, Stooq/yfinance daily equities (~50 M clean bars). **M1 (stretch, decided at Gate G3):** + crypto 15m (+~50 M) and 5m (+~150 M) toward ~0.25–0.3 B bars. |
| Honesty banner | Expected OOS: directional accuracy 50–53 %, RankIC 0.00–0.04. Volatility is the genuinely forecastable target and the centerpiece of evaluation. The durable value is the reproducible tokenizer + quantizer comparison + honest eval harness. This banner goes verbatim into the model card. |

---

## 2. Version ladder (overview)

Durations are focused-effort estimates; calendar time for v0.5 and v0.7 is quota-bound (see §5).

| Version | Handle | Scope (components) | Acceptance gate | Est. effort |
|---|---|---|---|---|
| v0.0 | Spine & Loop | Repo skeleton, tooling, config/repro core, secrets, CI, **proven dispatch→checkpoint→kill→resume loop** (C1, C11) | **G1** | ~6 days |
| v0.1 | Schema & First Bars | Canonical bar schema, provenance manifests, Binance loader (CHECKSUM-verified), raw Parquet → HF private, cloud-to-cloud (C2 part, C3) | Exit checklist | ~1–1.5 wk |
| v0.2 | Corpus Breadth | Dukascopy FX/commodities loader, Stooq/yfinance equities (manual-assisted), corpus registry, `upload_large_folder` flow (C2 rest, C3) | Exit checklist | ~1–2 wk |
| v0.3 | Clean | Kronos Algorithm 1 + Table 4 (1h: 256/0.20/1/3; 1d: 128/0.30/1/3), split/dividend policy, edge-case + property tests, survivorship documentation (C4) | Exit checklist | ~1 wk |
| v0.4 | Contract | Versioned preprocessing contract: candle-geometry params, causal normalization, golden vectors, hypothesis tests, `schema_version=1` freeze (C5) | **G2** | ~1 wk |
| v0.5 | Tokenizer | BSQ default + FSQ ablation, tokenizer training on Kaggle, codebook health + reconstruction report, temporal firewall (freeze date) (C6) | **G3** | ~2 wk |
| v0.6 | Shards | Pre-tokenization map job (Modal), uint16 (coarse,fine) + time features + conditioning IDs, MDS shards on HF, deterministic mid-epoch resume verified (C7) | Exit checklist | ~1 wk |
| v0.7 | Decoder | Adapted Kronos AR (25 M) + conditioning, fp16/GradScaler on Kaggle, checkpoint/resume across real session kills, train to val-loss plateau (C8) | **G4** | ~3–4 wk (quota-bound) |
| v0.8 | Judgment | Eval harness: CRPS/pinball/PIT (`scoringrules`), per-date RankIC, vol MAE/R² vs GARCH (`arch`), baselines (persistence, hist-mean, GARCH, LightGBM, Chronos-2 zero-shot), purged/embargoed walk-forward, sealed hash-committed holdout (C9) | **G5** | ~2 wk |
| v0.9 | Artifact | `PyTorchModelHubMixin` export → `model.safetensors` + `config.json`, Predictor class on the causal contract, ROCm inference verified on 7900 XTX, model card draft, optional 102 M attempt if TRC landed (C10) | **G6** | ~1 wk |
| v1.0 | Freeze | API freeze, model card final, fresh-clone reproducibility pass, tagged release candidate — **still private** | Exit checklist | ~0.5–1 wk |

Total: ~13–17 focused weeks ≈ 3–4 calendar months, consistent with the research estimate.

---

## 3. Repo & account topology (the documentation you asked for)

Everything below lives under the personal account `m-de-graaff`. **The source repo is public; every data and model repo is private.** No data or model artifact becomes public before the Publish Gate. Each version's TODO must update `docs/REPOS.md` in the monorepo when a repo is created.

| Service | Repo | Visibility | Created in | Purpose |
|---|---|---|---|---|
| GitHub | `m-de-graaff/axiom` | **Public** (2026-08-21) | v0.0 | The monorepo. Cloud jobs `pip install` it via a read-only fine-grained PAT. (Alternative if you want zero GitHub: build a wheel and attach it as a private Kaggle Dataset — documented in the v0.0 TODO as fallback.) |
| HF (dataset) | `m-de-graaff/axiom-runs` | Private | v0.0 | Training checkpoints, run manifests, resume pointers (`latest.json`). |
| HF (space, optional) | `m-de-graaff/axiom-trackio` | Private | v0.0 | trackio experiment dashboard sync (trackio may auto-create its own backing dataset — accept and document). |
| HF (dataset) | `m-de-graaff/axiom-raw` | Private | v0.1 | Raw-cache tier: cleaned-source Parquet (zstd) + provenance manifests. Loader+manifest only; never redistributed. |
| HF (dataset) | `m-de-graaff/axiom-tokenized` | Private (public at Publish Gate) | v0.6 | Pre-tokenized MDS shards (uint16 pairs + time features + conditioning IDs), ~2–8 GB. |
| HF (model) | `m-de-graaff/axiom-model` | Private (public at Publish Gate) | v0.9 | `model.safetensors`, `config.json`, model card, revision tags per training run. |
| Kaggle | account (phone-verified) | — | v0.0 | Execution backend #1 (GPU). Secrets: `GH_PAT`, `HF_TOKEN`. |
| Modal | workspace | — | v0.0 | Execution backend #2 (data/map jobs). Secrets: `axiom-gh`, `axiom-hf`. |
| GCP + TRC | project (optional) | — | ≥ v0.6, only if pursuing 102 M | Stretch TPU track. Requires credit-card billing; apply ~2 weeks before intended AR scale-up. |

Token inventory (details + rotation policy live in `docs/RUNBOOK.md`): GitHub fine-grained PAT `axiom-kaggle-read` (repo `axiom`, Contents: read-only); HF fine-grained token `axiom-write` (write, scoped to `axiom-*` repos only); Kaggle API `kaggle.json` (laptop only); Modal token (laptop only).

---

## 4. Version details

### v0.0 — Spine & Loop
Repo skeleton (`uv`, src layout, Ruff, ty/mypy, pytest + hypothesis, pre-commit, CI), config system (pydantic-settings + YAML + config-hash logging), deterministic seeding, secrets wiring, ADRs locking the six open design decisions, and **the Loop**: one command on the laptop dispatches a dummy trainer to Kaggle (CPU — zero GPU quota) which clones the private repo, runs, checkpoints full state (model-equivalent payload + optimizer-equivalent + RNG) to `axiom-runs` every N steps, survives a deliberate mid-run kill, and resumes **bit-identically**. Modal runs the same job as backend #2. Detailed plan: `todo-v0.0.md`.
**Gate G1 (exit):** kill-and-resume produces a final state bit-identical to an uninterrupted run, on Kaggle, with checkpoints on HF; CI green; ADRs merged; `docs/REPOS.md` current. No market data touched; no GPU minutes spent.

### v0.1 — Schema & First Bars
Canonical bar schema (UTC epoch ms int64 + `exchange_tz` + `session_id`; OHLCVA; amount synthesis = volume × mean(OHLC) when absent; quote-vs-base volume recorded per source), provenance manifest format (source, instrument, pull date, URL, sha256, row count, ts range, adjustment policy, volume convention), Binance loader on `binance_historical_data` (monthly zips + daily tail, `.CHECKSUM` verified, spot + `um`), Parquet layout (partition `asset_class/frequency/symbol`, zstd, 128–512 k row groups), first cloud-to-cloud pull job (Modal) landing in `axiom-raw`.
**Gate:** ≥ 100 liquid Binance pairs × (1h + 1d) in `axiom-raw` with manifests; a re-pull is byte-identical or produces a documented manifest diff; zero bytes on laptop.

### v0.2 — Corpus Breadth
Dukascopy loader (`dukascopy-python` v4.x; FX majors/minors + commodities + index CFDs, 1h + 1d), equities daily via Stooq bulk (manual CAPTCHA download → upload) and yfinance gap-fill (rate-limit-aware, loader-only, adjustment factors captured), corpus registry (single queryable index over `axiom-raw`), `upload_large_folder` + resumable-upload flow from Kaggle/Modal, per-source redistribution classification recorded (everything loader+manifest).
**Gate:** corpus M0 assembled (~50 M raw bars target across 4 asset classes); registry answers "what do we have, from where, pulled when"; storage < 100 GB with headroom.

### v0.3 — Clean
`FilterLowQualitySegments` per Kronos Appendix B Algorithm 1: `PartitionByPriceJumps` on |open_t/close_{t−1} − 1| > θ, consecutive-illiquid and consecutive-stagnant runs, min-length; Table 4 thresholds wired per frequency (incl. 5m/15m rows for the M1 stretch). Split/dividend policy: split-adjusted OHLC for tokenization; separate total-return series for eval labels (Stooq-adjusted cross-checked against yfinance `auto_adjust`, discrepancies logged). Survivorship bias: accept-and-document (model-card limitation). Edge-case unit tests (splits, session gaps, flash crash, limit-up/down, DST, holidays, rollover) + hypothesis invariants (monotone ts, high ≥ max(o,c) ≥ min(o,c) ≥ low, volume ≥ 0).
**Gate:** all cleaning tests green; per-rule drop statistics reported per source/frequency and eyeballed sane.

### v0.4 — Contract
The versioned preprocessing contract, used identically by training pre-tokenization and the inference Predictor: candle-geometry parameterization (log h/o, log l/o, log c/o, gap log o_t/c_{t−1}) as default with plain per-field log-returns as A/B; volume/amount `log1p` + causal robust scaling (expanding median/IQR, strictly-past, per-asset with global cold-start fallback); clipping policy; NaN/missing policy; `schema_version` field; golden test vectors frozen in-repo; hypothesis property tests (train/inference identity, no NaN escape, clip bounds, causality — no future bar influences any feature).
**Gate G2 (exit):** golden vectors + property tests green in CI; contract frozen at `schema_version=1`; a causality audit test fails loudly if any future-window statistic sneaks in. **No tokenizer work may begin before G2.**

### v0.5 — Tokenizer
Vendor Kronos `BSQuantizer` + tokenizer encoder/decoder (MIT; add `NOTICE`), config d_model 256, 3+3 layers, 4 heads, d_ff 512, k=20 split 10 coarse/10 fine, β=0.05 γ₀=1.0 γ=1.1 ζ=0.05 λ=1 group_size=5; Huber reconstruction in contract space. Flat-FSQ ablation via `vector-quantize-pytorch`. Declare and hash-commit the **temporal firewall**: tokenizer train data strictly precedes the sealed test period. Train on Kaggle (fp16 with care around entropy terms; P100 first), log codebook utilization/perplexity/dead codes + per-feature reconstruction. Produce the BSQ-vs-FSQ report.
**Gate G3 (exit):** reconstruction error at or better than target on held-out pre-firewall data; no codebook collapse; coarse utilization > fine (hierarchy holds; Kronos-reported ~97.7 %/85.3 % is the aspiration, not the bar); BSQ-vs-FSQ report written. **G3 also decides corpus M1** (add 15m/5m crypto) and whether to file the TRC application. AR training may not start before G3.

### v0.6 — Shards
Freeze the winning tokenizer; Modal map job converts the cleaned corpus → uint16 (coarse, fine) pairs + time-feature stream (minute/hour/weekday/day/month) + conditioning IDs (asset-class, frequency) → MosaicML `streaming` (MDS) shards → `axiom-tokenized` (~2–8 GB). Verify: random-access window sampling, deterministic mid-epoch resume from an interrupted iterator, streaming throughput from HF on a Kaggle CPU kernel. litdata is the documented fallback if MDS-from-HF ergonomics disappoint.
**Gate:** a Kaggle kernel streams shards at ≥ target throughput; iterator resume determinism test green; shard checksums manifest committed.

### v0.7 — Decoder
Adapt Kronos `model/kronos.py` (MIT): HierarchicalEmbedding, TemporalEmbedding, RMSNorm + RoPE pre-LN blocks, DualHead + DependencyAwareLayer with sampled-coarse `torch.multinomial`. Add asset-class + frequency embeddings (added-embedding form, not prepended tokens). 25 M config (8L/512d/1024ff/8h), ctx 512. Schedule: AdamW, cosine, 15 k-step warmup from 10 % peak; **verify the provisional Table 5 cells (small: LR 1e-3, wd 0.01) against the Kronos PDF/finetune configs before locking** — a named task, not a footnote. fp16 + conservative GradScaler on T4/P100 (entropy-term instability watch), single-GPU P100 first, DDP only if throughput-bound. Real training with the v0.0 checkpoint discipline: full state (model, optimizer, scheduler, scaler, RNG, dataloader position) to `axiom-runs`; resume across ≥ 3 real 12 h-session kills. Budget ≈ 90–120 GPU-h across 3–4 weeks of quota.
**Gate G4:** val loss (per-subtoken perplexity) plateaus sensibly with no divergence; resume determinism verified on real interrupts; run manifest (config hash, data snapshot, tokenizer version) complete. Optional: TRC/102 M branch only if G4 passes early and TRC granted.

### v0.8 — Judgment
Eval harness on the sealed holdout (hash-committed in v0.5): CRPS/pinball/PIT calibration via `scoringrules` on sampled paths; per-date cross-sectional IC/RankIC on the daily crypto+equities universe; volatility MAE/R² vs GARCH(1,1) (`arch`). Baselines: naive persistence, historical mean, GARCH(1,1), LightGBM-on-features (the embarrassment baseline), Chronos-2 zero-shot (`chronos-forecasting`). Purged + embargoed walk-forward CV; leakage tripwires as automated asserts (tokenizer firewall respected; no per-window future stats; conditioning IDs carry no future info).
**Gate G5 (exit):** beats naive persistence on volatility MAE/R²; CRPS competitive with Chronos-2 zero-shot on the vol task; RankIC reported honestly whatever it is (0.00–0.04 expected). If the model loses to LightGBM everywhere, v1.0 reframes as tokenizer/representation study — that path is a legitimate v1.0, not a failure.

### v0.9 — Artifact
Export via `PyTorchModelHubMixin` (`save_pretrained` → `model.safetensors` + `config.json`; JSON-serializable init args) to `axiom-model` with a revision tag per run. Predictor class mirroring `KronosPredictor` but on the causal contract: load safetensors → preprocess (`schema_version=1`) → sample N paths → quantile fan. Local inference on the 7900 XTX: ROCm 7.x PyTorch wheels, SDPA, bf16, ctx 512; CPU-vs-ROCm output parity within tolerance. Model card draft: provenance, freeze dates, eval numbers, limitations (survivorship, corpus scale vs Kronos's 12 B bars, honesty banner), failure modes.
**Gate G6 (exit):** fresh environment on the home PC loads the model from `axiom-model` and reproduces reference outputs; end-to-end predict on live-ish bars works offline from the corpus.

### v1.0 — Freeze
API freeze (Predictor + contract + config schema), CHANGELOG complete, model card final, fresh-clone reproducibility pass (new machine: clone → `uv sync` → tests green → load model → golden predictions match), tag `v1.0.0`. **Still private.** The Publish Gate is a separate, explicit decision after this.

---

## 5. Quota & compute budget

| Version | Kaggle GPU-h | Modal credits | Notes |
|---|---|---|---|
| v0.0 | 0 | ~$1 | Loop test runs on CPU kernels deliberately. |
| v0.1–v0.4 | 0 | ~$5–15/mo | Pull/clean/contract jobs are CPU (Modal + Kaggle CPU kernels). |
| v0.5 | ~40–60 | ~$2 | Tokenizer training, 2 weeks of quota. |
| v0.6 | ~0–5 | ~$5–10 | Map job mostly Modal CPU; tokenizer forward is cheap. |
| v0.7 | ~90–120 | ~$2 | The long pole; 3–4 weeks of 30 h/wk quota. |
| v0.8 | ~10–20 | ~$2 | Sampling for eval + baselines. |
| v0.9–v1.0 | ~5 | ~$1 | Export verification. |

Total ≈ 150–200 GPU-h ≈ 5–7 weeks of Kaggle quota; $0 cash if Modal stays within monthly credits. Paid burst decision rule: if v0.7 stalls > 2 weeks on quota, a ~$30–40 Vast.ai/RunPod 4090 burst finishes it — explicit opt-in, never default.

---

## 6. Gates summary

- **G1** (v0.0): dispatch→checkpoint→kill→resume bit-identical on Kaggle; CI green; ADRs merged.
- **G2** (v0.4): contract frozen `schema_version=1`; golden vectors + causality audit green. Blocks tokenizer.
- **G3** (v0.5): tokenizer reconstruction + codebook health pass; BSQ-vs-FSQ report; decides corpus M1 + TRC filing. Blocks AR.
- **G4** (v0.7): healthy val-loss plateau + real-interrupt resume determinism.
- **G5** (v0.8): beats persistence on vol; CRPS competitive vs Chronos-2 zero-shot; honest RankIC published internally. Decides model-vs-representation-study framing for v1.0.
- **G6** (v0.9): home-PC ROCm reproduction of reference outputs.
- **Publish Gate** (post-v1.0, Mark's explicit call): live-check and claim a distribution name (`axiom` is squatted on PyPI; candidates `axiom-kline` / `axiom-fm` / `axiom-quant` — availability re-verified at that moment, accepted risk of interim squatting since we reserve nothing now); flip `axiom-tokenized` + `axiom-model` public; public GitHub; PyPI release; TRC acknowledgment if TRC compute was used.

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kaggle quota/policy changes | Med | High | Resume-first design everywhere; Modal as backend #2; paid-burst decision rule (§5). |
| Distribution name squatted before publish | Med | Low | Multiple candidates; name chosen at Publish Gate; import name `axiom` unaffected. |
| yfinance/Stooq breakage | High | Med | Equities are one of four asset classes; Binance+Dukascopy are the automated backbone; Stooq is manual-assisted by design. |
| HF 100 GB private ceiling (esp. M1 intraday raw) | Med | Med | Raw 5m/15m may be tokenize-in-flight (never persisted raw); prune raw tiers; R2 cache; HF storage grant at publish time. |
| fp16 instability with BSQ entropy terms on T4/P100 | Med | Med | Conservative GradScaler init; compute entropy terms in fp32; P100 single-GPU baseline first. |
| Hierarchical-FSQ temptation | Med | Med | ADR locks BSQ-default + flat-FSQ ablation; novel FSQ factorization only as post-1.0 research. |
| TRC/TPU port cost (multinomial on XLA) | High if attempted | Med | GPU-only v1.0 by ADR; TRC is stretch, gated at G3/G4. |
| Provisional Kronos hyperparameters wrong (Table 5, utilization figures) | Med | Med | Named verification task at v0.7 start; treat as targets, not gospel. |
| Eval shows no edge anywhere | High (by design honesty) | Low | G5 reframing path: tokenizer/representation study is a legitimate v1.0. |

---

## 8. Versioning & release policy

SemVer-style 0.x with git tags only (`v0.0.0` … `v1.0.0`), trunk-based development, Keep-a-Changelog `CHANGELOG.md` updated at every version close. No artifact leaves the private repos before the Publish Gate. Every training run records: config hash, git commit, contract `schema_version`, tokenizer version + firewall date, data snapshot manifest — so any number in the eventual model card is reproducible from a tag.

**Next document:** `todo-v0.0.md` (this drop). The v0.1 TODO follows in the next session, scoped exactly to §4/v0.1 above.
