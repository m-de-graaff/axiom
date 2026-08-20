# Architecture

Two stages. Stage 1 turns continuous OHLCV bars into discrete tokens. Stage 2 is a decoder-only
transformer that autoregresses over those tokens. Everything else in this repo exists to get data
to stage 1 and to judge what comes out of stage 2 honestly.

```
raw bars → clean → contract (schema_version=1) → BSQ tokenizer → MDS shards → AR decoder → sampled paths → eval
   C2/C3     C4              C5                       C6            C7           C8            C10        C9
```

## Components and the version that delivers each

| | Component | Delivered in | State |
|---|---|---|---|
| C1 | Repo, tooling, config and reproducibility core, dispatch loop | v0.0 | **Built** |
| C2 | Data acquisition: Binance, Dukascopy, Stooq, yfinance | v0.1–v0.2 | **Built** — all four loaders, complete for v1.0 scope |
| C3 | Storage: Parquet layout, provenance manifests, corpus registry | v0.1–v0.2 | **Built** — schema, layout, manifests and registry |
| C4 | Cleaning: Kronos Algorithm 1, Table 4 thresholds, split/dividend policy | v0.3 | Not started |
| C5 | Preprocessing contract: candle geometry, causal normalization, golden vectors | v0.4 | Not started |
| C6 | Tokenizer: BSQ default, flat FSQ ablation, temporal firewall | v0.5 | Not started |
| C7 | Pre-tokenization: uint16 token pairs, time features, conditioning IDs, MDS shards | v0.6 | Not started |
| C8 | AR decoder: adapted Kronos, 25 M params, dual head, conditioning embeddings | v0.7 | Not started |
| C9 | Evaluation: CRPS, PIT, RankIC, volatility vs GARCH, five baselines | v0.8 | Not started |
| C10 | Export and inference: safetensors, Predictor, ROCm on the 7900 XTX | v0.9 | Not started |
| C11 | Checkpoint, resume, and run manifests | v0.0 | **Built** |

## What v0.0 built

C1 and C11, and nothing else. No market data was touched and no GPU minutes were spent — both are
hard non-goals of this version, not merely things that did not happen.

| Module | Does |
|---|---|
| `axiom.config.settings` | Environment settings and the run config, with unknown keys rejected |
| `axiom.config.hashing` | The config hash: the identity of an experiment, ignoring which run produced it |
| `axiom.ops.seeding` | Seeding, and RNG capture/restore across a process boundary |
| `axiom.ops.checkpoint` | Atomic local checkpoint write, sha256-verified read, keep-last-K pruning |
| `axiom.ops.hub` | Push to and pull from the private `axiom-runs` dataset, with `latest.json` as the resume pointer |
| `axiom.ops.logx` | Logging, run provenance, and best-effort trackio |
| `axiom.loop.dummy_trainer` | A stand-in trainer that exercises all of the above |
| `axiom.cli` | One entry point every backend calls identically |

**The design rule that makes v0.0 worth doing:** the loop code may not know it is a dummy. What
v0.5 and v0.7 replace is `_step` and the payload inside `TrainState`. The checkpoint writer, the
resume path, the config-hash guard, and both dispatch backends are used unchanged.

## What v0.2 built

The rest of C2 and the corpus-wide half of C3. Three things are worth knowing before reading the
modules.

**One driver, four sources.** `axiom.sources.base` owns everything true of every pull — the skip
test, validation, the Parquet write, the sidecar, the run manifest, the per-item blast wall — and
a source supplies four methods: `plan`, `build`, `manifest_extras`, `artifact_path`. Retry and
connection concurrency deliberately stay in each source's own transport, because they differ in
kind across an S3 bucket, a broker library with its own retry loop, and a single archive URL.

**The skip test is one list comparison, everywhere.** `is_current` compares the source
identifiers and digests a source reports now against the ones its sidecar records. Binance has
vendor checksums; Dukascopy has none, so a calendar year that has *ended* gets a constant token
and the current year's token carries the run's as-of date; Stooq's whole dump is one archive, so
every ticker in a run shares its digest. Three very different sources, no second skip mechanism.

**The registry is a cache with no authority.** Sidecars stay the truth. `axiom registry build`
reduces them to one table so questions are cheap, and rebuilding from an unchanged tier
reproduces the same `registry_hash`. A sidecar that will not parse is reported, never dropped.

| Module | Does |
|---|---|
| `axiom.sources.base` | The `Source` protocol, `WorkItem`, `SourcePlan`, the shared pull driver, letter sharding |
| `axiom.sources.dukascopy` | FX and commodities, year-chunked, prior years immutable |
| `axiom.sources.stooq` | The US daily bulk archive: member classification, parse tolerances |
| `axiom.sources.yahoo_events` | Split and dividend events; its own small loop, because events are not bars |
| `axiom.registry` | Registry build, its hash, and the canned coverage/storage/gap/staleness reports |
| `axiom.raw.crosscheck` | Stooq versus Yahoo adjusted closes, for the adjustment audit |
| `axiom.universe.dukascopy` | The 27 hand-pinned instruments and their measured start dates |
| `axiom.universe.equities` | The data-driven equities universe: registry filter, then dollar-volume ranking |

Next up is C4, the cleaning pass, in v0.3.

## What v0.1 built

The Binance half of C2, and the per-file half of C3. The corpus registry — one queryable index
over everything in `axiom-raw` — is v0.2, because a registry over one source is a list.

| Module | Does |
|---|---|
| `axiom.schema.bars` | Bar schema v1, its invariants, timestamp-unit detection, gap counting |
| `axiom.provenance.manifest` | Sidecar and pull-run manifests, canonical serialization, the idempotence test |
| `axiom.raw.store` | Where artifacts land: a local directory or the private `axiom-raw` dataset |
| `axiom.sources.binance_vision` | URLs, S3 enumeration, retrying downloads, checksum verification |
| `axiom.sources.binance_klines` | Zip to bar table: header sniffing, unit detection, seam resolution |
| `axiom.sources.binance` | `pull_symbol`, the work list, and the run manifest |
| `axiom.universe.binance` | The pinned universe: exclusions, ranking, and its hash |

**The design rule that makes v0.1 worth doing:** the pull has no checkpoint. `pull_symbol` is a
function of the universe, what the bucket publishes, and what the raw tier already holds, so a
killed run resumes by asking the same three questions again. There is no cursor to corrupt and
no progress file to go stale — the same discipline as v0.0's `latest.json`, arrived at from the
other direction.

## The three seams that later versions plug into

**`TrainState`** (`axiom/ops/checkpoint.py`) is the unit of resumability. v0.0 stores a scalar
where v0.7 stores model and optimizer tensors. Everything around it — atomic write, hash
verification, pruning, Hub transport — is agnostic to what is inside.

**`_step`** (`axiom/loop/dummy_trainer.py`) is the only function that knows what training means.

**`BARS_SCHEMA_V1`** (`axiom/schema/bars.py`) is the shape every source is translated into. v0.2
adds Dukascopy and Stooq by writing a parser that produces this table, not by widening the schema.
`exchange_tz` and `session_id` stay metadata: v0.2's session-bound markets make them vary between
files but not within one, so a column would repeat per row what the path already fixes (ADR-0014,
superseding the v0.1 plan to promote them).

**`schema_version`** appears on both the config and the checkpoint. It is 0 through v0.3 and
freezes at 1 in v0.4 when the preprocessing contract is locked (ADR-0005). A checkpoint carrying
a different `schema_version` than the running code is a refusal, not a warning.

## Guards that already exist

These are cheap now and expensive to retrofit:

- **Config-hash mismatch on resume is fatal.** Resuming a different experiment into an existing
  run would blend two things and report one number.
- **Checkpoint sha256 is verified on every read.** A truncated upload resuming into garbage is
  the one failure that would not announce itself.
- **Atomic writes.** A kernel killed mid-write leaves the previous checkpoint or a complete new
  one, never a half file.
- **Tracking failures never take the run down.** A 12-hour session lost to an unreachable metrics
  endpoint is 12 hours of quota gone.
- **Source archives are checksum-verified before extraction.** Corrupt bytes never reach a parser,
  and a second corrupt download fails the file loudly rather than caching it.
- **Bar invariants are enforced at parse time.** A file that breaks one is refused, so nothing
  downstream has to defend against a high below its own open.
- **A monthly/daily overlap must agree.** Two source files that disagree about the same bar fail
  the series rather than being silently reconciled.

## Where the honest numbers come from

Every run logs its config hash, git commit, package version, Python and torch versions, and
backend tag before the first step (`axiom.ops.logx.run_provenance`). From v0.5 the run manifest
also carries the tokenizer version and firewall date, and from v0.6 the data snapshot manifest.
That chain is what makes any number in the eventual model card reproducible from a git tag rather
than from memory.
