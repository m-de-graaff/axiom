# ADR-0019: Adjustment policy — what tokenization consumes, and where total return comes from

**Status:** Accepted (v0.3)

## Context

Equities are the only asset class in the corpus with corporate actions. Two questions follow, and
they have different answers: **which price series does the tokenizer see**, and **which price
series does an evaluation label come from**. Conflating them is how a backtest quietly measures
the wrong thing.

v0.2 measured what Stooq actually publishes rather than assuming it. The verdict is recorded in
`docs/reports/v0.2-adjustment-audit.md` and this ADR branches on it.

## The verdict, verbatim

> Run 2026-08-21. **Verdict: `split_and_dividend_adjusted`.**
>
> All 3 split probes show no discontinuity, and across 20 sampled tickers the median relative
> difference against Yahoo's dividend-adjusted closes is 0.0129, inside the 2% agreement
> threshold. The series track a total-return path, so they are split *and* dividend adjusted.

AAPL across its 4:1 split shows a close ratio of 0.9672, NVDA across 10:1 shows 0.9926, TSLA
across 3:1 shows 1.0035. An unadjusted series would show 0.25, 0.10 and 0.33.

This **inverts** the assumption v0.3 was planned under. The plan expected a price path that a
total-return series had to be built from. The measurement says the vendor already ships the
total-return path.

## Decision

### Tokenization consumes the vendor-adjusted Stooq OHLC as-is

No split handling in the cleaning pass — there are no unadjusted splits to partition on. No
dividend correction applied to tokenizer inputs.

Because the series is dividend-adjusted, the **ex-dividend gap problem does not arise** for this
corpus: the systematic small overnight drop on ex-date has already been removed by the vendor.
That is the opposite of the usual free-data situation and it is worth saying plainly, because the
standard caveat ("Kronos ignores ex-dividend gaps and so do we") does not apply to us in the
direction people expect.

What *is* true instead: the tokenizer sees a total-return path, so its notion of "price" for US
equities is not the price anybody traded at. Every historical bar is restated whenever a new
dividend or split lands. Two consequences, both recorded:

- **The series is not stable across pulls.** A re-pull of the same ticker legitimately changes old
  bars. The manifest hash catches it and the staleness guard re-cleans; that is the mechanism
  working, not a fault.
- **A price-path series would have to be derived** if anything downstream ever needs the actual
  traded price (order-book realism, a trading agent). Nothing in v0.3–v1.0 does, so it is not
  built. Named here so the absence is a decision rather than an oversight.

### `tr_close` is an identity under this verdict

The total-return close used for evaluation labels is defined by one interface with two branches,
selected by the recorded verdict:

| Verdict | `tr_close` |
|---|---|
| `split_and_dividend_adjusted` | `tr_close = close` — the vendor series already is the TR path |
| `split_adjusted` | `tr_t = tr_{t−1} × (close_t + div_t) / close_{t−1}`, anchored `tr_first = close_first` |
| `none` (crypto, FX, commodities) | `tr_close = close` — no corporate actions exist |

Both branches are implemented and unit-tested against exact arithmetic. The dividend-accumulation
branch is not dead code kept for symmetry: the verdict is a *measurement*, re-run on any future
Stooq pull, and a vendor that changes convention flips the branch without a rewrite.

### Under an identity verdict, nothing is materialized

The plan called for `derived/tr_close/us/1d/{first_char}/{TICKER}.parquet` for the equities tier.
Under the measured verdict that file would hold a byte-for-byte copy of the `close` column of the
raw file next to it, for roughly twelve thousand tickers.

That is the same duplication the plan already refuses for crypto ("don't duplicate 20 M rows to
store a copy of `close`"), and the verdict extends the refusal to equities. So:

- `axiom derive tr` writes **`derived/tr_close/manifest.json` only**: the verdict, the policy per
  source, per-ticker `tr_available`, and the coverage numbers. No bar files.
- `tr_close` is resolved **at read time** through `axiom.adjust.policy.tr_close`, uniformly for
  every source. Downstream calls one function and does not branch on asset class.
- If a future audit returns `split_adjusted`, the same command materializes the letter-sharded
  Parquet tier, because then the series genuinely differs from `close` and computing it per read
  would be both slow and non-reproducible.

The interface downstream sees is uniform either way. What changes is whether the answer is stored
or computed, and that is an implementation detail behind one function.

### Tickers without dividend-event coverage

`tr_available = false` in the derived manifest for any ticker where the events needed by the
active branch are missing. Under the identity verdict this is empty by construction — no events
are needed — and the field is still written, because the eval-slice definitions in v0.8 read it
and must not have to know which branch produced it.

A ticker is **never silently approximated**. If a future branch needs dividends and yfinance has
none for a name, that name is excluded from dividend-sensitive eval slices and says so.

## Consequences

- v0.3's Phase D is small, and that is the measurement's doing rather than a corner cut.
- v0.8's eval labels for US equities are total-return by construction, with no reconstruction step
  and no reconstruction error.
- The corpus carries a documented instability: adjusted history is restated by later corporate
  actions. The raw-file hash is the tripwire.
- Survivorship applies to the verdict itself. The audit sampled tickers that still exist; it says
  what Stooq did to survivors and nothing about the delisted (ADR-0016). Inherited by everything
  downstream and repeated in `LIMITATIONS.md`.


---

## Amendment, 2026-08-21 — the verdict is stamped into the sidecars

All 12,425 Stooq sidecars recorded `adjustment_policy: vendor_adjusted_unverified`, which is what
the loader honestly believed at pull time: the audit had not run yet. It ran afterwards. Leaving
the sidecars alone meant the first thing a reader of `axiom-raw` saw was "unverified" for a corpus
that had in fact been verified.

Correcting the field in place is not possible. `adjustment_policy` is inside `manifest_sha256`,
which is stamped into every Parquet's own key-value metadata, so editing it breaks the
file-to-sidecar link on twelve thousand artifacts and turns a label fix into a re-pull — which
would also change `artifact_sha256` and invalidate the entire segment index.

**So the verdict goes in a second field, `adjustment_policy_verified`, held outside the identity
hash.** `VOLATILE_MANIFEST_FIELDS` already exists for exactly this category: things that describe
the run rather than the bytes. A measurement made after the pull belongs there — the same file
with and without a verdict written against it is the same file. `adjustment_policy` itself stays
*in* the hash, because that one is a property of the bytes: a split-adjusted file and an
unadjusted one are different data.

Two fields rather than one correction, because they are two different facts and both are worth
keeping: what was known when the file was written, and what was established afterwards.

`axiom raw stamp-verdict --source stooq` writes it, batched into about seven Hub commits. Nothing
but sidecars is touched, no `artifact_sha256` moves, and `clean/v1/` stays valid. It is
idempotent, so a partial run costs only what it did not reach, and a test asserts the property the
whole approach rests on: stamping does not move `manifest_sha256`.

`axiom derive tr` reads the stamped verdict when it is there and falls back to `RECORDED_POLICY`
when it is not, saying which happened either way.
