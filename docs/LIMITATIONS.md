# Limitations

What is wrong with this corpus and this model, measured where measuring is possible and stated
plainly where it is not. Started in v0.3; every version adds to it. The v0.9 model card is built
from this file, so a limitation that is not written down here does not reach the people who use
the model.

Nothing here is an apology. A known bias with a number beside it is a usable model; an unknown
one is a liability.

---

## 1. Survivorship: the equities tier contains no dead companies

**Measured 2026-08-21, over the 12,425 Stooq series in `axiom-raw`.**

| Question | Answer |
|---|---|
| Earliest bar anywhere in the tier | **1962-01-02** |
| Earliest *last* bar in the tier | **2026-02-06** |
| Series whose last bar predates the pull date by more than 30 days | 143 (**1.15 %**) |
| … by more than 90 days | 42 (**0.34 %**) |
| … by more than 365 days | **0** |

That last row is the finding. The archive spans sixty-four years of US equity history and **not
one of its twelve thousand tickers stopped trading more than a year before the pull date.** The
real market does not behave that way: US listings die at several percent a year through mergers,
bankruptcies, buyouts and delistings, so a sixty-four-year sample of everything that ever traded
would be dominated by names that no longer exist.

The Stooq bulk archive is therefore not a sample of the market's history. It is a snapshot of the
**currently listed** market, with each survivor's history extended backwards. Every equity number
this project produces — the adjustment audit, the drop statistics, the v0.8 evaluation — is
conditioned on the instrument having survived to 2026.

**Consequences, in order of how much they matter:**

- **Returns are biased upward and drawdowns downward.** The worst outcomes — the ones that ended
  in a delisting — are structurally absent. A model trained here has never seen a company die.
- **The adjustment audit inherits it.** ADR-0019's verdict says what Stooq did to survivors, and
  nothing about what it did to the delisted, because there are none to check.
- **Any v0.8 equity backtest overstates.** This must be said in the eval report, not only here.
- **Crypto and FX are affected differently, not less.** Binance delists pairs and the archive does
  retain them, so the crypto tier is closer to honest; the Dukascopy instruments are 26 hand-picked
  majors, which is a selection bias of a different and more obvious kind.

**Not fixed, and why.** A survivorship-free US equity history is a paid product (CRSP, Norgate).
The roadmap is free-first, so the bias is accepted and documented rather than removed. If it is
ever removed, this section is the before-picture.

---

## 2. The cleaning pass excises real market behaviour

Three of ADR-0018's rules delete or partition bars that genuinely traded. All three are inherited
from Kronos and kept for comparability; keeping them is a decision, not an oversight.

### 2.1 Extreme moves are cut along with data errors

`PartitionByPriceJumps` cuts wherever `|open_t / close_{t−1} − 1|` exceeds the frequency's
threshold — 0.20 at 1h, 0.30 at 1d. The rule cannot tell an unadjusted corporate action from a
genuine overnight collapse, so it cuts both.

The consequence is specific and worth stating precisely: the model is **systematically
under-exposed to extreme discontinuities**. It sees the calm approach to every crash and the calm
recovery after it, and never the boundary itself. Crypto delivers those boundaries regularly.

An intrabar crash — a low that spikes down and a close that recovers — is correctly *not* cut,
because the rule reads open against the previous close and neither moved. So the corpus keeps
violent *within*-bar behaviour and drops violent *between*-bar behaviour, which is an odd shape to
train on and the honest description of what Kronos does.

### 2.2 Halts and thin tape are excised

The stagnant rule removes runs of more than `max_stagnant` bars with exactly equal closes — three
at 1h and at 1d. US limit-up/limit-down halts print exactly that, as do thin small-caps and any
instrument in a trading suspension.

Removing them is defensible: a model that learns "the price is usually exactly what it was" has
learned a market-structure artifact rather than a price process. The cost is that the corpus
contains no halts, so the model has no representation of one.

The illiquid rule removes runs of more than `max_illiquid` bars with zero volume — one bar at both
1h and 1d, which is aggressive. Two consecutive hours in which nothing traded are gone.

### 2.3 Measured drop rates

Per-rule, per-source drop statistics are in `docs/reports/v0.3-clean-qa.md` and regenerate from
`clean/v1/dropstats.parquet`. Read that file for the numbers; this section says what they mean.

---

## 3. Adjusted equity prices are not traded prices

Stooq's US closes are split **and** dividend adjusted (ADR-0019, measured in
`docs/reports/v0.2-adjustment-audit.md`). Two things follow.

**The tokenizer's notion of an equity price is a total-return index, not a price.** No trade ever
happened at most of the numbers in the equities tier. For a model that predicts distributions of
returns this is arguably the *right* series, and it removes the ex-dividend gap problem that
usually has to be caveated — but it is not what a naive reader assumes "close" means.

**The equity history is not stable across pulls.** Every historical bar is restated when a new
dividend or split lands. A re-pull legitimately changes bars from 1998. The manifest hash catches
it and the clean run's staleness guard re-cleans the affected series; that is the mechanism
working, not a corruption. It does mean "the corpus" is a moving object unless a pull is pinned.

A price-path series would have to be derived if anything ever needs actual traded prices. Nothing
in v0.3–v1.0 does, so it does not exist.

---

## 4. Corpus scale

**42.4 M raw bars** across four asset classes, against the roadmap's ~50 M M0 target and against
Kronos's reported 12 billion. This project is roughly **three orders of magnitude smaller than the
paper it adapts**, and no amount of care about cleaning changes that.

What actually constrains training is smaller still. The usable-window count — `Σ max(0, n_bars −
511)` over surviving segments — is the number v0.5 sizes against, and it is far below the bar
count because short series contribute nothing:

- 1,384 of 12,425 Stooq series hold fewer than 128 bars and are dropped outright by `min_bars`.
- 3,512 hold fewer than 512 and contribute **zero** context-512 windows even though their bars
  survive cleaning.

The per-slice usable-bar and usable-window tables are in `docs/reports/v0.3-clean-qa.md`.

The honesty banner from the roadmap applies and goes verbatim into the model card: expected
out-of-sample directional accuracy 50–53 %, RankIC 0.00–0.04. Volatility is the genuinely
forecastable target.

---

## 5. Source-specific artifacts carried forward

- **Off-grid Binance bars.** Stretches of hourly bars phase-shifted after an exchange restart —
  43 consecutive bars on spot BTCUSDT from 2018-02-09, each exactly one hour after the last but
  all offset by 28m14.789s. Real bars that really traded. They are counted, never snapped or
  dropped (ADR-0010), and the gap rule maps each to the slot it opened in.
- **Dukascopy weekend padding.** Some eras pad the weekend with synthetic flat zero-volume bars
  carrying the Friday close forward. The raw tier records how many; the illiquid and stagnant
  rules remove the long runs.
- **Dukascopy's moving week boundary.** The vendor's server clock observes European DST, so the
  week has opened at 19:00, 21:00 and 22:00 UTC across its history. The 24x5 expected-gap window
  is deliberately wider than any single regime, which means a genuine two-to-three-hour
  Friday-evening outage is counted as expected rather than cut. That is the safe direction and it
  is a window three hours wide.
- **Only 26 Dukascopy instruments.** FX majors and minors, commodities and index CFDs, chosen by
  hand. Not a sample of anything.

---

## 6. Not yet measured

Named here so their absence is a decision rather than an oversight.

- **Leakage.** The v0.4 causality audit and the v0.5 tokenizer temporal firewall have not run.
  Until they do, no claim about out-of-sample performance means anything.
- **Vendor dividend convention.** `axiom.adjust.policy` assumes vendor dividend values are as
  paid rather than back-adjusted for later splits. Under the recorded verdict that code path is
  dead, so the assumption has not been tested against reality. It must be measured before a
  re-audit ever switches it on.
- **Four Kronos Table 4 rows.** The 10m, 20m, 40m and 2H thresholds came from a secondary
  extraction and have not been re-read against the paper. They carry `verified: false` in
  `clean_v1.yaml` and the config refuses to hand them out. Only 1h and 1d are exercised.
