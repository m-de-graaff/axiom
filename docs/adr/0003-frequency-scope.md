# ADR-0003: Frequency scope

**Status:** Accepted (v0.0)

## Context

Kronos trained on roughly 12 billion bars. The free-tier stack available here supports two or
three orders of magnitude less. The question is which frequencies to spend that budget on.

Intraday crypto is the only source that can supply hundreds of millions of bars cheaply, but
5-minute raw data at corpus scale threatens the 100 GB Hugging Face private ceiling and would
let a single asset class dominate the corpus.

## Decision

Corpus M0 is mandatory: 1-hour and 1-day bars across crypto, FX, commodities, and equities,
targeting roughly 50 million clean bars. Four asset classes at two frequencies is the floor, and
no version past v0.2 may proceed without it.

Corpus M1 is a stretch, decided at Gate G3 and not before: crypto 15-minute (about +50 million
bars) and 5-minute (about +150 million), taking the total toward 0.25–0.3 billion.

The decision is deferred to G3 because that is the first point where tokenizer reconstruction
quality tells us whether more data is the binding constraint.

## Consequences

Cleaning thresholds for 5m and 15m are wired in v0.3 even though M1 may never happen. Wiring an
unused row of Kronos Table 4 is cheaper than retrofitting the cleaner after the corpus decision.

If M1 is taken, raw intraday may be tokenized in flight rather than persisted, to stay under the
storage ceiling. That path is a v0.6 concern and is noted in the roadmap risk register.
