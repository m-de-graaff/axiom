# ADR-0006: Corpus ambition

**Status:** Accepted (v0.0)

## Context

Kronos trained on roughly 12 billion bars. This project can realistically assemble 50 million, or
300 million if the M1 stretch is taken. That is 40× to 240× less data for a model of comparable
size, which means undertraining is not a risk to mitigate — it is a certainty to document.

The failure mode is not the undertraining itself. It is publishing evaluation numbers as though
the model were trained to convergence.

## Decision

M0, roughly 50 million bars, is the mandatory floor. M1 is a stretch decided at G3 per ADR-0003.

Undertraining relative to Kronos is an accepted limitation and goes in the model card explicitly,
alongside the honesty banner and the survivorship-bias note.

If evaluation at G5 shows the model losing to a LightGBM baseline everywhere, v1.0 reframes as a
tokenizer and representation study. That is a legitimate v1.0, decided at G5 rather than argued
about at the end.

## Consequences

Every evaluation number carries the corpus scale next to it. A RankIC of 0.02 from 50 million bars
is a different claim than the same number from 12 billion, and the model card has to let a reader
tell them apart.

The reframing path means G5 has two passing outcomes, not one. Planning for the study outcome now
is what stops it from feeling like a failure later.
