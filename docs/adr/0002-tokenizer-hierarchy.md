# ADR-0002: Tokenizer hierarchy

**Status:** Accepted (v0.0)

## Context

Stage 1 discretizes continuous candles into tokens. Three options were on the table: Kronos's
Binary Spherical Quantization with its coarse/fine split, flat Finite Scalar Quantization from
`vector-quantize-pytorch`, and a novel hierarchical-FSQ factorization nobody has published.

The hierarchical split is what makes the dual-head decoder work in stage 2, so this choice
constrains the AR architecture. It is not an isolated component swap.

## Decision

BSQ, vendored from Kronos under MIT with attribution in `NOTICE`, is the default and is what the
v1.0 model ships with.

Flat FSQ is an ablation only. It exists to produce the BSQ-vs-FSQ report at G3 and to give the
comparison a control arm. If FSQ wins on reconstruction, that is a finding for the report; it
does not automatically become the default, because a flat codebook gives the dual head nothing to
split on.

Novel hierarchical FSQ is post-1.0 research. It is the most interesting of the three and the one
with no prior art to fall back on when it fails.

## Consequences

v0.5 vendors Kronos code and takes on its MIT attribution obligation, recorded in `NOTICE` from
v0.0 so it cannot be forgotten at vendoring time.

The AR decoder in v0.7 can assume a coarse/fine token pair. That assumption is safe because this
ADR forbids shipping a flat quantizer as the default.

G3 gates AR training on tokenizer health, including the check that coarse utilization exceeds
fine. If the hierarchy does not hold, the dual head is meaningless, and the problem is caught
before 90 GPU-hours go into training on it.
