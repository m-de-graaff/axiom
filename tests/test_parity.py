"""Parity harness placeholder (P4-02).

Will assert, for axiom_model generation:
  1) greedy decoding: token-identical before/after any optimization,
     on CPU (tiny config, CI), Modal CUDA, and — before any
     axiom-runtime-* release tag — local ROCm (RX 7900 XTX);
  2) sampled MC: return-distribution moments (mean/std/q10/q50/q90 over
     ~1k paths) within tight tolerance.
NEVER weaken tolerances to pass (CLAUDE.md).
"""

import pytest


@pytest.mark.skip(reason="P4-02: implement real parity harness")
def test_parity_placeholder():
    raise AssertionError("unreachable")
