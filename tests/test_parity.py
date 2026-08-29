"""Parity harness (P4-02): the cached generation loop must equal the reference one.

The reference is `_kronos.auto_regressive_inference`, ported verbatim from upstream
and never optimized. Every speedup is judged against it here:

  1. **token-identical** — same seed, same sampled tokens, same decoded bars;
  2. **MC distribution** — over many paths, the mean/std/quantiles of the generated
     returns agree within tolerance.

Tiny random-weight config, CPU, no network, so CI runs it on every push. The CUDA
leg runs on Modal before a merge and the ROCm leg (RX 7900 XTX) before any
`axiom-runtime-*` tag — see CLAUDE.md.

NEVER weaken these tolerances to make a change pass. A change that cannot hold them
is a change to the model, not an optimization of it, and it needs its own before/after
numbers from `axiom-eval`.
"""

import numpy as np
import pytest
import torch
from axiom_model._kronos import auto_regressive_inference
from axiom_model.generate import cached_inference
from conftest import tiny_predictor

CONTEXT, PRED_LEN, MAX_CONTEXT = 24, 6, 32


@pytest.fixture
def rig():
    predictor = tiny_predictor(max_context=MAX_CONTEXT)
    rng = np.random.default_rng(11)
    x = torch.from_numpy(rng.standard_normal((1, CONTEXT, 6)).astype("float32"))

    def stamps(n, start):  # minute, hour, weekday, day, month — real ranges
        hours = np.arange(start, start + n)
        return torch.from_numpy(
            np.stack(
                [np.zeros(n), hours % 24, (hours // 24) % 7, 1 + (hours // 24) % 28,
                 1 + (hours // 720) % 12],
                axis=-1,
            )[None].astype("float32")
        )

    return predictor, (x, stamps(CONTEXT, 0), stamps(PRED_LEN, CONTEXT))


def _run(fn, predictor, inputs, seed, sample_count, **kwargs):
    torch.manual_seed(seed)
    return fn(
        predictor.tokenizer, predictor.model, *inputs, MAX_CONTEXT, PRED_LEN,
        clip=predictor.clip, T=1.0, top_k=0, top_p=0.9, sample_count=sample_count,
        verbose=False, **kwargs,
    )


def test_cached_generation_is_token_identical_to_the_reference(rig):
    predictor, inputs = rig
    reference = _run(auto_regressive_inference, predictor, inputs, 4, 8, reduce="none")
    cached = _run(cached_inference, predictor, inputs, 4, 8, reduce="none")

    assert cached.shape == reference.shape
    assert np.array_equal(cached, reference), "same seed must give the same bars"


def test_cached_generation_matches_the_mc_distribution(rig):
    """Even with different seeds the sampled distribution must be the same one."""
    predictor, inputs = rig
    reference = _run(auto_regressive_inference, predictor, inputs, 1, 256, reduce="none")
    cached = _run(cached_inference, predictor, inputs, 2, 256, reduce="none")

    close = reference[0, :, :, 3], cached[0, :, :, 3]  # close-price paths
    for stat in (np.mean, np.std):
        assert np.allclose(stat(close[0], axis=0), stat(close[1], axis=0), rtol=0.15, atol=0.02)
    for q in (0.1, 0.5, 0.9):
        assert np.allclose(
            np.quantile(close[0], q, axis=0), np.quantile(close[1], q, axis=0),
            rtol=0.15, atol=0.05,
        )


def test_the_cache_refuses_a_sliding_window(rig):
    """Upstream slides its window and re-rotates every token; no cache can match that."""
    predictor, inputs = rig
    with pytest.raises(ValueError, match="max_context"):
        cached_inference(
            predictor.tokenizer, predictor.model, *inputs, CONTEXT + PRED_LEN - 1, PRED_LEN,
        )


def test_predictor_dispatches_to_the_cache_and_can_be_told_not_to(rig):
    predictor, inputs = rig
    args = (*[t.numpy() for t in inputs], PRED_LEN, 1.0, 0, 0.9, 4, False)

    torch.manual_seed(7)
    cached = predictor.generate(*args, reduce="none")
    torch.manual_seed(7)
    uncached = predictor.generate(*args, reduce="none", use_cache=False)

    assert np.array_equal(cached, uncached)
