"""The one change made to the vendored generation loop: `reduce="none"`.

Calibration (P2-04) needs the individual MC sample paths, which upstream averages
away inside `auto_regressive_inference`. This asserts the escape hatch changes
nothing else: same seed, same samples, and their mean is exactly what upstream
returned. Tiny random-weight config, CPU, no network — CI runs it.

This is *not* the parity harness (P4-02); that one covers greedy token identity and
MC moment tolerance across CPU/CUDA/ROCm.
"""

import numpy as np
import pytest
import torch
from conftest import tiny_predictor

CONTEXT, PRED_LEN, SAMPLES = 12, 3, 4


@pytest.fixture
def predictor():
    return tiny_predictor()


@pytest.fixture
def inputs():
    x = np.random.default_rng(0).standard_normal((1, CONTEXT, 6)).astype("float32")
    return x, np.zeros((1, CONTEXT, 5), "float32"), np.zeros((1, PRED_LEN, 5), "float32")


def generate(predictor, inputs, seed, **kwargs):
    torch.manual_seed(seed)
    return predictor.generate(*inputs, PRED_LEN, 1.0, 0, 0.9, SAMPLES, False, **kwargs)


def test_reduce_none_returns_the_paths_that_upstream_averages(predictor, inputs):
    averaged = generate(predictor, inputs, seed=3)
    paths = generate(predictor, inputs, seed=3, reduce="none")

    assert averaged.shape == (1, PRED_LEN, 6)
    assert paths.shape == (1, SAMPLES, PRED_LEN, 6)
    assert np.array_equal(paths.mean(axis=1), averaged)


def test_the_default_is_still_upstream_behaviour(predictor, inputs):
    assert generate(predictor, inputs, seed=5).shape == (1, PRED_LEN, 6)
    assert np.array_equal(generate(predictor, inputs, seed=5), generate(predictor, inputs, seed=5))
