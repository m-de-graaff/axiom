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
from axiom_model._kronos import Kronos, KronosPredictor, KronosTokenizer

CONTEXT, PRED_LEN, SAMPLES = 12, 3, 4


@pytest.fixture
def predictor():
    tokenizer = KronosTokenizer(
        d_in=6, d_model=16, n_heads=2, ff_dim=32, n_enc_layers=1, n_dec_layers=1,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.0, gamma0=1.0, gamma=1.0, zeta=1.0, group_size=2,
    )
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=1, d_model=16, n_heads=2, ff_dim=32,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=True,
    )
    return KronosPredictor(model.eval(), tokenizer.eval(), device="cpu", max_context=16)


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
