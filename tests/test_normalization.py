"""Normalization must stay bit-compatible with upstream Kronos (CLAUDE.md, #1 failure mode)."""

import numpy as np
import pandas as pd
from axiom_data import normalization as norm


def window(seed: int = 0, n: int = 40, k: int = 6) -> np.ndarray:
    return np.random.default_rng(seed).normal(100, 5, (n, k)).astype(np.float32)


def test_matches_upstream_formula():
    """Same arithmetic as finetune/dataset.py and KronosPredictor.predict."""
    x = window()
    ctx = 30
    past = x[:ctx]
    expected = np.clip((x - past.mean(0)) / (past.std(0) + 1e-5), -5.0, 5.0)
    got, mean, std = norm.normalize_window(x, ctx)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
    np.testing.assert_allclose(mean, past.mean(0))
    np.testing.assert_allclose(std, past.std(0))


def test_stats_ignore_the_horizon():
    """Shifting the horizon rows must not move the normalization statistics."""
    x = window()
    ctx = 30
    _, mean_a, std_a = norm.normalize_window(x, ctx)
    y = x.copy()
    y[ctx:] += 1000.0
    _, mean_b, std_b = norm.normalize_window(y, ctx)
    np.testing.assert_array_equal(mean_a, mean_b)
    np.testing.assert_array_equal(std_a, std_b)


def test_round_trip():
    x = window()
    mean, std = norm.fit(x)
    np.testing.assert_allclose(norm.invert(norm.apply(x, mean, std), mean, std), x, rtol=1e-5)


def test_clip_is_applied():
    x = np.zeros((10, 2), dtype=np.float32)
    x[-1] = 1e6
    out = norm.apply(x, *norm.fit(x[:9]))
    assert out.max() == norm.CLIP


def test_time_features_order_and_values():
    ts = pd.to_datetime(["2024-03-05 13:45:00"])
    tf = norm.time_features(ts)
    assert list(tf.columns) == norm.TIME_FEATURES
    assert tf.iloc[0].tolist() == [45, 13, 1, 5, 3]  # Tuesday == weekday 1


def test_ensure_amount_uses_mean_price():
    df = pd.DataFrame(
        {"open": [1.0], "high": [3.0], "low": [1.0], "close": [3.0], "volume": [10.0]}
    )
    assert norm.ensure_amount(df).amount.iloc[0] == 20.0
