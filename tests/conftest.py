"""Shared synthetic bars. Random walks have no signal -- fine for plumbing, never for eval."""

import numpy as np
import pandas as pd
import pytest
from axiom_data import store


def bars(start: str, periods: int, seed: int = 0, freq: str = "1h") -> pd.DataFrame:
    """Well-formed OHLCV bars: monotonic close-labeled `ts`, sane highs and lows."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, periods)))
    df = pd.DataFrame(
        {
            "ts": pd.date_range(start, periods=periods, freq=freq),
            "open": close,
            "close": close,
            "volume": rng.uniform(1, 10, periods),
        }
    )
    df["high"] = df[["open", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "close"]].min(axis=1) * 0.999
    df["amount"] = df.close * df.volume
    return df[store.COLUMNS]


def tiny_predictor(max_context: int = 16):
    """A random-weight Kronos small enough for CPU CI: no weights, no network.

    Same tiny config the parity harness (P4-02) will use. Dropout is off so runs
    differ only by the sampling seed.
    """
    from axiom_model._kronos import Kronos, KronosPredictor, KronosTokenizer

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
    return KronosPredictor(model.eval(), tokenizer.eval(), device="cpu", max_context=max_context)


def pytest_addoption(parser):
    parser.addoption("--network", action="store_true", help="also run tests that hit the network")


def pytest_configure(config):
    config.addinivalue_line("markers", "network: hits data.binance.vision; needs --network")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--network"):
        return
    skip = pytest.mark.skip(reason="needs --network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
