"""Dataset builder: reproducible hash, no leakage across splits, no windows over gaps."""

import json

import pandas as pd
import pytest
import yaml
from axiom_data import datasets, store
from conftest import bars

TF = "1h"
CTX, HORIZON = 8, 2
SYMBOLS = ["AAAUSDT", "BBBUSDT"]


@pytest.fixture
def corpus(tmp_path):
    """A tiny two-symbol corpus plus the configs that describe it."""
    root = tmp_path / "parquet"
    for i, symbol in enumerate(SYMBOLS):
        store.write_months(bars("2020-01-01", 24 * 400, seed=i), root, "binance", symbol, TF)

    universe = tmp_path / "universe.yaml"
    universe.write_text(yaml.safe_dump({"venue": "binance", "symbols": SYMBOLS}))
    config = tmp_path / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "universe": str(universe),
                "source_tf": TF,
                "timeframes": [TF],
                "resample": "right_closed_right_labeled",
                "context_bars": CTX,
                "horizons": [HORIZON],
                "normalization": "upstream_v1",
                "embargo_bars": CTX,
                "splits": {
                    "train": {"start": "2020-01-01", "end": "2020-09-30"},
                    "val": {"start": "2020-10-05", "end": "2020-11-30"},
                    "test": {"start": "2020-12-05", "end": "2021-01-20"},
                },
            }
        )
    )
    return config, root, tmp_path


def test_hash_is_reproducible(corpus):
    config, root, tmp = corpus
    a = datasets.build(config, root=root, out_dir=tmp / "out")
    b = datasets.build(config, root=root, out_dir=tmp / "out")
    assert a["dataset_hash"] == b["dataset_hash"]
    written = json.loads((tmp / "out" / "data" / "manifest.json").read_text())
    assert written["dataset_hash"] == a["dataset_hash"]
    assert all(s["windows"] > 0 for s in a["splits"].values())


def test_hash_changes_when_a_bar_changes(corpus):
    config, root, tmp = corpus
    before = datasets.build(config, root=root, out_dir=tmp / "out")["dataset_hash"]
    path = store.month_path(root, "binance", SYMBOLS[0], TF, 2020, 3)
    df = pd.read_parquet(path)
    df.loc[10, "close"] *= 1.0001
    df.to_parquet(path, index=False)
    assert datasets.build(config, root=root, out_dir=tmp / "out")["dataset_hash"] != before


def test_windows_never_cross_a_split_boundary(corpus):
    """The embargo: a val/test window may not reach back into an earlier split."""
    config, root, tmp = corpus
    cfg = datasets.load_config(config)
    index = datasets.build(config, root=root, out_dir=tmp / "out")["segments"]
    for row in index.itertuples():
        lo, hi = datasets.split_bounds(cfg, row.split, row.tf)
        assert row.start_ts >= lo
        assert row.end_ts <= hi
        assert row.bars >= cfg.window_bars


def test_windows_never_span_a_gap(corpus):
    """Punch a hole in the data; the segment must split around it, not straddle it."""
    config, root, tmp = corpus
    path = store.month_path(root, "binance", SYMBOLS[0], TF, 2020, 3)
    df = pd.read_parquet(path)
    df.drop(index=range(100, 130)).to_parquet(path, index=False)

    index = datasets.build(config, root=root, out_dir=tmp / "out")["segments"]
    segs = index[(index.symbol == SYMBOLS[0]) & (index.split == "train")]
    assert len(segs) == 2
    for row in segs.itertuples():
        expected = int((row.end_ts - row.start_ts) / pd.Timedelta("1h")) + 1
        assert row.bars == expected
        assert row.windows == row.bars - (CTX + HORIZON) + 1


def test_expand_matches_the_window_count(corpus):
    config, root, tmp = corpus
    row = datasets.build(config, root=root, out_dir=tmp / "out")["segments"].iloc[0]
    starts = list(datasets.expand(row.start_ts, row.bars, row.tf, CTX + HORIZON))
    assert len(starts) == row.windows
    assert starts[-1] + (CTX + HORIZON - 1) * pd.Timedelta("1h") == row.end_ts


def test_build_refuses_a_dirty_corpus(corpus):
    config, root, tmp = corpus
    path = store.month_path(root, "binance", SYMBOLS[0], TF, 2020, 3)
    df = pd.read_parquet(path)
    df.loc[5, "high"] = df.loc[5, "low"] / 2  # impossible bar
    df.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="ohlc_violations"):
        datasets.build(config, root=root, out_dir=tmp / "out")


def test_overlapping_splits_rejected():
    with pytest.raises(ValueError, match="overlap"):
        datasets.check_splits(
            {
                "train": {"start": "2020-01-01", "end": "2020-12-31"},
                "val": {"start": "2020-06-01", "end": "2021-01-01"},
            }
        )
