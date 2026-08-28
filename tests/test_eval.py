"""Eval harness: leak-free windows, metrics that are right on known inputs, a run
that produces a report. The model forecaster is covered by the smoke/parity tests —
here everything is CPU-only and runs in CI.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from axiom_data import datasets, store
from axiom_eval import forecasters, metrics
from axiom_eval.panel import cross_sections, load_config, load_split_bars
from axiom_eval.run import run
from conftest import bars
from pandas.testing import assert_frame_equal

TF = "1h"
CTX, HORIZONS = 8, [1, 2]
SYMBOLS = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
SPLITS = {
    "train": {"start": "2020-01-01", "end": "2020-09-30"},
    "val": {"start": "2020-10-05", "end": "2020-11-30"},
    "test": {"start": "2020-12-05", "end": "2021-01-20"},
}


@pytest.fixture
def corpus(tmp_path):
    """Tiny four-symbol corpus + the data and eval configs that describe it."""
    root = tmp_path / "parquet"
    for i, symbol in enumerate(SYMBOLS):
        store.write_months(bars("2020-01-01", 24 * 400, seed=i), root, "binance", symbol, TF)

    universe = tmp_path / "universe.yaml"
    universe.write_text(yaml.safe_dump({"venue": "binance", "symbols": SYMBOLS}))
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "universe": str(universe),
                "source_tf": TF,
                "timeframes": [TF],
                "resample": "right_closed_right_labeled",
                "context_bars": CTX,
                "horizons": HORIZONS,
                "normalization": "upstream_v1",
                "embargo_bars": CTX,
                "splits": SPLITS,
            }
        )
    )
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "data": str(data_config),
                "split": "test",
                "models": [],
                "baselines": {"persistence": True, "ewma": True, "lightgbm": True},
                "mc": {"samples": 16, "temperature": 1.0, "top_p": 0.9},
                "costs": {"taker_fee_bps": 10, "slippage_bps": 7},
                "metrics": ["rankic"],
                "slices": ["year", "vol_tercile"],
                "report_dir": str(tmp_path / "reports"),
                "panel": {
                    "timeframes": [TF],
                    "horizons": HORIZONS,
                    "stride_bars": 4,
                    "max_anchors": 6,
                    "min_symbols": 2,
                    "vol_window": 4,
                    "band": [0.1, 0.9],
                },
                "ewma": {"halflife_bars": 4},
                "lightgbm": {
                    "fit_splits": ["train"],
                    "lags": [1, 2],
                    "vol_windows": [4],
                    "train_stride_bars": 8,
                    "max_train_rows": 2000,
                    "num_boost_round": 20,
                    "params": {"num_leaves": 7, "min_data_in_leaf": 5},
                },
                "strategy": {"threshold_mult": 1.0},
                "wandb": {"enabled": False},
            }
        )
    )
    return eval_config, data_config, root, tmp_path


def sections(eval_config, root):
    cfg, data_cfg = load_config(eval_config)
    bars_by_symbol = load_split_bars(cfg, data_cfg, TF, root=root)
    return cfg, data_cfg, list(cross_sections(cfg, data_cfg, TF, bars_by_symbol))


def test_windows_are_leak_free(corpus):
    """Context inside the split, horizon inside the split, no future bars in context."""
    eval_config, _, root, _ = corpus
    cfg, data_cfg, found = sections(eval_config, root)
    lo, hi = datasets.split_bounds(data_cfg, "test", TF)
    assert found, "no cross-sections built"
    for anchor, windows in found:
        for w in windows:
            assert len(w.context) == CTX and w.context.ts.iloc[-1] == anchor
            assert w.context.ts.iloc[0] >= lo
            assert anchor + max(HORIZONS) * pd.Timedelta(TF) <= hi
            assert (w.context.ts < anchor + pd.Timedelta(TF)).all()


def test_every_symbol_is_scored_at_the_same_anchors(corpus):
    """A cross-sectional RankIC only means something if the cross-section is real."""
    eval_config, _, root, _ = corpus
    _, _, found = sections(eval_config, root)
    assert all(len({w.symbol for w in windows}) == len(SYMBOLS) for _, windows in found)
    assert len({anchor for anchor, _ in found}) == len(found)


def test_realized_returns_come_from_the_bars_after_the_anchor(corpus):
    eval_config, _, root, _ = corpus
    _, _, found = sections(eval_config, root)
    anchor, windows = found[0]
    w = windows[0]
    actual = store.read(w.symbol, TF, root=root, venue="binance")
    i = int(actual.index[actual.ts == anchor][0])
    expected = np.log(actual.close.to_numpy()[[i + h for h in HORIZONS]] / actual.close.iloc[i])
    assert np.allclose(w.realized(HORIZONS), expected)


def test_windows_never_span_a_gap(corpus):
    """Punch a hole in the test split; no window may straddle it."""
    eval_config, _, root, _ = corpus
    path = store.month_path(root, "binance", SYMBOLS[0], TF, 2020, 12)
    df = pd.read_parquet(path)
    hole = df.ts.iloc[200:230]
    df.drop(index=range(200, 230)).to_parquet(path, index=False)

    _, _, found = sections(eval_config, root)
    for _, windows in found:
        for w in (w for w in windows if w.symbol == SYMBOLS[0]):
            assert not w.context.ts.isin(hole).any()
            assert (w.context.ts.diff().iloc[1:] == pd.Timedelta(TF)).all()


def test_anchor_grid_is_machine_independent(corpus):
    """Anchors are epoch-aligned, so two machines pick the same ones from config alone."""
    eval_config, _, root, _ = corpus
    _, _, found = sections(eval_config, root)
    grid = int((4 * pd.Timedelta(TF)).value)
    assert all(pd.Timestamp(a).value % grid == 0 for a, _ in found)


def _panel(pred, realized, samples=None, ts=None, symbols=None):
    n = len(pred)
    ts = ts if ts is not None else ["2024-01-01"] * n
    symbols = symbols if symbols is not None else [f"S{i}" for i in range(n)]
    lo = samples[0] if samples else np.array(pred) - 1.0
    hi = samples[1] if samples else np.array(pred) + 1.0
    return pd.DataFrame(
        {
            "model": "m", "tf": "1h", "horizon": 1, "ts": pd.to_datetime(ts), "symbol": symbols,
            "pred": pred, "realized": realized, "q_lo": lo, "q_hi": hi,
            "pit": np.full(n, 0.5), "ctx_vol": np.full(n, 0.01),
        }
    )


def test_rankic_is_one_for_a_perfect_ranker_and_minus_one_when_inverted():
    real = [0.03, 0.01, -0.01, -0.03, 0.02]
    good = pd.concat([_panel(real, real), _panel(real, real, ts=["2024-01-02"] * 5)])
    assert metrics.rankic(good, min_symbols=3)["rankic"] == pytest.approx(1.0)
    flipped = good.assign(pred=-good["pred"])
    assert metrics.rankic(flipped, min_symbols=3)["rankic"] == pytest.approx(-1.0)


def test_rankic_ignores_cross_sections_that_are_too_thin():
    real = [0.03, -0.01]
    assert metrics.rankic(_panel(real, real), min_symbols=3)["n_cross_sections"] == 0


def test_directional_accuracy_only_counts_forecasts_that_clear_costs():
    cost = 0.0034  # 34 bps round trip
    #        below cost (ignored)   above, right     above, wrong
    pred = [0.001, -0.002, 0.010, -0.010, 0.010]
    real = [-0.05, 0.05, 0.02, -0.02, -0.02]
    out = metrics.directional(_panel(pred, real), cost)
    assert out["n_signals"] == 3
    assert out["dir_acc_cost"] == pytest.approx(2 / 3)
    assert out["net_edge_bps"] == pytest.approx(((0.02 + 0.02 - 0.02) / 3 - cost) * 1e4)


def test_errors_are_computed_on_log_returns():
    out = metrics.errors(_panel([0.01, -0.01], [0.02, -0.03]))
    assert out["mae_logret"] == pytest.approx(0.015)
    assert out["rmse_logret"] == pytest.approx(np.sqrt((0.01**2 + 0.02**2) / 2))


def test_calibration_recovers_a_known_coverage():
    rng = np.random.default_rng(0)
    realized = rng.standard_normal(4000) * 0.01
    pred = np.zeros(4000)
    q = 1.2815515655446004 * 0.01  # 10th/90th percentile of the true distribution
    panel = _panel(pred, realized, samples=(np.full(4000, -q), np.full(4000, q)))
    panel["pit"] = pd.Series(realized).rank(pct=True)
    out = metrics.calibration(panel)
    assert out["coverage_10_90"] == pytest.approx(0.8, abs=0.02)
    assert out["pit_ks"] < 0.05
    assert len(out["pit_hist"]) == 10


def test_tripwire_pays_the_round_trip_on_every_trade():
    cost = 0.0034
    pred = [0.01, 0.01, 0.0001]  # the third never clears the threshold
    real = [0.02, -0.01, 0.05]
    out = metrics.tripwire(_panel(pred, real), cost)
    assert out["n_trades"] == 2
    assert out["net_return_bps"] == pytest.approx(((0.02 - cost) + (-0.01 - cost)) / 2 * 1e4)
    assert out["win_rate"] == pytest.approx(0.5)


def test_baselines_are_deterministic_and_shaped_right(corpus):
    eval_config, _, root, _ = corpus
    cfg, _, found = sections(eval_config, root)
    _, windows = found[0]
    for baseline in (forecasters.Persistence(cfg), forecasters.Ewma(cfg)):
        a = baseline.forecast(windows, HORIZONS, 16, cfg.seed)
        b = baseline.forecast(windows, HORIZONS, 16, cfg.seed)
        assert a.shape == (len(windows), 16, len(HORIZONS))
        assert np.array_equal(a, b) and np.isfinite(a).all()


def test_lightgbm_features_never_look_ahead(corpus):
    """Features at the anchor must not change when later bars are altered."""
    eval_config, _, root, _ = corpus
    cfg, _, found = sections(eval_config, root)
    _, windows = found[0]
    w = windows[0]
    before = forecasters.feature_frame(w.context, [1, 2], [4]).iloc[-1]
    tampered = w.context.copy()
    tampered.loc[tampered.index[-1], "close"] *= 1.5  # the anchor bar itself may matter
    changed = forecasters.feature_frame(tampered, [1, 2], [4]).iloc[-1]
    assert not before.equals(changed)
    earlier = forecasters.feature_frame(w.context.iloc[:-1], [1, 2], [4]).iloc[-1]
    assert earlier.equals(forecasters.feature_frame(w.context, [1, 2], [4]).iloc[-2])


def test_run_writes_a_report_and_a_panel(corpus):
    eval_config, data_config, root, tmp_path = corpus
    datasets.build(data_config, root=root, out_dir=tmp_path / "datasets")
    out = run(eval_config, root=root, datasets_dir=tmp_path / "datasets", use_wandb=False)

    dest = pd.Series(out["path"])[0]
    written = json.loads((pd.io.common.Path(dest) / "metrics.json").read_text())
    assert written["dataset_hash"]
    assert (pd.io.common.Path(dest) / "report.html").exists()
    assert (pd.io.common.Path(dest) / "panel.parquet").exists()

    panel = out["panel"]
    assert set(panel["model"]) == {"persistence", "ewma", "lightgbm"}
    assert set(panel["horizon"]) == set(HORIZONS)
    assert panel["pit"].between(0, 1).all()
    assert (panel["q_lo"] <= panel["q_hi"]).all()
    assert set(out["table"].columns) >= {"rankic", "dir_acc_cost", "mae_logret",
                                         "coverage_10_90", "pit_ks", "tw_n_trades"}


def test_model_forecaster_returns_log_returns_from_the_last_context_close(corpus):
    """The Axiom path with a tiny random-weight model: no weights, no network, no GPU.

    Real weights are exercised by the Modal smoke test; what matters here is the
    plumbing around the model — context-only normalization, denormalization, and
    log returns measured from the anchor's close.
    """
    from axiom_eval.forecasters import AxiomForecaster
    from conftest import tiny_predictor

    eval_config, _, root, _ = corpus
    cfg, _, found = sections(eval_config, root)
    _, windows = found[0]
    fc = AxiomForecaster("axiom-zero-tiny", cfg, predictor=tiny_predictor())

    a = fc.forecast(windows[:2], HORIZONS, 4, cfg.seed)
    b = fc.forecast(windows[:2], HORIZONS, 4, cfg.seed)
    assert a.shape == (2, 4, len(HORIZONS))
    assert np.isfinite(a).all()
    assert np.array_equal(a, b), "per-window seeding is not reproducible"
    assert np.abs(a).max() < 50, "returns are not on a log-return scale"


def test_chunking_partitions_the_anchors_without_changing_them(corpus):
    """Sharding is a scheduling detail: the union of chunks is the whole run."""
    eval_config, _, root, _ = corpus
    cfg, data_cfg, whole = sections(eval_config, root)
    bars_by_symbol = load_split_bars(cfg, data_cfg, TF, root=root)
    chunked = [
        section
        for i in range(3)
        for section in cross_sections(cfg, data_cfg, TF, bars_by_symbol, chunk=(i, 3))
    ]
    assert sorted(a for a, _ in chunked) == sorted(a for a, _ in whole)
    assert len({a for a, _ in chunked}) == len(chunked), "an anchor was scored twice"


def test_a_chunked_forecast_is_identical_to_the_unchunked_one(corpus):
    """Per-window seeding is what makes sharding safe — prove it, don't assume it."""
    eval_config, _, root, _ = corpus
    cfg, data_cfg, _ = sections(eval_config, root)
    from axiom_eval.run import build_panel

    whole = build_panel(cfg, data_cfg, root, models=["persistence"], timeframes=[TF])
    parts = pd.concat(
        [
            build_panel(cfg, data_cfg, root, models=["persistence"], timeframes=[TF],
                        chunk=(i, 2))
            for i in range(2)
        ],
        ignore_index=True,
    )
    key = ["ts", "symbol", "horizon"]
    assert_frame_equal(
        metrics.add_slices(whole).sort_values(key).reset_index(drop=True),
        metrics.add_slices(parts).sort_values(key).reset_index(drop=True),
        check_like=True,
    )
