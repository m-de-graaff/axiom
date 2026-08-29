"""Fine-tune port (P3-02): windows come from the segment index, normalization from
`axiom_data.normalization`, both training loops run and checkpoint on CPU."""

import numpy as np
import pytest
import torch
import yaml
from axiom_data import datasets, normalization, store
from conftest import bars

TF = "1h"
CTX, HORIZON = 8, 2
SYMBOLS = ["AAAUSDT", "BBBUSDT"]


@pytest.fixture
def rig(tmp_path):
    """Tiny corpus + built dataset + a finetune config pointing at both."""
    root = tmp_path / "parquet"
    for i, symbol in enumerate(SYMBOLS):
        store.write_months(bars("2020-01-01", 24 * 400, seed=i), root, "binance", symbol, TF)

    universe = tmp_path / "universe.yaml"
    universe.write_text(yaml.safe_dump({"venue": "binance", "symbols": SYMBOLS}))
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
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
    datasets_dir = tmp_path / "datasets"
    manifest = datasets.build(data_yaml, root=root, out_dir=datasets_dir)

    ft_yaml = tmp_path / "finetune.yaml"
    ft_yaml.write_text(
        yaml.safe_dump(
            {
                "run_name": "test-ft",
                "seed": 7,
                "precision": "fp32",
                "data": str(data_yaml),
                "init": {"model": "axiom-zero-small"},
                "window": {"timeframes": [TF], "context_bars": CTX, "horizon_bars": HORIZON},
                "splits": {"fit": "train", "select": "val"},
                "loader": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "train_samples_per_epoch": 16,
                    "val_samples": 8,
                },
                "stage_a": {"enabled": True, "epochs": 2, "lr": 2.0e-4},
                "stage_b": {"enabled": True, "epochs": 1, "lr": 4.0e-5},
                "log_interval": 1000,
                "out_dir": str(tmp_path / "ckpts"),
                "wandb": {"enabled": False},
            }
        )
    )
    return ft_yaml, root, datasets_dir, manifest


def _tiny_modules():
    from axiom_model._kronos import Kronos, KronosTokenizer

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
    return tokenizer, model


def test_window_dataset_matches_the_manifest_and_the_normalization_module(rig):
    from axiom_model.train.data import WindowDataset

    ft_yaml, root, datasets_dir, manifest = rig
    data_cfg = datasets.load_config(yaml.safe_load(ft_yaml.read_text())["data"])
    ds = WindowDataset(data_cfg, "train", TF, CTX, HORIZON, root=root, datasets_dir=datasets_dir)

    # Same window length as the built dataset here, so the counts must agree exactly.
    assert len(ds) == manifest["splits"]["train"]["windows"]

    x, stamp = ds[0]
    assert x.shape == (CTX + HORIZON, 6) and stamp.shape == (CTX + HORIZON, 5)

    # Byte-for-byte the single normalization implementation, context stats only.
    seg = manifest["segments"]
    row = seg[(seg.split == "train") & (seg.tf == TF)].iloc[0]
    raw = store.read(row.symbol, TF, root=root, venue="binance")
    raw = raw[(raw.ts >= row.start_ts) & (raw.ts <= row.end_ts)]
    expected, _, _ = normalization.normalize_window(
        raw[normalization.FEATURES].to_numpy("float32")[: CTX + HORIZON], CTX
    )
    np.testing.assert_array_equal(x.numpy(), expected)


def test_fit_loader_is_deterministic_per_epoch(rig):
    from axiom_model.train.config import load_config
    from axiom_model.train.data import build_dataset, fit_loader

    ft_yaml, root, datasets_dir, _ = rig
    cfg, data_cfg = load_config(ft_yaml)
    ds = build_dataset(cfg, data_cfg, "train", root, datasets_dir)

    def first_batch(epoch):
        return next(iter(fit_loader(ds, cfg, epoch)))[0]

    assert torch.equal(first_batch(0), first_batch(0))
    assert not torch.equal(first_batch(0), first_batch(1))


def test_config_refuses_test_split_todos_and_oversized_windows(rig):
    from axiom_model.train.config import load_config

    ft_yaml = rig[0]
    base = yaml.safe_load(ft_yaml.read_text())

    def write(**changes):
        edited = {**base, **changes}
        ft_yaml.write_text(yaml.safe_dump(edited))
        return ft_yaml

    with pytest.raises(ValueError, match="M1 verdict"):
        load_config(write(splits={"fit": "train", "select": "test"}))
    with pytest.raises(ValueError, match="TODO"):
        load_config(write(stage_a={"enabled": True, "epochs": "TODO", "lr": 2.0e-4}))
    with pytest.raises(ValueError, match="max_context"):
        load_config(
            write(window={"timeframes": [TF], "context_bars": 500, "horizon_bars": 500})
        )


def test_stage_a_trains_and_checkpoints_a_loadable_tokenizer(rig, tmp_path):
    from axiom_model.tokenizer import AxiomTokenizer
    from axiom_model.train.config import load_config
    from axiom_model.train.data import build_dataset
    from axiom_model.train.stages import fit_tokenizer

    ft_yaml, root, datasets_dir, _ = rig
    cfg, data_cfg = load_config(ft_yaml)
    fit_ds = build_dataset(cfg, data_cfg, "train", root, datasets_dir)
    select_ds = build_dataset(cfg, data_cfg, "val", root, datasets_dir)
    tokenizer, _ = _tiny_modules()

    result = fit_tokenizer(tokenizer, fit_ds, select_ds, cfg, "cpu", tmp_path / "tok")
    assert np.isfinite(result["best_val_loss"])
    assert len(result["history"]) == cfg.stage_a["epochs"]
    reloaded = AxiomTokenizer.from_pretrained(result["path"])
    assert reloaded.s1_bits == tokenizer.s1_bits


def test_stage_b_trains_and_checkpoints_a_loadable_predictor(rig, tmp_path):
    from axiom_model.train.config import load_config
    from axiom_model.train.data import build_dataset
    from axiom_model.train.stages import fit_predictor
    from axiom_model.transformer import Axiom

    ft_yaml, root, datasets_dir, _ = rig
    cfg, data_cfg = load_config(ft_yaml)
    fit_ds = build_dataset(cfg, data_cfg, "train", root, datasets_dir)
    select_ds = build_dataset(cfg, data_cfg, "val", root, datasets_dir)
    tokenizer, model = _tiny_modules()

    result = fit_predictor(model, tokenizer, fit_ds, select_ds, cfg, "cpu", tmp_path / "pred")
    assert np.isfinite(result["best_val_loss"])
    reloaded = Axiom.from_pretrained(result["path"])
    assert reloaded.n_layers == model.n_layers
