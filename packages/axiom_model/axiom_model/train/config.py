"""Fine-tune config: one committed YAML per run (CLAUDE.md rule 5).

The YAML carries every hyperparameter; nothing here invents a value. Guards that
protect the golden rules live in `load_config` so a bad config dies before any
GPU time is spent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from axiom_data import datasets

from ..registry import resolve

DATASETS_DIR = Path("data/datasets")

# Upstream defaults (vendor/kronos/finetune_csv + finetune/config.py). Present here
# only as documentation of provenance — the committed YAML must spell them out.
_REQUIRED = ["run_name", "seed", "data", "init", "window", "splits", "stage_a", "stage_b"]


@dataclass
class FinetuneConfig:
    """Resolved `configs/finetune/*.yaml`."""

    run_name: str
    seed: int
    precision: str
    data: str
    init_model: str
    window: dict
    splits: dict
    loader: dict
    stage_a: dict
    stage_b: dict
    out_dir: str
    log_interval: int
    wandb: dict
    raw: dict = field(repr=False)

    @property
    def sample_bars(self) -> int:
        """Bars per training sample: context + horizon, same geometry the eval feeds."""
        return int(self.window["context_bars"]) + int(self.window["horizon_bars"])

    def stage_dir(self, out_root: Path, stage: str) -> Path:
        return Path(out_root) / self.run_name / stage


def load_config(path: str | Path) -> tuple[FinetuneConfig, datasets.DataConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if _has_todo(raw):
        raise ValueError(f"{path}: TODO placeholders left — fill every value before running")

    cfg = FinetuneConfig(
        run_name=raw["run_name"],
        seed=int(raw["seed"]),
        precision=raw.get("precision", "fp32"),
        data=raw["data"],
        init_model=raw["init"]["model"],
        window=raw["window"],
        splits=raw["splits"],
        loader=raw.get("loader", {}),
        stage_a=raw["stage_a"],
        stage_b=raw["stage_b"],
        out_dir=raw.get("out_dir", "data/checkpoints"),
        log_interval=int(raw.get("log_interval", 50)),
        wandb=raw.get("wandb", {}),
        raw=raw,
    )
    data_cfg = datasets.load_config(cfg.data)

    # Rule 3: test is read-only. A fine-tune may not fit on it, and may not select
    # its checkpoint on it either.
    for role, split in cfg.splits.items():
        if split == "test":
            raise ValueError(f"splits.{role} = 'test' — test years are reserved for the M1 verdict")

    spec = resolve(cfg.init_model)
    if cfg.sample_bars > spec.max_context:
        raise ValueError(
            f"window {cfg.window['context_bars']}+{cfg.window['horizon_bars']} bars exceeds "
            f"{cfg.init_model}'s max_context {spec.max_context}"
        )
    if cfg.precision not in ("fp32", "bf16"):
        raise ValueError(f"precision {cfg.precision!r} not in ('fp32', 'bf16')")
    return cfg, data_cfg


def dataset_hash(data_cfg: datasets.DataConfig, datasets_dir: Path = DATASETS_DIR) -> str:
    """The hash `axiom-data build` printed; refuses a manifest that no longer matches
    the data config (a silently stale segment index would poison every run after it)."""
    manifest_path = Path(datasets_dir) / data_cfg.name / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing — run `axiom-data build --config {data_cfg.name}` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["spec"] != data_cfg.spec():
        raise ValueError(
            f"{manifest_path} was built from a different data config — rebuild the dataset"
        )
    return manifest["dataset_hash"]


def _has_todo(node) -> bool:
    if isinstance(node, dict):
        return any(_has_todo(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_todo(v) for v in node)
    return isinstance(node, str) and node.strip().upper() == "TODO"


__all__ = ["DATASETS_DIR", "FinetuneConfig", "dataset_hash", "load_config"]
