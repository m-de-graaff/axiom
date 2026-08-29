"""Fine-tune orchestration: config in, checkpoints + a W&B record out.

Every run is reproducible from the committed YAML + git SHA + dataset hash
(CLAUDE.md rule 5); all three land in `meta.json` next to the checkpoints and in
the W&B run. Stage A adapts the tokenizer, Stage B the predictor on Stage A's
output. `infra/modal_app/train.py` (P3-05) calls `run` with volume paths.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from axiom_data import store

from ..registry import resolve
from ..tokenizer import AxiomTokenizer
from ..transformer import Axiom
from .config import DATASETS_DIR, dataset_hash, load_config
from .data import build_dataset
from .stages import fit_predictor, fit_tokenizer


def _git_sha() -> str:
    injected = os.environ.get("AXIOM_GIT_SHA")
    if injected:
        return injected
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wandb_run(cfg, stage: str, meta: dict, use_wandb: bool | None):
    """A live W&B run, or None when tracking is off. Loud when asked-for tracking
    cannot start (golden rule 1 — an untracked run didn't happen)."""
    enabled = cfg.wandb.get("enabled", False) if use_wandb is None else use_wandb
    if not enabled:
        return None
    import wandb  # declared dependency; ImportError here should fail the run

    return wandb.init(
        project=cfg.wandb.get("project", "axiom"),
        entity=cfg.wandb.get("entity"),
        name=f"{cfg.run_name}/{stage}",
        job_type="finetune",
        config=meta,
    )


def run(
    config_path: str | Path,
    stage: str = "all",
    device: str | None = None,
    root: Path = store.DEFAULT_ROOT,
    datasets_dir: Path = DATASETS_DIR,
    out_dir: Path | None = None,
    use_wandb: bool | None = None,
    on_stage_end=None,
) -> dict:
    """Run Stage A, Stage B, or both. Returns the per-stage results.

    `on_stage_end(stage_name)` fires after each completed stage — the Modal app
    commits the checkpoint volume there, so a Stage B failure hours later cannot
    lose Stage A's checkpoint."""
    if stage not in ("a", "b", "all"):
        raise ValueError(f"stage {stage!r} not in ('a', 'b', 'all')")
    cfg, data_cfg = load_config(config_path)
    seed_everything(cfg.seed)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = Path(out_dir or cfg.out_dir)
    spec = resolve(cfg.init_model)

    meta = {
        "run_name": cfg.run_name,
        "config": str(config_path),
        "git_sha": _git_sha(),
        "dataset": data_cfg.name,
        "dataset_hash": dataset_hash(data_cfg, datasets_dir),
        "device": device,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "finetune_config": cfg.raw,
    }

    fit_ds = build_dataset(cfg, data_cfg, cfg.splits["fit"], root, datasets_dir)
    select_ds = build_dataset(cfg, data_cfg, cfg.splits["select"], root, datasets_dir)
    print(
        f"{cfg.run_name}: fit={cfg.splits['fit']} ({len(fit_ds)} windows)  "
        f"select={cfg.splits['select']} ({len(select_ds)} windows)  device={device}",
        flush=True,
    )

    results: dict[str, dict] = {}
    if stage in ("a", "all") and cfg.stage_a.get("enabled", True):
        results["stage_a"] = _run_stage(
            "stage_a", cfg, meta, use_wandb,
            lambda wb: fit_tokenizer(
                AxiomTokenizer.from_pretrained(spec.tokenizer_source),
                fit_ds, select_ds, cfg, device, cfg.stage_dir(out_root, "tokenizer"), wb,
            ),
        )
        if on_stage_end is not None:
            on_stage_end("stage_a")

    if stage in ("b", "all") and cfg.stage_b.get("enabled", True):
        tokenizer = _stage_b_tokenizer(cfg, out_root, spec)
        results["stage_b"] = _run_stage(
            "stage_b", cfg, meta, use_wandb,
            lambda wb: fit_predictor(
                Axiom.from_pretrained(spec.model_source), tokenizer,
                fit_ds, select_ds, cfg, device, cfg.stage_dir(out_root, "predictor"), wb,
            ),
        )
        if on_stage_end is not None:
            on_stage_end("stage_b")

    (out_root / cfg.run_name).mkdir(parents=True, exist_ok=True)
    (out_root / cfg.run_name / "meta.json").write_text(
        json.dumps({**meta, "results": results}, indent=2, default=str), encoding="utf-8"
    )
    return {"meta": meta, "results": results}


def _run_stage(stage: str, cfg, meta: dict, use_wandb: bool | None, work) -> dict:
    wb = _wandb_run(cfg, stage, meta, use_wandb)
    try:
        result = work(wb)
        if wb is not None:
            result["wandb_id"] = wb.id
            wb.summary[f"{stage}/best_val_loss"] = result["best_val_loss"]
        return result
    finally:
        if wb is not None:
            wb.finish()


def _stage_b_tokenizer(cfg, out_root: Path, spec) -> AxiomTokenizer:
    """Stage B consumes Stage A's tokenizer by default; `stage_b.tokenizer: init`
    opts into the un-finetuned one explicitly — never by silent fallback."""
    source = cfg.stage_b.get("tokenizer", "stage_a")
    if source == "init":
        return AxiomTokenizer.from_pretrained(spec.tokenizer_source)
    if source != "stage_a":
        raise ValueError(f"stage_b.tokenizer {source!r} not in ('stage_a', 'init')")
    best = cfg.stage_dir(out_root, "tokenizer") / "best_model"
    if not best.exists():
        raise FileNotFoundError(
            f"{best} missing — run Stage A first, or set stage_b.tokenizer: init"
        )
    return AxiomTokenizer.from_pretrained(str(best))


__all__ = ["run", "seed_everything"]
