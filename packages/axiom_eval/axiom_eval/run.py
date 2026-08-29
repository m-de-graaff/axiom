"""The harness itself (P2-10, P2-12): config in, `reports/{run_id}/` out.

A run is reproducible from the committed eval YAML + git SHA + dataset hash, and
nothing else (CLAUDE.md rule 5). Sampling is seeded per window from
`forecasters.window_seed`, so results do not depend on evaluation order, on how
many symbols happened to be present, or on which machine ran it.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from axiom_data import store

from . import forecasters, metrics, report
from .panel import cross_sections, load_config, load_split_bars

DATASETS_DIR = Path("data/datasets")


def _git_sha() -> str:
    """The run's identity. `AXIOM_GIT_SHA` covers machines with the code but no repo
    (Modal containers get the directory, not the .git)."""
    injected = os.environ.get("AXIOM_GIT_SHA")
    if injected:
        return injected
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _dataset_hash(name: str, datasets_dir: Path) -> str:
    """The hash `axiom-data build` printed. No manifest, no run — comparability dies."""
    manifest = Path(datasets_dir) / name / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} missing — run `axiom-data build --config <data yaml>` first"
        )
    return json.loads(manifest.read_text(encoding="utf-8"))["dataset_hash"]


def environment_info(device: str | None) -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "device": device}
    try:
        import torch

        env["torch"] = str(torch.__version__)  # TorchVersion pickles as a torch type
        env["device"] = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch"] = None
    return env


def _rows(fc, tf: str, anchor, windows, horizons: Sequence[int], samples: int, seed: int,
          band, vol_window: int) -> pd.DataFrame:
    """Score one cross-section: MC paths in, panel rows out."""
    preds = fc.forecast(windows, horizons, samples, seed)
    realized = np.array([w.realized(horizons) for w in windows])
    q_lo, q_hi = np.quantile(preds, band, axis=1)
    n, h = realized.shape
    return pd.DataFrame(
        {
            "model": fc.name,
            "tf": tf,
            "horizon": np.tile(horizons, n),
            "ts": anchor,
            "symbol": np.repeat([w.symbol for w in windows], h),
            "pred": np.median(preds, axis=1).ravel(),
            "realized": realized.ravel(),
            "q_lo": q_lo.ravel(),
            "q_hi": q_hi.ravel(),
            "pit": (preds < realized[:, None, :]).mean(axis=1).ravel(),
            "ctx_vol": np.repeat([w.context_vol(vol_window) for w in windows], h),
        }
    )


def build_panel(
    cfg,
    data_cfg,
    root: Path = store.DEFAULT_ROOT,
    device: str | None = None,
    models: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    chunk: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Score every configured forecaster over every configured timeframe.

    `chunk=(i, n)` scores only every n-th anchor — see `panel.cross_sections`.
    """
    horizons, band = cfg.horizons, tuple(cfg.panel["band"])
    frames = []
    for tf in timeframes or cfg.panel["timeframes"]:
        bars = load_split_bars(cfg, data_cfg, tf, root=root, symbols=symbols)
        if not bars:
            print(f"  {tf}: no bars under {root}, skipped", flush=True)
            continue
        for fc in forecasters.build(
            cfg, data_cfg, tf, root=root, device=device, models=models, symbols=symbols
        ):
            t0, n = time.time(), 0
            for a, (anchor, windows) in enumerate(
                cross_sections(cfg, data_cfg, tf, bars, chunk), start=1
            ):
                frames.append(
                    _rows(fc, tf, anchor, windows, horizons, cfg.mc["samples"], cfg.seed,
                          band, cfg.panel["vol_window"])
                )
                n += len(windows)
                if a % 10 == 0:  # a multi-hour run must not look hung in a log file
                    print(f"  {tf:>4} {fc.name:<20} anchor {a:>4}  {time.time() - t0:7.1f}s",
                          flush=True)
            print(f"  {tf:>4} {fc.name:<20} {n:>6} windows  {time.time() - t0:7.1f}s",
                  flush=True)
    if not frames:
        raise ValueError("empty panel — no windows scored (check the corpus and split bounds)")
    return pd.concat(frames, ignore_index=True)


def run(
    config_path: str | Path,
    root: Path = store.DEFAULT_ROOT,
    out_dir: Path | None = None,
    datasets_dir: Path = DATASETS_DIR,
    device: str | None = None,
    models: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    max_anchors: int | None = None,
    use_wandb: bool | None = None,
) -> dict:
    """Run the harness end to end and write `reports/{run_id}/`."""
    cfg, data_cfg = load_config(config_path)
    if _wandb_requested(cfg, use_wandb):
        try:  # fail before the panel build, not after the multi-hour part (B-08)
            import wandb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "the config asks for W&B but wandb is not importable — "
                "`uv sync` (it is a declared dependency now) or pass --no-wandb"
            ) from exc
    if max_anchors is not None:
        cfg.panel["max_anchors"] = max_anchors
    seed_everything(cfg.seed)
    panel = build_panel(cfg, data_cfg, root, device, models, timeframes, symbols)
    return finalize(
        cfg, data_cfg, panel, config_path,
        out_dir=out_dir, datasets_dir=datasets_dir, device=device, use_wandb=use_wandb,
    )


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def finalize(
    cfg,
    data_cfg,
    panel: pd.DataFrame,
    config_path: str | Path,
    out_dir: Path | None = None,
    datasets_dir: Path = DATASETS_DIR,
    device: str | None = None,
    use_wandb: bool | None = None,
    environment: dict | None = None,
) -> dict:
    """Panel in, `reports/{run_id}/` out.

    Split out from `run` so a sharded run (see `infra/modal_app/eval.py`) writes
    exactly the same report from the concatenated panel. Per-window seeding is what
    makes that legitimate: sharding cannot change a number. Slices are cut *here*,
    on the whole panel, for the same reason — vol-tercile edges cut per shard would
    make the slice depend on how the work was split.
    """
    panel = metrics.add_slices(panel)
    git_sha = _git_sha()
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{cfg.name}-{git_sha[:7]}"
    dest = Path(out_dir or cfg.report_dir) / run_id
    dest.mkdir(parents=True, exist_ok=True)

    cost = cfg.round_trip_cost
    headline = metrics.table(panel, cost, cfg.panel["min_symbols"],
                             cfg.strategy.get("threshold_mult", 1.0))
    slices = {
        key: metrics.by_slice(panel, cost, key, cfg.panel["min_symbols"]) for key in cfg.slices
    }

    meta = {
        "run_id": run_id,
        "config": str(config_path),
        "git_sha": git_sha,
        "dataset": data_cfg.name,
        "dataset_hash": _dataset_hash(data_cfg.name, datasets_dir),
        "split": cfg.split,
        "round_trip_cost_bps": round(cost * 1e4, 2),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment or environment_info(device),
        "eval_config": cfg.raw,
    }

    panel.to_parquet(dest / "panel.parquet", index=False)
    (dest / "metrics.json").write_text(
        json.dumps(
            {**meta, "table": headline.to_dict("records"),
             "slices": {k: v.to_dict("records") for k, v in slices.items()}},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    shutil.copy(config_path, dest / "eval_config.yaml")
    shutil.copy(cfg.data, dest / "data_config.yaml")
    report.write_html(dest / "report.html", meta, headline, slices)
    _log_wandb(cfg, meta, headline, dest, use_wandb)

    print(headline.to_string(index=False))
    print(f"\nreport   {dest / 'report.html'}\ndataset  {meta['dataset_hash']}")
    return {"run_id": run_id, "path": str(dest), "meta": meta, "table": headline, "panel": panel}


def _wandb_requested(cfg, use_wandb: bool | None) -> bool:
    return cfg.wandb.get("enabled", False) if use_wandb is None else use_wandb


def _log_wandb(cfg, meta: dict, headline: pd.DataFrame, dest: Path, use_wandb: bool | None) -> None:
    """Failures here must not take the run down — the report is already on disk by the
    time this is called. But a run the config asked to track and that silently isn't
    tracked "didn't happen" (golden rule 1), so failures are loud, not a footnote."""
    if not _wandb_requested(cfg, use_wandb):
        return
    try:
        import wandb

        with wandb.init(
            project=cfg.wandb.get("project", "axiom"),
            entity=cfg.wandb.get("entity"),
            name=meta["run_id"],
            job_type="eval",
            config={**meta, "table": None},
        ) as w:
            w.log({"results": wandb.Table(dataframe=headline)})
            for row in headline.to_dict("records"):
                tag = f"{row['model']}/{row['tf']}/h{row['horizon']}"
                w.log({f"{tag}/{k}": v for k, v in row.items()
                       if isinstance(v, int | float) and k not in ("horizon",)})
            w.save(str(dest / "report.html"), base_path=str(dest.parent), policy="now")
    except Exception as exc:  # noqa: BLE001 — see docstring: loud, but not fatal
        print(
            f"\n*** W&B LOGGING FAILED for {meta['run_id']}: {exc}\n"
            f"*** The report under {dest} is intact — re-log it, do not re-run.",
            file=sys.stderr, flush=True,
        )


__all__ = ["build_panel", "environment_info", "finalize", "run", "seed_everything"]
