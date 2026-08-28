"""The eval harness on Modal (P2-13): the second machine.

The corpus already lives on the `axiom-data` volume, so this reads exactly the bars
the local run reads and must land on the same dataset hash. Cross-machine
reproduction means: identical anchors and identical baseline numbers (both are
seeded, neither touches the GPU), and model metrics within tolerance — MC sampling
is not bit-identical across devices and never will be.

One container per (model, timeframe): the grid is embarrassingly parallel, the
GPU-seconds are the same either way, and the wall clock drops by ~9x. Per-window
seeding is what makes sharding safe — how the work is split cannot change a number.
The panels come home, and the report is assembled locally.

    modal run infra/modal_app/eval.py                          # the full config
    modal run infra/modal_app/eval.py --timeframes 1h --max-anchors 4    # smoke
"""

import io
import pathlib

import modal

app = modal.App("axiom-eval")

REPO = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/root")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",  # CUDA build inside Modal — local installs stay manual (CPU/ROCm)
        "pandas", "pyarrow", "duckdb", "numpy", "scipy", "einops", "lightgbm",
        "huggingface_hub", "safetensors", "wandb", "pyyaml", "tqdm",
    )
    .add_local_dir(str(REPO / "packages" / "axiom_model" / "axiom_model"), "/root/axiom_model")
    .add_local_dir(str(REPO / "packages" / "axiom_data" / "axiom_data"), "/root/axiom_data")
    .add_local_dir(str(REPO / "packages" / "axiom_eval" / "axiom_eval"), "/root/axiom_eval")
    .add_local_dir(str(REPO / "configs"), "/root/configs")
)
data_vol = modal.Volume.from_name("axiom-data", create_if_missing=True)


@app.function(image=image)
def plan(config: str, models: str, timeframes: str) -> list[list[str]]:
    """Resolve the config remotely: `modal run` executes the local entrypoint in the
    Modal CLI's own environment, which has no pandas/yaml."""
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")
    from axiom_eval.panel import load_config

    cfg, _ = load_config(config)
    wanted = models.split(",") if models else [
        *[b for b, on in cfg.baselines.items() if on], *cfg.models
    ]
    tfs = timeframes.split(",") if timeframes else cfg.panel["timeframes"]
    return [[model, tf] for model in wanted for tf in tfs]


@app.function(image=image, gpu="L4", timeout=6 * 60 * 60, volumes={"/data": data_vol})
def shard(
    config: str,
    model: str,
    tf: str,
    symbols: list[str] | None = None,
    max_anchors: int | None = None,
) -> dict:
    """Score one (model, timeframe) and return its slice of the panel as parquet."""
    import os
    import sys
    import time

    sys.path.insert(0, "/root")
    os.chdir("/root")  # config paths inside the YAMLs are repo-relative
    import torch
    from axiom_eval.panel import load_config
    from axiom_eval.run import build_panel, environment_info, seed_everything

    cfg, data_cfg = load_config(config)
    if max_anchors is not None:
        cfg.panel["max_anchors"] = max_anchors
    seed_everything(cfg.seed)

    started = time.time()
    panel = build_panel(
        cfg, data_cfg, pathlib.Path("/data/parquet"),
        models=[model], timeframes=[tf], symbols=symbols,
    )
    buffer = io.BytesIO()
    panel.to_parquet(buffer, index=False)
    return {
        "model": model,
        "tf": tf,
        "rows": len(panel),
        "seconds": round(time.time() - started, 1),
        "environment": environment_info(None) | {"gpu": torch.cuda.get_device_name(0)},
        "panel": buffer.getvalue(),
    }


@app.function(image=image, volumes={"/data": data_vol}, secrets=[modal.Secret.from_name("wandb")])
def assemble(config: str, panels: list[bytes], environment: dict, use_wandb: bool) -> dict:
    """Concatenate the shards and write the report — on the volume's machine, so the
    dataset hash in the report is the one the bars actually came from."""
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")
    import pandas as pd
    from axiom_eval.panel import load_config
    from axiom_eval.run import finalize

    cfg, data_cfg = load_config(config)
    out = finalize(
        cfg, data_cfg,
        pd.concat([pd.read_parquet(io.BytesIO(b)) for b in panels], ignore_index=True),
        config,
        out_dir=pathlib.Path("/tmp/reports"),
        datasets_dir=pathlib.Path("/data/datasets"),
        use_wandb=use_wandb,
        environment=environment,
    )
    dest = pathlib.Path(out["path"])
    return {
        "run_id": out["run_id"],
        "dataset_hash": out["meta"]["dataset_hash"],
        "table": out["table"].to_csv(index=False),
        "files": {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()},
    }


@app.local_entrypoint()
def main(
    config: str = "configs/eval/default.yaml",
    models: str = "",
    timeframes: str = "",
    symbols: str = "",
    max_anchors: int = 0,
    wandb: bool = True,
):
    pairs = plan.remote(config, models, timeframes)
    args = [
        (config, model, tf, symbols.split(",") if symbols else None, max_anchors or None)
        for model, tf in pairs
    ]
    print(f"{len(args)} shards")

    panels, env = [], None
    for result in shard.starmap(args):
        print(f"  {result['tf']:>4} {result['model']:<20} {result['rows']:>7} rows "
              f"{result['seconds']:>8.1f}s")
        panels.append(result["panel"])
        env = result["environment"]

    out = assemble.remote(config, panels, env, wandb)
    dest = REPO / "reports" / out["run_id"]
    dest.mkdir(parents=True, exist_ok=True)
    for name, blob in out["files"].items():
        (dest / name).write_bytes(blob)
    print(out["table"])
    print(f"dataset {out['dataset_hash']}\nwrote   {dest}")
