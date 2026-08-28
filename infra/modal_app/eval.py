"""The eval harness on Modal (P2-13): the second machine.

The corpus already lives on the `axiom-data` volume, so this reads exactly the bars
the local run reads and must land on the same dataset hash. Cross-machine
reproduction means: identical anchors and identical baseline numbers (both are
seeded, neither touches the GPU), and model metrics within tolerance — MC sampling
is not bit-identical across devices and never will be.

Work is split into (model, timeframe, anchor chunk) shards, and **every shard writes
its panel to the volume before returning**. That is not premature engineering: the
first attempt returned panels through the map, one `axiom-zero-base` shard hit the
container timeout, and ~40 L4-hours of finished work went in the bin with it. A
re-run now skips whatever is already on the volume, so a timeout costs one chunk.

    modal run infra/modal_app/eval.py                            # the full config
    modal run infra/modal_app/eval.py --chunks 8                 # smaller shards
    modal run infra/modal_app/eval.py --timeframes 1h --max-anchors 4    # smoke
    modal run infra/modal_app/eval.py::assemble_only --tag default-a15fed4
"""

import pathlib
import subprocess

import modal

app = modal.App("axiom-eval")

REPO = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/root")
PANELS = pathlib.Path("/data/eval_panels")

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


def _setup():
    """Every remote function starts the same way: repo on the path, repo as cwd."""
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")  # config paths inside the YAMLs are repo-relative


@app.function(image=image)
def plan(config: str, models: str, timeframes: str, chunks: int) -> list[list]:
    """Resolve the config remotely: `modal run` executes the local entrypoint in the
    Modal CLI's own environment, which has no pandas/yaml."""
    _setup()
    from axiom_eval.panel import load_config

    cfg, _ = load_config(config)
    wanted = models.split(",") if models else [
        *[b for b, on in cfg.baselines.items() if on], *cfg.models
    ]
    tfs = timeframes.split(",") if timeframes else cfg.panel["timeframes"]
    # Baselines take seconds; only the models are worth splitting across containers.
    return [
        [model, tf, index, chunks if model in cfg.models else 1]
        for model in wanted
        for tf in tfs
        for index in range(chunks if model in cfg.models else 1)
    ]


@app.function(image=image, gpu="L4", timeout=4 * 60 * 60, volumes={"/data": data_vol})
def shard(
    config: str,
    tag: str,
    model: str,
    tf: str,
    index: int,
    chunks: int,
    symbols: list[str] | None = None,
    max_anchors: int | None = None,
) -> dict:
    """Score one (model, timeframe, chunk) and checkpoint it to the volume."""
    _setup()
    import time

    import torch
    from axiom_eval.panel import load_config
    from axiom_eval.run import build_panel, environment_info, seed_everything

    dest = PANELS / tag / f"{model}__{tf}__{index:02d}of{chunks:02d}.parquet"
    if dest.exists():
        return {"model": model, "tf": tf, "index": index, "seconds": 0.0, "cached": True}

    cfg, data_cfg = load_config(config)
    if max_anchors is not None:
        cfg.panel["max_anchors"] = max_anchors
    seed_everything(cfg.seed)

    started = time.time()
    panel = build_panel(
        cfg, data_cfg, pathlib.Path("/data/parquet"),
        models=[model], timeframes=[tf], symbols=symbols,
        chunk=(index, chunks) if chunks > 1 else None,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(dest, index=False)
    data_vol.commit()
    return {
        "model": model,
        "tf": tf,
        "index": index,
        "rows": len(panel),
        "seconds": round(time.time() - started, 1),
        "cached": False,
        "environment": environment_info(None) | {"gpu": torch.cuda.get_device_name(0)},
    }


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def assemble(
    config: str, tag: str, environment: dict | None = None, use_wandb: bool = True
) -> dict:
    """Concatenate every checkpointed chunk and write the report.

    Runs where the volume is, so the dataset hash in the report is the one the bars
    actually came from.
    """
    _setup()
    import pandas as pd
    from axiom_eval.panel import load_config
    from axiom_eval.run import finalize

    data_vol.reload()
    parts = sorted((PANELS / tag).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no chunks under {PANELS / tag}")
    print(f"assembling {len(parts)} chunks")

    cfg, data_cfg = load_config(config)
    out = finalize(
        cfg, data_cfg,
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True),
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
        "chunks": [p.name for p in parts],
        "table": out["table"].to_csv(index=False),
        "files": {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()},
    }


def _write_report(out: dict) -> pathlib.Path:
    dest = REPO / "reports" / out["run_id"]
    dest.mkdir(parents=True, exist_ok=True)
    for name, blob in out["files"].items():
        (dest / name).write_bytes(blob)
    print(out["table"])
    print(f"dataset {out['dataset_hash']}\nwrote   {dest}")
    return dest


def _tag(config: str) -> str:
    """Chunks are keyed by config + code version, so a resume never mixes versions."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return f"{pathlib.Path(config).stem}-{sha or 'nogit'}"


@app.local_entrypoint()
def main(
    config: str = "configs/eval/default.yaml",
    models: str = "",
    timeframes: str = "",
    symbols: str = "",
    chunks: int = 6,
    max_anchors: int = 0,
    wandb: bool = True,
    tag: str = "",
):
    tag = tag or _tag(config)
    args = [
        (config, tag, model, tf, index, total,
         symbols.split(",") if symbols else None, max_anchors or None)
        for model, tf, index, total in plan.remote(config, models, timeframes, chunks)
    ]
    print(f"{len(args)} shards, tag {tag}")

    env = None
    for result in shard.starmap(args, return_exceptions=True):
        if isinstance(result, Exception):  # one dead chunk must not lose the rest
            print(f"  SHARD FAILED: {result}")
            continue
        state = "cached" if result["cached"] else f"{result['seconds']:>8.1f}s"
        print(f"  {result['tf']:>4} {result['model']:<20} chunk {result['index']:>2}  {state}")
        env = result.get("environment") or env

    _write_report(assemble.remote(config, tag, env, wandb))


@app.local_entrypoint()
def assemble_only(config: str = "configs/eval/default.yaml", tag: str = "", wandb: bool = True):
    """Build the report from whatever chunks are already on the volume."""
    _write_report(assemble.remote(config, tag or _tag(config), None, wandb))
