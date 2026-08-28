"""The eval harness on Modal (P2-13): the second machine.

The corpus already lives on the `axiom-data` volume, so this reads exactly the
bars the local run reads and must land on the same dataset hash. Cross-machine
reproduction means: identical anchors and identical baseline numbers (both are
seeded, neither touches the GPU), and model metrics within tolerance — MC sampling
is not bit-identical across devices and never will be.

    modal run infra/modal_app/eval.py                       # full config, L4
    modal run infra/modal_app/eval.py --models "persistence,ewma,lightgbm" --gpu ""
    modal run infra/modal_app/eval.py --timeframes 1h --max-anchors 4   # smoke
"""

import pathlib

import modal

app = modal.App("axiom-eval")

REPO = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/root")
LOCAL_REPORTS = REPO / "reports"

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


@app.function(
    image=image,
    gpu="L4",
    timeout=8 * 60 * 60,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def evaluate(
    config: str = "configs/eval/default.yaml",
    models: list[str] | None = None,
    timeframes: list[str] | None = None,
    symbols: list[str] | None = None,
    max_anchors: int | None = None,
    use_wandb: bool = True,
) -> dict:
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")  # config paths inside the YAMLs are repo-relative
    from axiom_eval.run import run

    out = run(
        config,
        root=pathlib.Path("/data/parquet"),
        out_dir=pathlib.Path("/tmp/reports"),
        datasets_dir=pathlib.Path("/data/datasets"),
        models=models,
        timeframes=timeframes,
        symbols=symbols,
        max_anchors=max_anchors,
        use_wandb=use_wandb,
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
    result = evaluate.remote(
        config=config,
        models=models.split(",") if models else None,
        timeframes=timeframes.split(",") if timeframes else None,
        symbols=symbols.split(",") if symbols else None,
        max_anchors=max_anchors or None,
        use_wandb=wandb,
    )
    dest = LOCAL_REPORTS / result["run_id"]
    dest.mkdir(parents=True, exist_ok=True)
    for name, blob in result.pop("files").items():
        (dest / name).write_bytes(blob)
    print(result["table"])
    print(f"dataset {result['dataset_hash']}\nwrote   {dest}")
