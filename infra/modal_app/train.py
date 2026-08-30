"""Modal training app (P3-05) — `axiom_model.train.finetune.run` on the volumes.

usage:
  modal run --detach infra/modal_app/train.py --config configs/finetune/crypto_v0.yaml

The entrypoint reads the config locally and ships its *content* — passing YAML
through the shell as an argument mangles it on Windows.

GPU defaults to the full-run policy (A100-80GB). For subset fine-tunes and smoke
runs, set AXIOM_TRAIN_GPU before `modal run` (read at app build time, locally):

  AXIOM_TRAIN_GPU=L4 modal run infra/modal_app/train.py::train --config-yaml ...

The repo's `configs/` ships into the image so the finetune YAML's relative
`data:` path (and the data config's `universe:` path) resolve from /root.
Checkpoints land on the `axiom-ckpts` volume under /ckpts/{run_name}/.
"""

import os
import pathlib

import modal

app = modal.App("axiom-train")

REPO = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/root")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",  # CUDA build inside Modal — local installs stay manual (CPU/ROCm)
        "pandas", "pyarrow", "numpy", "einops",
        "huggingface_hub", "safetensors", "wandb", "pyyaml",
        "duckdb",  # axiom_data.store
    )
    .workdir("/root")
    .add_local_dir(str(REPO / "packages"), "/root/packages")
    .add_local_dir(str(REPO / "configs"), "/root/configs")
)
data_vol = modal.Volume.from_name("axiom-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("axiom-ckpts", create_if_missing=True)


@app.function(
    image=image,
    gpu=os.environ.get("AXIOM_TRAIN_GPU", "A100-80GB"),
    timeout=12 * 60 * 60,
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(config_yaml: str, stage: str = "all", git_sha: str = "") -> dict:
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, "/root/packages/axiom_model")
    sys.path.insert(0, "/root/packages/axiom_data")
    if git_sha:  # containers get the code, not the .git — see run.py/_git_sha
        os.environ["AXIOM_GIT_SHA"] = git_sha
    from axiom_model.train.finetune import run

    cfg = Path(tempfile.mkdtemp()) / "finetune.yaml"
    cfg.write_text(config_yaml, encoding="utf-8")
    out = run(
        cfg,
        stage=stage,
        root=Path("/data/parquet"),
        datasets_dir=Path("/data/datasets"),
        out_dir=Path("/ckpts"),
        on_stage_end=lambda _stage: ckpt_vol.commit(),  # survive a later-stage crash
        on_epoch_end=ckpt_vol.commit,  # preemption loses at most one epoch (resume)
    )
    ckpt_vol.commit()
    return {"meta": out["meta"], "results": out["results"]}


@app.local_entrypoint()
def main(config: str = "configs/finetune/crypto_v0.yaml", stage: str = "all"):
    import subprocess

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        sha = ""
    config_yaml = pathlib.Path(config).read_text(encoding="utf-8")
    out = train.remote(config_yaml=config_yaml, stage=stage, git_sha=sha)
    for stage_name, result in out["results"].items():
        print(f"{stage_name}: best_val_loss {result['best_val_loss']:.4f} -> {result['path']}")
