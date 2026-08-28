"""Modal training app (P3-05). Skeleton — verify decorator/arg names against
current Modal docs before first use.

usage:
  modal run infra/modal_app/train.py::train --config-yaml "$(cat configs/finetune/crypto_v0.yaml)"
"""

import modal

app = modal.App("axiom-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",  # CUDA build inside Modal — local installs stay manual (CPU/ROCm)
        "pandas", "pyarrow", "numpy", "einops",
        "huggingface_hub", "safetensors", "wandb", "pyyaml",
    )
    .add_local_dir("packages", "/root/packages")
)
data_vol = modal.Volume.from_name("axiom-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("axiom-ckpts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=12 * 60 * 60,
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(config_yaml: str):
    import sys

    sys.path.insert(0, "/root/packages/axiom_model")
    # TODO P3-05: from axiom_model.train.finetune import run
    # run(config_yaml, data_root="/data", ckpt_root="/ckpts")
    raise NotImplementedError("P3-05: wire axiom_model.train.finetune.run")
