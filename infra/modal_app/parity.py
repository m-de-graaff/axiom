"""Parity + speed of the KV cache on real weights and a real GPU (P4-02, P4-04).

`tests/test_parity.py` proves the cached loop on a tiny random-weight CPU config in
CI. This is the other half CLAUDE.md asks for: the CUDA leg, on the checkpoints that
actually get evaluated, plus the before/after numbers for P4-09.

    modal run infra/modal_app/parity.py
    modal run infra/modal_app/parity.py --model axiom-zero-base --samples 64
"""

import pathlib

import modal

app = modal.App("axiom-parity")

REPO = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/root")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "pandas", "numpy", "scipy", "einops", "huggingface_hub", "safetensors", "tqdm",
    )
    .add_local_dir(str(REPO / "packages" / "axiom_model" / "axiom_model"), "/root/axiom_model")
    .add_local_dir(str(REPO / "packages" / "axiom_data" / "axiom_data"), "/root/axiom_data")
    .add_local_dir(str(REPO / "packages" / "axiom_eval" / "axiom_eval"), "/root/axiom_eval")
)


@app.function(image=image, gpu="L4", timeout=60 * 60)
def check(model_name: str = "axiom-zero-small", samples: int = 16, horizon: int = 24) -> dict:
    """The CUDA leg. Same measurement the XTX runs via `scripts/rocm_check.py`."""
    import sys

    sys.path.insert(0, "/root")
    from axiom_eval.bench import parity_and_speed

    return parity_and_speed(model_name, samples, horizon)


@app.local_entrypoint()
def main(model: str = "axiom-zero-small", samples: int = 16, horizon: int = 24):
    result = check.remote(model_name=model, samples=samples, horizon=horizon)
    for key, value in result.items():
        print(f"{key:>18}: {value}")
    if not result["token_identical"]:
        print("\nNOT token-identical — do not ship this cache (CLAUDE.md rule 2).")
