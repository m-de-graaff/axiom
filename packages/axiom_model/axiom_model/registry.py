"""The only place that maps Axiom model names to weights (CLAUDE.md: "Configs are law").

Nothing else in the codebase may hardcode an HF repo id or a checkpoint path.
Fine-tunes get added here as new entries (`axiom-ft-102m-crypto1-512-v0` -> a
Modal volume path or HF repo), never by passing a path around.
"""

from dataclasses import dataclass

import torch

from .predictor import AxiomPredictor
from .tokenizer import AxiomTokenizer
from .transformer import Axiom


@dataclass(frozen=True)
class ModelSpec:
    """Where a named model's weights live, and the context it was trained for."""

    model_source: str
    tokenizer_source: str
    max_context: int
    params_m: float


# `axiom-zero-*` = upstream Kronos weights, unmodified. Pairings and context
# lengths come from the upstream model card (vendor/kronos/README.md).
REGISTRY: dict[str, ModelSpec] = {
    "axiom-zero-mini": ModelSpec(
        model_source="NeoQuasar/Kronos-mini",
        tokenizer_source="NeoQuasar/Kronos-Tokenizer-2k",
        max_context=2048,
        params_m=4.1,
    ),
    "axiom-zero-small": ModelSpec(
        model_source="NeoQuasar/Kronos-small",
        tokenizer_source="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        params_m=24.7,
    ),
    "axiom-zero-base": ModelSpec(
        model_source="NeoQuasar/Kronos-base",
        tokenizer_source="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        params_m=102.3,
    ),
}


def resolve(name: str) -> ModelSpec:
    """Look up a registered model name, or fail with the list of valid names."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}") from None


def default_device() -> str:
    """CUDA (also ROCm, which torch presents as `cuda`) when present, else CPU.

    A local GPU is optional everywhere in this repo — see CLAUDE.md.
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_predictor(name: str, device: str | None = None, **kwargs) -> AxiomPredictor:
    """Load a registered model + its tokenizer and return a ready AxiomPredictor."""
    spec = resolve(name)
    tokenizer = AxiomTokenizer.from_pretrained(spec.tokenizer_source)
    model = Axiom.from_pretrained(spec.model_source)
    return AxiomPredictor(
        model,
        tokenizer,
        device=device or default_device(),
        max_context=spec.max_context,
        **kwargs,
    )


__all__ = ["REGISTRY", "ModelSpec", "default_device", "load_predictor", "resolve"]
