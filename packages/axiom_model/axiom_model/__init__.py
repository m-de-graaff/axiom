"""axiom_model — see docs/AXIOM_BUILD_ORDER.md and CLAUDE.md.

Kronos core (Shi et al. 2025, MIT — see NOTICE) ported into `_kronos.py` +
`module.py`, with `Axiom*` as the stable public API over it. Internals get
refactored behind that API; the `Axiom*` classes keep loading upstream
`NeoQuasar/Kronos-*` checkpoints.
"""

from .predictor import AxiomPredictor
from .registry import REGISTRY, ModelSpec, default_device, load_predictor, resolve
from .tokenizer import AxiomTokenizer
from .transformer import Axiom

__all__ = [
    "REGISTRY",
    "Axiom",
    "AxiomPredictor",
    "AxiomTokenizer",
    "ModelSpec",
    "default_device",
    "load_predictor",
    "resolve",
]
