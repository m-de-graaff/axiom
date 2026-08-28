"""Axiom predictor — compatibility bridge over the ported Kronos predictor.

Thin subclass on purpose (P0-06): the generation loop stays upstream's until a
change can be defended by `tests/test_parity.py`. Refactors go into `_kronos.py`,
never here.
"""

from ._kronos import KronosPredictor


class AxiomPredictor(KronosPredictor):
    """Axiom's forecast interface. Same call signature as upstream KronosPredictor."""


__all__ = ["AxiomPredictor", "KronosPredictor"]
