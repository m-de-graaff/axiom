"""Axiom decoder-only transformer — compatibility bridge over the ported Kronos core.

Thin subclass on purpose (P0-06): `Axiom.from_pretrained("NeoQuasar/Kronos-base")`
must keep loading upstream weights. Refactors go into `_kronos.py` / `module.py`,
never here.
"""

from ._kronos import Kronos


class Axiom(Kronos):
    """Axiom's K-line foundation model. Weight-compatible with upstream Kronos."""


__all__ = ["Axiom", "Kronos"]
