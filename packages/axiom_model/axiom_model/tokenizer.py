"""Axiom tokenizer — compatibility bridge over the ported Kronos tokenizer.

Thin subclass on purpose (P0-06): `AxiomTokenizer.from_pretrained(...)` must keep
loading upstream `NeoQuasar/Kronos-Tokenizer-*` checkpoints. Refactors go into
`_kronos.py` / `module.py`, never here.
"""

from ._kronos import KronosTokenizer


class AxiomTokenizer(KronosTokenizer):
    """Axiom's tokenizer. Weight-compatible with upstream Kronos tokenizers."""


__all__ = ["AxiomTokenizer", "KronosTokenizer"]
