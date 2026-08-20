"""Axiom: a K-line foundation model built discretize-then-autoregress."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("axiom")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
