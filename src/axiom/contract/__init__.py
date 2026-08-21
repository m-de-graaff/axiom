"""The versioned preprocessing contract (v0.4, ADR-0020).

Four functions, and every consumer uses these four. Training pre-tokenization (v0.6) and the
inference Predictor (v0.9) import the same :func:`transform`, so a drift between what the model
was trained on and what it is asked to predict from is a test failure rather than a silent skew.

    from axiom.contract import load_constants, load_spec, transform

    spec = load_spec("contract_geo_v1")
    block = transform(bars, spec, load_constants(), asset_class="crypto", frequency="1h")

There is no second path. A version that needs different features bumps
:data:`SCHEMA_VERSION`, refits the constants and re-cuts the snapshots; it does not add an
alternate transform beside this one.
"""

from axiom.contract.inverse import inverse
from axiom.contract.spec import SCHEMA_VERSION, load_constants, load_spec
from axiom.contract.transform import transform

__all__ = ["SCHEMA_VERSION", "inverse", "load_constants", "load_spec", "transform"]
