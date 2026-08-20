"""Config hashing: the identity of an experiment, independent of which run produced it."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from axiom.config.settings import VOLATILE_FIELDS

#: How many hex characters of the digest appear in artifact paths. Twelve is enough that a
#: collision across this project's lifetime is not a thing to think about.
SHORT_LEN = 12


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize deterministically: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: BaseModel, *, short: bool = True) -> str:
    """Hash a config, ignoring fields that identify the run rather than the experiment.

    Two runs of the same experiment on different backends hash the same, which is what makes a
    resumed cloud run comparable to the local run that produced the reference numbers.
    """
    payload = {k: v for k, v in cfg.model_dump(mode="json").items() if k not in VOLATILE_FIELDS}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:SHORT_LEN] if short else digest
