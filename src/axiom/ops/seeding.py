"""Deterministic seeding, and RNG state capture/restore across a process boundary.

The whole resume story rests on this module. A checkpoint that restores weights but not RNG
resumes into a different random stream, and the divergence is invisible until evaluation.
"""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every generator this project draws from.

    torch state is serialized to bytes rather than kept as a tensor so the checkpoint round-trips
    through ``torch.save`` without depending on tensor-sharing semantics.
    """
    buf = io.BytesIO()
    torch.save(torch.get_rng_state(), buf)
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": buf.getvalue(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        cuda_buf = io.BytesIO()
        torch.save(torch.cuda.get_rng_state_all(), cuda_buf)
        state["torch_cuda"] = cuda_buf.getvalue()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore what ``capture_rng_state`` snapshotted.

    CUDA state captured on a GPU box is skipped on a CPU box rather than raising, because the
    local determinism drill and the cloud run share this code path.
    """
    random.setstate(_as_tuple(state["python"]))
    np.random.set_state(_as_tuple(state["numpy"]))
    torch.set_rng_state(torch.load(io.BytesIO(state["torch"]), weights_only=False))
    cuda = state.get("torch_cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch.load(io.BytesIO(cuda), weights_only=False))


def _as_tuple(value: Any) -> Any:
    """Undo the list-ification that JSON-ish round trips inflict on RNG state tuples."""
    if isinstance(value, list):
        return tuple(_as_tuple(v) for v in value)
    return value
