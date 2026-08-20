"""Local checkpoint writing and reading.

Writes are atomic and content-addressed by sha256. A kernel killed mid-write must leave either
the previous checkpoint or a complete new one, never a truncated file that resumes into garbage.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

#: How many step directories to keep on local disk. The durable copy lives on the Hub.
KEEP_LAST_K = 3

STATE_FILENAME = "state.pt"
META_FILENAME = "meta.json"


@dataclass
class TrainState:
    """Everything needed to resume a run as though it had never stopped.

    v0.0 carries a scalar accumulator where v0.7 carries model and optimizer tensors. The shape
    of this dataclass is the contract; the payload grows, the machinery around it does not.
    """

    step: int
    acc: float
    rng: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    run_id: str = ""
    schema_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainState:
        return cls(**payload)


def step_dirname(step: int) -> str:
    return f"step_{step:08d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(state: TrainState, root: Path) -> Path:
    """Write ``state`` under ``root/step_XXXXXXXX/`` atomically. Returns the step directory."""
    step_dir = root / step_dirname(state.step)
    step_dir.mkdir(parents=True, exist_ok=True)

    state_path = step_dir / STATE_FILENAME
    tmp_path = state_path.with_suffix(".pt.tmp")
    torch.save(state.to_dict(), tmp_path)
    os.replace(tmp_path, state_path)

    meta = {
        "step": state.step,
        "sha256": sha256_file(state_path),
        "wall_time": time.time(),
        "config_hash": state.config_hash,
        "run_id": state.run_id,
        "schema_version": state.schema_version,
    }
    meta_path = step_dir / META_FILENAME
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_meta, meta_path)

    return step_dir


def load_checkpoint(step_dir: Path) -> TrainState:
    """Read a checkpoint back, verifying it against the sha256 its meta recorded."""
    state_path = step_dir / STATE_FILENAME
    meta = json.loads((step_dir / META_FILENAME).read_text(encoding="utf-8"))

    actual = sha256_file(state_path)
    if actual != meta["sha256"]:
        raise ValueError(
            f"{state_path}: sha256 mismatch (meta says {meta['sha256']}, file is {actual})"
        )

    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    return TrainState.from_dict(payload)


def list_checkpoints(root: Path) -> list[Path]:
    """Step directories under ``root``, oldest first.

    A missing root is an empty list rather than an error: the first launch of a cloud run asks
    before anything has been written.
    """
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_"))


def latest_checkpoint(root: Path) -> Path | None:
    found = list_checkpoints(root)
    return found[-1] if found else None


def prune_checkpoints(root: Path, keep: int = KEEP_LAST_K) -> list[Path]:
    """Delete all but the newest ``keep`` step directories. Returns what was removed."""
    found = list_checkpoints(root)
    doomed = found[:-keep] if keep > 0 else found
    for path in doomed:
        shutil.rmtree(path, ignore_errors=True)
    return doomed
