"""Hugging Face Hub transport for checkpoints.

The Hub is the only durable surface a killed kernel leaves behind, so ``latest.json`` is the
resume pointer and everything else is addressed relative to it.

No token ever reaches a log line or an exception message here. Failures report the repo and the
path; whether the token was wrong is something you find out from the Hub, not from our output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from axiom.config.settings import AxiomSettings
from axiom.ops.checkpoint import (
    META_FILENAME,
    STATE_FILENAME,
    TrainState,
    load_checkpoint,
    sha256_file,
    step_dirname,
)

log = logging.getLogger("axiom.hub")

#: Everything v0.0 writes lives under this prefix, so v0.1+ can share the repo without collision.
LOOP_PREFIX = "loop-test"

LATEST_FILENAME = "latest.json"


def _api(settings: AxiomSettings) -> HfApi:
    """An authenticated client, or a clear failure before any network call happens."""
    if settings.hf_token is None:
        raise RuntimeError(
            "AXIOM_HF_TOKEN is not set. Put it in .env locally, or attach it as a secret in the "
            "cloud kernel. See docs/RUNBOOK.md."
        )
    return HfApi(token=settings.hf_token.get_secret_value())


def run_prefix(run_id: str) -> str:
    return f"{LOOP_PREFIX}/{run_id}"


def push_checkpoint(
    local_step_dir: Path,
    run_id: str,
    step: int,
    settings: AxiomSettings | None = None,
) -> Any:
    """Upload a step directory and repoint ``latest.json`` at it.

    The folder upload is non-blocking so a slow network does not stall training; the pointer
    write is not, because a ``latest.json`` naming a half-uploaded directory is worse than a
    stale one. Callers must await the returned future before the process exits.
    """
    settings = settings or AxiomSettings()
    api = _api(settings)
    repo_id = settings.runs_repo_id
    path_in_repo = f"{run_prefix(run_id)}/{step_dirname(step)}"

    future = api.upload_folder(
        folder_path=str(local_step_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        run_as_future=True,
    )

    pointer = {
        "step": step,
        "path_in_repo": path_in_repo,
        "sha256": sha256_file(local_step_dir / STATE_FILENAME),
    }
    api.upload_file(
        path_or_fileobj=json.dumps(pointer, indent=2, sort_keys=True).encode("utf-8"),
        path_in_repo=f"{run_prefix(run_id)}/{LATEST_FILENAME}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    log.info("pushed step %d to %s:%s", step, repo_id, path_in_repo)
    return future


def pull_latest(
    run_id: str,
    dest_root: Path,
    settings: AxiomSettings | None = None,
) -> TrainState | None:
    """Fetch the newest checkpoint for ``run_id``, or ``None`` if the run has never checkpointed.

    A missing pointer is the normal first-launch case and returns ``None``. A pointer whose
    sha256 does not match the downloaded bytes raises, because resuming from a corrupt state is
    the one failure that would not announce itself.
    """
    settings = settings or AxiomSettings()
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    repo_id = settings.runs_repo_id

    try:
        pointer_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{run_prefix(run_id)}/{LATEST_FILENAME}",
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:
        log.info("no resume pointer for %s in %s (%s)", run_id, repo_id, type(exc).__name__)
        return None

    pointer = json.loads(Path(pointer_path).read_text(encoding="utf-8"))
    step_dir = dest_root / step_dirname(pointer["step"])
    step_dir.mkdir(parents=True, exist_ok=True)

    for filename in (STATE_FILENAME, META_FILENAME):
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=f"{pointer['path_in_repo']}/{filename}",
            repo_type="dataset",
            token=token,
        )
        (step_dir / filename).write_bytes(Path(downloaded).read_bytes())

    actual = sha256_file(step_dir / STATE_FILENAME)
    if actual != pointer["sha256"]:
        raise ValueError(
            f"{repo_id}:{pointer['path_in_repo']} sha256 mismatch "
            f"(pointer says {pointer['sha256']}, downloaded {actual})"
        )

    state = load_checkpoint(step_dir)
    log.info("resumed %s from step %d", run_id, state.step)
    return state
