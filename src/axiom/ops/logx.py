"""Logging and experiment tracking.

Tracking is best-effort by design: a cloud kernel that dies because the dashboard was
unreachable has lost 12 hours of quota to a metrics call.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from typing import Any

from pydantic import BaseModel

from axiom import __version__
from axiom.config.hashing import config_hash

log = logging.getLogger("axiom")

_tracking_active = False


def setup_logging(level: str | None = None) -> None:
    """Configure console logging once. Level comes from ``AXIOM_LOG_LEVEL``, default INFO."""
    resolved = (level or os.environ.get("AXIOM_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def git_commit() -> str:
    """The commit this code came from.

    Cloud kernels get it from ``AXIOM_GIT_COMMIT`` because pip-installing from git leaves no
    working tree to ask.
    """
    env = os.environ.get("AXIOM_GIT_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_provenance(cfg: BaseModel) -> dict[str, Any]:
    """Everything needed to tie a number back to the code and config that produced it."""
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:  # pragma: no cover - torch is a hard dep of every run path
        torch_version = "absent"

    return {
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "axiom_version": __version__,
        "python": platform.python_version(),
        "torch": torch_version,
        "platform": platform.platform(),
        "backend_tag": getattr(cfg, "backend_tag", "unknown"),
    }


def init_tracking(cfg: BaseModel) -> bool:
    """Start a trackio run, or log why it did not start and carry on.

    Returns whether tracking is live, so callers can skip building metric payloads that nothing
    will read.
    """
    global _tracking_active

    setup_logging()
    prov = run_provenance(cfg)
    for key, value in prov.items():
        log.info("provenance %s=%s", key, value)

    if os.environ.get("AXIOM_DISABLE_TRACKING"):
        log.info("tracking disabled by AXIOM_DISABLE_TRACKING")
        _tracking_active = False
        return False

    try:
        import trackio

        trackio.init(
            project="axiom",
            name=getattr(cfg, "run_id", None),
            config={**cfg.model_dump(mode="json"), **prov},
        )
        _tracking_active = True
    except Exception as exc:
        log.warning("tracking unavailable (%s); continuing without it", exc)
        _tracking_active = False
    return _tracking_active


def log_metrics(metrics: dict[str, Any]) -> None:
    """Forward metrics to trackio if it is live. Silent no-op otherwise."""
    if not _tracking_active:
        return
    try:
        import trackio

        trackio.log(metrics)
    except Exception as exc:
        log.warning("tracking log failed (%s); continuing", exc)


def finish_tracking() -> None:
    """Close the tracking run so its last metrics flush."""
    global _tracking_active
    if not _tracking_active:
        return
    try:
        import trackio

        trackio.finish()
    except Exception as exc:
        log.warning("tracking finish failed (%s)", exc)
    _tracking_active = False
