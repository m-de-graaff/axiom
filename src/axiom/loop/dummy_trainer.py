"""The v0.0 stand-in trainer.

Design rule for this whole module: the loop code may not know it is a dummy. Every piece of
checkpoint, resume, and dispatch machinery exercised here is the same code the tokenizer (v0.5)
and the AR decoder (v0.7) will run unchanged. What gets swapped later is the step function and
the payload inside ``TrainState``, not anything around them.

``acc`` accumulates draws from a seeded generator, so any divergence between an uninterrupted run
and a resumed one shows up in the final float. That is the point: a checkpoint that restores
weights but loses RNG position would pass a shape check and fail here.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch

from axiom.config.hashing import config_hash
from axiom.config.settings import AxiomSettings, LoopConfig
from axiom.ops import hub
from axiom.ops.checkpoint import (
    TrainState,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from axiom.ops.logx import finish_tracking, init_tracking, log_metrics
from axiom.ops.seeding import capture_rng_state, restore_rng_state, seed_all

log = logging.getLogger("axiom.loop")

#: Where local checkpoints go. Gitignored; the durable copy is on the Hub.
CHECKPOINT_ROOT = Path("checkpoints")


class KilledAtStep(SystemExit):
    """Fault injection for the kill drills. Exit code 137 mimics a SIGKILL-ed kernel."""

    def __init__(self, step: int) -> None:
        super().__init__(137)
        self.step = step


def _kill_at_step() -> int | None:
    raw = os.environ.get("AXIOM_KILL_AT_STEP")
    return int(raw) if raw else None


def checkpoint_root(run_id: str) -> Path:
    return CHECKPOINT_ROOT / run_id


def initial_state(cfg: LoopConfig) -> TrainState:
    """A fresh run at step 0, with the RNG seeded and captured before a single draw."""
    seed_all(cfg.seed)
    return TrainState(
        step=0,
        acc=0.0,
        rng=capture_rng_state(),
        config_hash=config_hash(cfg),
        run_id=cfg.run_id,
        schema_version=cfg.schema_version,
    )


def resolve_start_state(
    cfg: LoopConfig,
    resume: bool,
    settings: AxiomSettings | None = None,
) -> TrainState:
    """Decide where this process starts: fresh, from local disk, or from the Hub.

    Local disk is preferred over the Hub when both have a checkpoint, because a local one is at
    least as new and costs no network. On a cloud kernel the local directory is always empty, so
    that branch simply never fires there.
    """
    if not resume:
        return initial_state(cfg)

    root = checkpoint_root(cfg.run_id)
    local = latest_checkpoint(root)
    if local is not None:
        state = load_checkpoint(local)
        log.info("resuming from local checkpoint %s (step %d)", local, state.step)
    else:
        state = hub.pull_latest(cfg.run_id, root, settings=settings)
        if state is None:
            log.info("nothing to resume; starting fresh")
            return initial_state(cfg)

    expected = config_hash(cfg)
    if state.config_hash != expected:
        raise ValueError(
            f"checkpoint config hash {state.config_hash} does not match config {expected}. "
            "Resuming a different experiment into this run would silently corrupt it."
        )

    restore_rng_state(state.rng)
    return state


def _step(state: TrainState, generator: torch.Generator, sleep_s: float) -> None:
    """One training step's worth of state change.

    v0.7 replaces the body of this function and nothing else in the file.
    """
    state.acc += torch.rand((), generator=generator).item()
    state.step += 1
    if sleep_s:
        time.sleep(sleep_s)


def run(
    cfg: LoopConfig,
    resume: bool = False,
    settings: AxiomSettings | None = None,
    push: bool = True,
) -> TrainState:
    """Run ``cfg`` to completion, checkpointing every ``save_every`` steps.

    Returns the final state. Raises ``KilledAtStep`` if ``AXIOM_KILL_AT_STEP`` names a step in
    range, which is how both the local drill and the Kaggle drill simulate a session death.
    """
    settings = settings or AxiomSettings()
    init_tracking(cfg)

    state = resolve_start_state(cfg, resume, settings=settings)
    root = checkpoint_root(cfg.run_id)
    kill_at = _kill_at_step()
    futures = []

    # The generator is seeded once and then carried in the checkpoint's RNG state, so a resumed
    # run continues the same stream rather than restarting it.
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    if state.step > 0:
        # Replay the generator to the checkpointed position. Cheap here, and the honest thing:
        # the alternative is checkpointing the generator separately, which v0.7 will do when
        # replay stops being free.
        for _ in range(state.step):
            torch.rand((), generator=generator)

    log.info("starting at step %d of %d (%s)", state.step, cfg.total_steps, cfg.backend_tag)

    try:
        while state.step < cfg.total_steps:
            if kill_at is not None and state.step == kill_at:
                log.warning("fault injection: dying at step %d", state.step)
                raise KilledAtStep(state.step)

            _step(state, generator, cfg.sleep_s)

            if state.step % cfg.save_every == 0 or state.step == cfg.total_steps:
                state.rng = capture_rng_state()
                step_dir = save_checkpoint(state, root)
                prune_checkpoints(root)
                log_metrics({"step": state.step, "acc": state.acc})
                if push:
                    futures.append(
                        hub.push_checkpoint(step_dir, cfg.run_id, state.step, settings=settings)
                    )
    finally:
        for future in futures:
            future.result()
        finish_tracking()

    log.info("finished at step %d with acc=%r", state.step, state.acc)
    return state
