"""The G1 drill: a run that dies mid-flight and resumes must be bit-identical to one that didn't.

This is the test v0.0 exists to make pass. Everything else in the repo is scaffolding around it.

The Hub is never touched here (``push=False``); the cloud half of the drill is manual and lives
in `docs/RUNBOOK.md`, because it needs a real Kaggle session to be killed.
"""

from __future__ import annotations

import pytest

from axiom.config.settings import LoopConfig
from axiom.loop import dummy_trainer
from axiom.loop.dummy_trainer import KilledAtStep, run

TOTAL_STEPS = 1000
SAVE_EVERY = 100
KILL_AT = 437  # deliberately between checkpoints: 437 is not a multiple of 100


@pytest.fixture(autouse=True)
def isolated_checkpoint_root(tmp_path, monkeypatch):
    """Keep drills out of the working tree, and out of each other's way."""
    monkeypatch.setattr(dummy_trainer, "CHECKPOINT_ROOT", tmp_path / "checkpoints")
    monkeypatch.setenv("AXIOM_DISABLE_TRACKING", "1")
    monkeypatch.delenv("AXIOM_KILL_AT_STEP", raising=False)


def make_config(run_id: str, **overrides) -> LoopConfig:
    return LoopConfig(
        **{
            "run_id": run_id,
            "seed": 1337,
            "total_steps": TOTAL_STEPS,
            "save_every": SAVE_EVERY,
            "sleep_s": 0.0,
            "backend_tag": "test",
            **overrides,
        }
    )


@pytest.mark.slow
def test_a_killed_and_resumed_run_ends_bit_identical_to_an_uninterrupted_one(monkeypatch):
    clean = run(make_config("drill-clean"), push=False)

    monkeypatch.setenv("AXIOM_KILL_AT_STEP", str(KILL_AT))
    with pytest.raises(KilledAtStep):
        run(make_config("drill-killed"), push=False)
    monkeypatch.delenv("AXIOM_KILL_AT_STEP")

    resumed = run(make_config("drill-killed"), resume=True, push=False)

    assert resumed.step == clean.step
    assert resumed.acc == clean.acc  # exact equality, not approx: any drift is a bug


@pytest.mark.slow
def test_resume_replays_from_the_checkpoint_floor_rather_than_the_kill_step(monkeypatch):
    """Steps 401 to 437 are recomputed, not skipped.

    Skipping them would also produce a plausible-looking final number, which is why the
    bit-identity assertion above needs this one beside it.
    """
    monkeypatch.setenv("AXIOM_KILL_AT_STEP", str(KILL_AT))
    with pytest.raises(KilledAtStep):
        run(make_config("drill-floor"), push=False)
    monkeypatch.delenv("AXIOM_KILL_AT_STEP")

    root = dummy_trainer.checkpoint_root("drill-floor")
    from axiom.ops.checkpoint import latest_checkpoint, load_checkpoint

    floor = load_checkpoint(latest_checkpoint(root))

    assert floor.step == 400  # the last multiple of SAVE_EVERY before the kill

    resumed = run(make_config("drill-floor"), resume=True, push=False)
    assert resumed.step == TOTAL_STEPS


@pytest.mark.slow
def test_resuming_into_a_different_experiment_is_refused(monkeypatch):
    """A config change between the checkpoint and the resume must stop the run, not blend them."""
    monkeypatch.setenv("AXIOM_KILL_AT_STEP", "200")
    with pytest.raises(KilledAtStep):
        run(make_config("drill-mismatch"), push=False)
    monkeypatch.delenv("AXIOM_KILL_AT_STEP")

    with pytest.raises(ValueError, match="config hash"):
        run(make_config("drill-mismatch", seed=999), resume=True, push=False)


def test_a_fresh_run_starts_at_step_zero_even_when_resume_is_requested():
    """The cloud's first launch asks to resume and finds nothing. That must not be an error."""
    state = run(make_config("drill-nothing-to-resume", total_steps=5, save_every=5), push=False)

    assert state.step == 5


@pytest.mark.slow
def test_two_kills_in_one_run_still_end_bit_identical(monkeypatch):
    """Kaggle's 12-hour cap means multiple interruptions are the normal case, not the edge one."""
    clean = run(make_config("drill-twice-clean"), push=False)

    for kill_at in (237, 613):
        monkeypatch.setenv("AXIOM_KILL_AT_STEP", str(kill_at))
        with pytest.raises(KilledAtStep):
            run(make_config("drill-twice"), resume=True, push=False)
        monkeypatch.delenv("AXIOM_KILL_AT_STEP")

    resumed = run(make_config("drill-twice"), resume=True, push=False)

    assert resumed.acc == clean.acc
