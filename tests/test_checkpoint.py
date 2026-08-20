"""Checkpoint write/read round trip. Local filesystem only; the Hub is v0.0's manual drill."""

from __future__ import annotations

import json

import pytest
import torch

from axiom.ops.checkpoint import (
    META_FILENAME,
    STATE_FILENAME,
    TrainState,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    step_dirname,
)
from axiom.ops.seeding import capture_rng_state, seed_all


def make_state(**overrides) -> TrainState:
    seed_all(overrides.pop("seed", 1337))
    return TrainState(
        **{
            "step": 400,
            "acc": 123.456789,
            "rng": capture_rng_state(),
            "config_hash": "abc123def456",
            "run_id": "test-001",
            "schema_version": 0,
            **overrides,
        }
    )


def test_roundtrip_preserves_every_scalar_field(tmp_path):
    original = make_state()

    restored = load_checkpoint(save_checkpoint(original, tmp_path))

    assert (restored.step, restored.acc, restored.config_hash, restored.run_id) == (
        original.step,
        original.acc,
        original.config_hash,
        original.run_id,
    )


def test_roundtrip_preserves_the_rng_stream_exactly(tmp_path):
    """The field that matters. A checkpoint that loses RNG position resumes into a different run."""
    from axiom.ops.seeding import restore_rng_state

    original = make_state()
    expected = [torch.rand(()).item() for _ in range(10)]

    restored = load_checkpoint(save_checkpoint(original, tmp_path))
    restore_rng_state(restored.rng)

    assert [torch.rand(()).item() for _ in range(10)] == expected


def test_step_directory_is_zero_padded_so_lexical_order_is_numeric_order(tmp_path):
    for step in (9, 10, 100, 1000):
        save_checkpoint(make_state(step=step), tmp_path)

    found = [p.name for p in list_checkpoints(tmp_path)]

    assert found == [step_dirname(s) for s in (9, 10, 100, 1000)]


def test_latest_checkpoint_is_none_for_a_run_that_never_saved(tmp_path):
    assert latest_checkpoint(tmp_path / "never-ran") is None


def test_load_rejects_a_state_file_that_does_not_match_its_recorded_hash(tmp_path):
    step_dir = save_checkpoint(make_state(), tmp_path)
    (step_dir / STATE_FILENAME).write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_checkpoint(step_dir)


def test_meta_records_the_hash_of_the_state_file_it_sits_beside(tmp_path):
    step_dir = save_checkpoint(make_state(), tmp_path)

    meta = json.loads((step_dir / META_FILENAME).read_text(encoding="utf-8"))

    assert meta["step"] == 400
    assert len(meta["sha256"]) == 64


def test_prune_keeps_the_newest_k_and_deletes_the_rest(tmp_path):
    for step in range(100, 700, 100):
        save_checkpoint(make_state(step=step), tmp_path)

    removed = prune_checkpoints(tmp_path, keep=3)

    assert [p.name for p in removed] == [step_dirname(s) for s in (100, 200, 300)]
    assert [p.name for p in list_checkpoints(tmp_path)] == [
        step_dirname(s) for s in (400, 500, 600)
    ]


def test_prune_is_a_no_op_when_fewer_than_k_checkpoints_exist(tmp_path):
    save_checkpoint(make_state(step=100), tmp_path)

    assert prune_checkpoints(tmp_path, keep=3) == []
    assert len(list_checkpoints(tmp_path)) == 1
