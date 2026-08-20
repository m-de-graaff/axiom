"""RNG capture and restore.

If these fail, every resume in the project silently continues from a different random stream.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from axiom.ops.seeding import capture_rng_state, restore_rng_state, seed_all


def draw_sequence(n: int = 8) -> list[float]:
    """One sample from each generator the project touches, interleaved."""
    out: list[float] = []
    for _ in range(n):
        out.append(random.random())
        out.append(float(np.random.rand()))
        out.append(torch.rand(()).item())
    return out


def test_restore_reproduces_the_exact_sequence_that_followed_capture():
    seed_all(1337)
    draw_sequence()
    state = capture_rng_state()

    first = draw_sequence()
    restore_rng_state(state)
    second = draw_sequence()

    assert first == second


def test_restore_from_a_stale_state_rewinds_rather_than_continuing():
    seed_all(99)
    early = capture_rng_state()
    early_draws = draw_sequence()
    draw_sequence()

    restore_rng_state(early)

    assert draw_sequence() == early_draws


def test_seed_all_makes_two_processes_agree():
    seed_all(4242)
    first = draw_sequence()

    seed_all(4242)

    assert draw_sequence() == first


def test_different_seeds_produce_different_sequences():
    seed_all(1)
    first = draw_sequence()

    seed_all(2)

    assert draw_sequence() != first


def test_restore_survives_a_list_ified_state():
    """Round-tripping through a format that turns tuples into lists must still restore.

    ``torch.save`` preserves tuples, but this guards the invariant rather than the current
    serializer, because the checkpoint format is going to change at v0.7.
    """
    seed_all(7)
    state = capture_rng_state()
    expected = draw_sequence()

    listified = {**state, "python": _listify(state["python"]), "numpy": _listify(state["numpy"])}
    restore_rng_state(listified)

    assert draw_sequence() == expected


def _listify(value):
    if isinstance(value, tuple):
        return [_listify(v) for v in value]
    return value


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_capture_restore_is_an_identity_for_any_seed(seed: int):
    seed_all(seed)
    state = capture_rng_state()

    first = draw_sequence(3)
    restore_rng_state(state)

    assert draw_sequence(3) == first
