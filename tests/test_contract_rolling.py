"""The strictly-past median, checked at every boundary it has.

This function is the whole causality argument, and its two failure modes are one index apart: a
window that ends at ``t`` instead of ``t-1`` leaks, and a window that ends at ``t-2`` throws away
a bar. Neither shows up as an exception, and on a long random series both produce plausible
numbers. So the tests are boundary tests, and the reference is the definition.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom.contract.rolling import reference_past_median, strictly_past_median

WINDOW = 8


def test_the_first_row_has_no_past_and_says_so() -> None:
    out = strictly_past_median(np.arange(5.0), WINDOW)

    assert np.isnan(out[0])


def test_the_second_row_is_the_median_of_exactly_one_value() -> None:
    """The expanding phase starts here, and a prior-blend would show up as anything else."""
    out = strictly_past_median(np.array([7.0, 100.0, 200.0]), WINDOW)

    assert out[1] == 7.0


def test_the_window_ends_before_its_own_index() -> None:
    """The leak, stated as a test: row t must not be able to see a[t]."""
    a = np.array([1.0, 2.0, 3.0, 1000.0, 5.0])

    out = strictly_past_median(a, WINDOW)

    assert out[3] == np.median(a[:3])
    assert out[4] == np.median(a[:4])


@pytest.mark.parametrize("n", [WINDOW - 1, WINDOW, WINDOW + 1, WINDOW + 2, 3 * WINDOW])
def test_matches_the_reference_across_the_window_boundary(n: int) -> None:
    """`t = window` is the first full window and `t = window + 1` the first that rolls."""
    a = np.sin(np.arange(n, dtype=np.float64)) * 10.0

    out = strictly_past_median(a, WINDOW)

    np.testing.assert_array_equal(out[1:], reference_past_median(a, WINDOW)[1:])


def test_the_rolling_phase_forgets_the_bar_that_fell_out_of_the_window() -> None:
    """An expanding window that never starts rolling passes every other test in this file."""
    a = np.concatenate(([1000.0], np.zeros(2 * WINDOW)))

    out = strictly_past_median(a, WINDOW)

    assert out[WINDOW + 1] == 0.0


@settings(max_examples=100, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=2, max_size=200
    ),
    window=st.integers(min_value=1, max_value=64),
)
def test_agrees_with_the_reference_on_anything(values, window) -> None:
    a = np.array(values, dtype=np.float64)

    np.testing.assert_array_equal(
        strictly_past_median(a, window)[1:], reference_past_median(a, window)[1:]
    )


@settings(max_examples=60, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=3, max_size=120
    ),
    window=st.integers(min_value=1, max_value=32),
    data=st.data(),
)
def test_a_prefix_of_the_output_is_the_output_of_the_prefix(values, window, data) -> None:
    """Prefix-consistency at the level below the contract, where it is cheapest to localize."""
    a = np.array(values, dtype=np.float64)
    split = data.draw(st.integers(min_value=2, max_value=a.size))

    prefix = strictly_past_median(a[:split], window)

    np.testing.assert_array_equal(prefix[1:], strictly_past_median(a, window)[1:split])


def test_a_zero_window_is_refused() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        strictly_past_median(np.arange(4.0), 0)
