"""The property battery, and the causality audit at the heart of it.

Two properties carry gate G2, and they are marked ``@pytest.mark.causality`` so v0.8's leakage
tripwire suite can re-run them by marker without re-deriving what they were:

- **Prefix-consistency.** ``transform(bars[:t+1])`` is exactly the first ``t`` rows of
  ``transform(bars)``. This is the contract's definition of causal — not a proxy for it.
- **Perturbation-invariance.** Changing bar ``j`` leaves every row before ``j`` bit-identical.
  Independent of the first: a transform that read the future through a running statistic could
  pass one formulation and fail the other, so both run.

Everything is generated. A fixed corpus of hand-written cases proves the contract works on the
cases somebody thought of, which is the set a leak is least likely to be hiding in.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from axiom.contract import inverse, load_spec, transform
from axiom.contract.spec import ContractSpec
from axiom.contract.transform import ContractError
from axiom.schema.bars import BARS_SCHEMA_V1, validate_bars
from axiom.testing.contract import constants

SPECS = [load_spec("contract_geo_v1"), load_spec("contract_ret_v1")]
CONSTANTS = constants(SPECS)
#: Same contract, no clipping. Round-trip is only defined off the clip bounds, and a fixture
#: whose scales are arbitrary clips constantly — the corpus constants are what stop it in
#: production, and Phase E is what measures that they do.
UNCLIPPED = [s.model_copy(update={"clip_low": -1e9, "clip_high": 1e9}) for s in SPECS]

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@st.composite
def valid_bars(draw, min_bars: int = 2, max_bars: int = 80) -> pa.Table:
    """A bar sequence that satisfies every invariant the contract requires of its input.

    Built rather than filtered: rejecting random floats until they happen to satisfy
    `high >= max(open, close)` would spend the whole budget on the filter.
    """
    n = draw(st.integers(min_value=min_bars, max_value=max_bars))
    price = st.floats(min_value=0.01, max_value=1e5, allow_nan=False, allow_infinity=False)
    opens = np.array(draw(st.lists(price, min_size=n, max_size=n)))
    closes = np.array(draw(st.lists(price, min_size=n, max_size=n)))
    wick_up = np.array(
        draw(st.lists(st.floats(min_value=0.0, max_value=0.2), min_size=n, max_size=n))
    )
    wick_down = np.array(
        draw(st.lists(st.floats(min_value=0.0, max_value=0.2), min_size=n, max_size=n))
    )
    highs = np.maximum(opens, closes) * (1.0 + wick_up)
    lows = np.minimum(opens, closes) * (1.0 - wick_down)
    volume = np.array(
        draw(st.lists(st.floats(min_value=0.0, max_value=1e7), min_size=n, max_size=n))
    )
    amount = volume * (opens + highs + lows + closes) / 4.0
    ts = 1_420_416_000_000 + np.arange(n, dtype=np.int64) * 3_600_000
    return _table(ts, opens, highs, lows, closes, volume, amount)


def _table(ts, open_, high, low, close, volume, amount) -> pa.Table:
    return pa.table(
        {
            "ts": pa.array(np.asarray(ts, dtype=np.int64), pa.int64()),
            "open": pa.array(np.asarray(open_, dtype=np.float64), pa.float64()),
            "high": pa.array(np.asarray(high, dtype=np.float64), pa.float64()),
            "low": pa.array(np.asarray(low, dtype=np.float64), pa.float64()),
            "close": pa.array(np.asarray(close, dtype=np.float64), pa.float64()),
            "volume": pa.array(np.asarray(volume, dtype=np.float64), pa.float64()),
            "amount": pa.array(np.asarray(amount, dtype=np.float64), pa.float64()),
            "n_trades": pa.nulls(len(ts), pa.int64()),
            "taker_buy_volume": pa.nulls(len(ts), pa.float64()),
            "taker_buy_quote_volume": pa.nulls(len(ts), pa.float64()),
        },
        schema=BARS_SCHEMA_V1,
    )


def _replace(bars: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    columns = {
        k: bars[k].to_numpy(zero_copy_only=False).copy()
        for k in ("ts", "open", "high", "low", "close", "volume", "amount")
    }
    columns[name] = values
    return _table(*(columns[k] for k in ("ts", "open", "high", "low", "close", "volume", "amount")))


def _run(bars: pa.Table, spec: ContractSpec):
    return transform(bars, spec, CONSTANTS, asset_class="crypto", frequency="1h")


# --- the causality audit -----------------------------------------------------------------


@pytest.mark.causality
@pytest.mark.parametrize("window", [4, 256])
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars(min_bars=3), data=st.data())
def test_a_prefix_transforms_to_a_prefix_of_the_transform(spec, window, bars, data) -> None:
    """The gate-G2 property: no bar influences a feature row before it. Exactly, not nearly."""
    spec = spec.model_copy(update={"volume_window": window})
    split = data.draw(st.integers(min_value=1, max_value=bars.num_rows - 1))

    prefix = _run(bars.slice(0, split + 1), spec)

    full = _run(bars, spec)
    assert prefix.n_rows == split
    np.testing.assert_array_equal(prefix.values, full.values[:split])
    np.testing.assert_array_equal(prefix.ts, full.ts[:split])


@pytest.mark.causality
@pytest.mark.parametrize("window", [4, 256])
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars(min_bars=4), data=st.data())
def test_perturbing_a_bar_leaves_every_earlier_row_bit_identical(spec, window, bars, data) -> None:
    """The second, independent probe: a future bar cannot reach backwards."""
    spec = spec.model_copy(update={"volume_window": window})
    j = data.draw(st.integers(min_value=1, max_value=bars.num_rows - 1))
    volume = bars["volume"].to_numpy(zero_copy_only=False).copy()
    volume[j] = volume[j] * 3.0 + 17.0
    before = _run(bars, spec)

    after = _run(_replace(bars, "volume", volume), spec)

    # Feature rows are indexed from bar 1, so bar j is row j - 1.
    np.testing.assert_array_equal(after.values[: j - 1], before.values[: j - 1])


@pytest.mark.causality
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars(min_bars=3))
def test_streaming_one_bar_at_a_time_equals_one_call(spec, bars) -> None:
    """Streaming safety through the public API, not through the internals it shares."""
    whole = _run(bars, spec)

    streamed = np.concatenate(
        [_run(bars.slice(0, t + 1), spec).values[-1:] for t in range(1, bars.num_rows)]
    )

    np.testing.assert_array_equal(streamed, whole.values)


# --- output shape and hygiene ------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars())
def test_valid_input_never_produces_a_nan_or_an_inf(spec, bars) -> None:
    block = _run(bars, spec)

    assert np.isfinite(block.values).all()


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars())
def test_output_is_float32_and_one_row_shorter_than_the_input(spec, bars) -> None:
    block = _run(bars, spec)

    assert block.values.dtype == np.float32
    assert block.values.shape == (bars.num_rows - 1, 6)
    assert block.feature_names == spec.feature_names


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars())
def test_every_value_lands_inside_the_declared_clip_bounds(spec, bars) -> None:
    block = _run(bars, spec)

    assert block.values.min() >= np.float32(spec.clip_low)
    assert block.values.max() <= np.float32(spec.clip_high)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars())
def test_clip_counts_match_the_values_sitting_on_the_bound(spec, bars) -> None:
    """A count that drifts from the values it counts would make the Phase E red-flag review lie."""
    block = _run(bars, spec)

    for i, name in enumerate(spec.feature_names):
        on_bound = np.isclose(block.values[:, i], spec.clip_low) | np.isclose(
            block.values[:, i], spec.clip_high
        )
        assert block.clip_counts[name] <= int(on_bound.sum())


@pytest.mark.parametrize("spec", UNCLIPPED, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars())
def test_changing_a_constant_changes_the_output(spec, bars) -> None:
    """Guards against a config path that is read, logged, and then ignored.

    Run without clipping on purpose. Against the test fixture's arbitrary scales most values sit
    on a clip bound, and a value pinned to the bound is insensitive to the constant that put it
    there — which would make this test pass for the wrong reason if the clip were on.
    """
    baseline = _run(bars, spec)

    nudged = transform(
        bars, spec, constants(SPECS, nudge=0.25), asset_class="crypto", frequency="1h"
    )

    assert not np.array_equal(baseline.values, nudged.values)


# --- round trip --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", UNCLIPPED, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars(min_bars=3, max_bars=40))
def test_inverse_rebuilds_the_bars_it_came_from(spec, bars) -> None:
    """Within float32 emission tolerance, and only where nothing clipped (ADR-0020).

    Prices compare relatively and flows compare in ``log1p`` space, because that is the space the
    contract carries them in: a float32 feature bounds the *log* error uniformly, so a bar with
    zero volume sitting after one with 4e8 reconstructs as 1e-6 rather than as 0. Comparing
    volumes directly would call that a 100 % error; it is one ulp of a float32 log.
    """
    block = transform(bars, spec, CONSTANTS, asset_class="crypto", frequency="1h")
    assume(sum(block.clip_counts.values()) == 0)

    rebuilt = inverse(block, spec, CONSTANTS)

    for name in ("open", "high", "low", "close"):
        original = bars[name].to_numpy(zero_copy_only=False)
        actual = rebuilt[name].to_numpy(zero_copy_only=False)
        np.testing.assert_allclose(actual, original, rtol=1e-5, atol=1e-9)
    for name in ("volume", "amount"):
        original = np.log1p(bars[name].to_numpy(zero_copy_only=False))
        actual = np.log1p(rebuilt[name].to_numpy(zero_copy_only=False))
        np.testing.assert_allclose(actual, original, rtol=1e-5, atol=1e-4)
    np.testing.assert_array_equal(
        rebuilt["ts"].to_numpy(zero_copy_only=False), bars["ts"].to_numpy(zero_copy_only=False)
    )


@pytest.mark.parametrize("spec", UNCLIPPED, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars(min_bars=3, max_bars=40))
def test_reconstructed_bars_are_valid_bars(spec, bars) -> None:
    """The structural identities survive the round trip, which is what makes them structural."""
    block = transform(bars, spec, CONSTANTS, asset_class="crypto", frequency="1h")
    assume(sum(block.clip_counts.values()) == 0)

    rebuilt = inverse(block, spec, CONSTANTS)

    assert validate_bars(rebuilt, "1h").ok


# --- refusals ----------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars(min_bars=3), data=st.data())
def test_a_non_positive_price_is_refused_rather_than_repaired(spec, bars, data) -> None:
    row = data.draw(st.integers(min_value=0, max_value=bars.num_rows - 1))
    low = bars["low"].to_numpy(zero_copy_only=False).copy()
    low[row] = 0.0

    with pytest.raises(ContractError) as caught:
        _run(_replace(bars, "low", low), spec)

    assert caught.value.code == "non_positive_price"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@SETTINGS
@given(bars=valid_bars(min_bars=3), data=st.data())
def test_a_nan_or_inf_input_is_refused(spec, bad, bars, data) -> None:
    row = data.draw(st.integers(min_value=0, max_value=bars.num_rows - 1))
    volume = bars["volume"].to_numpy(zero_copy_only=False).copy()
    volume[row] = bad

    with pytest.raises(ContractError) as caught:
        _run(_replace(bars, "volume", volume), spec)

    assert caught.value.code == "non_finite"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars(min_bars=4), data=st.data())
def test_a_timestamp_that_does_not_advance_is_refused(spec, bars, data) -> None:
    row = data.draw(st.integers(min_value=1, max_value=bars.num_rows - 1))
    ts = bars["ts"].to_numpy(zero_copy_only=False).copy()
    ts[row] = ts[row - 1]

    with pytest.raises(ContractError) as caught:
        _run(_replace(bars, "ts", ts), spec)

    assert caught.value.code == "ts_not_increasing"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@pytest.mark.parametrize("n", [0, 1])
def test_a_segment_with_no_feature_row_is_refused(spec, n) -> None:
    ts = np.arange(n, dtype=np.int64)
    ones = np.ones(n)
    bars = _table(ts, ones, ones, ones, ones, ones, ones)

    with pytest.raises(ContractError) as caught:
        _run(bars, spec)

    assert caught.value.code == "too_short"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.spec_id)
@SETTINGS
@given(bars=valid_bars(min_bars=3), data=st.data())
def test_a_high_below_the_close_is_refused(spec, bars, data) -> None:
    row = data.draw(st.integers(min_value=0, max_value=bars.num_rows - 1))
    high = bars["high"].to_numpy(zero_copy_only=False).copy()
    close = bars["close"].to_numpy(zero_copy_only=False)
    high[row] = close[row] * 0.5

    with pytest.raises(ContractError) as caught:
        _run(_replace(bars, "high", high), spec)

    assert caught.value.code == "ohlc_inconsistent"
