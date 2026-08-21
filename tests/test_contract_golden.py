"""Golden vectors, checked twice.

The frozen numbers in ``tests/fixtures/contract/`` are what :func:`axiom.contract.transform`
produced when the contract was frozen, and the first test says it still produces them. That alone
would only prove the implementation has not changed — including if it was wrong on the day it was
frozen.

So the second test recomputes every fixture from :func:`reference_features` below, which is
written from the formulas in ADR-0020 rather than from ``axiom.contract``: plain Python, one bar
at a time, ``statistics.median`` instead of the sliding-window trick, no vectorization to be
subtly wrong in. Two implementations that agree on six pathological fixtures is the evidence the
frozen numbers rest on.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from axiom.contract import load_spec, transform
from axiom.contract.spec import ContractConstants
from axiom.schema.bars import BARS_SCHEMA_V1

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "contract"
FIXTURES = sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))
SPECS = {"geo-v1": "contract_geo_v1", "ret-v1": "contract_ret_v1"}


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def fixture_table(fixture: dict) -> pa.Table:
    bars = fixture["bars"]
    return pa.table(
        {
            **{
                name: pa.array(bars[name], type=BARS_SCHEMA_V1.field(name).type)
                for name in ("ts", "open", "high", "low", "close", "volume", "amount")
            },
            "n_trades": pa.nulls(len(bars["ts"]), pa.int64()),
            "taker_buy_volume": pa.nulls(len(bars["ts"]), pa.float64()),
            "taker_buy_quote_volume": pa.nulls(len(bars["ts"]), pa.float64()),
        },
        schema=BARS_SCHEMA_V1,
    )


def reference_features(
    bars: dict[str, list],
    parameterization: str,
    window: int,
    scaling: list[tuple[float, float]],
    clip: tuple[float, float],
) -> list[list[float]]:
    """ADR-0020 transcribed into a loop. Deliberately slow and deliberately obvious."""
    open_, high, low, close = (bars[k] for k in ("open", "high", "low", "close"))
    log_volume = [math.log1p(v) for v in bars["volume"]]
    log_amount = [math.log1p(a) for a in bars["amount"]]
    rows = []
    for t in range(1, len(bars["ts"])):
        if parameterization == "geo":
            raw = [
                math.log(open_[t] / close[t - 1]),
                math.log(close[t] / open_[t]),
                math.log(high[t] / open_[t]),
                math.log(low[t] / open_[t]),
            ]
        else:
            raw = [math.log(x[t] / close[t - 1]) for x in (open_, high, low, close)]
        for series in (log_volume, log_amount):
            raw.append(series[t] - statistics.median(series[max(0, t - window) : t]))
        rows.append(
            [
                min(max((value - center) / scale, clip[0]), clip[1])
                for value, (center, scale) in zip(raw, scaling, strict=True)
            ]
        )
    return rows


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("spec_id", sorted(SPECS))
def test_frozen_output_is_unchanged(name: str, spec_id: str) -> None:
    fixture = load_fixture(name)
    spec = load_spec(SPECS[spec_id])
    constants = ContractConstants.model_validate(fixture["constants"])
    block = transform(
        fixture_table(fixture),
        spec,
        constants,
        asset_class=fixture["asset_class"],
        frequency=fixture["frequency"],
    )
    expected = np.array(fixture["expected"][spec_id], dtype=np.float32)
    np.testing.assert_array_equal(block.values, expected)
    assert block.values.dtype == np.float32


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("spec_id", sorted(SPECS))
def test_second_implementation_agrees(name: str, spec_id: str) -> None:
    fixture = load_fixture(name)
    spec = load_spec(SPECS[spec_id])
    constants = ContractConstants.model_validate(fixture["constants"])
    scaling = [
        (s.center, s.scale)
        for s in constants.scaling_for(spec, fixture["asset_class"], fixture["frequency"])
    ]
    reference = reference_features(
        fixture["bars"],
        spec.parameterization,
        spec.volume_window,
        scaling,
        (spec.clip_low, spec.clip_high),
    )
    expected = np.array(fixture["expected"][spec_id], dtype=np.float64)
    # Cross-platform tolerance, per ADR-0020: the frozen file is float32, the reference is
    # float64 arithmetic in a different order.
    np.testing.assert_allclose(np.array(reference), expected, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("name", FIXTURES)
def test_geometry_identities_hold(name: str) -> None:
    """`upper >= max(0, body)` and `lower <= min(0, body)`, on unscaled geo features.

    They follow from `high >= max(open, close)` and `low <= min(open, close)`, which the bar
    schema already enforces — so a failure here means the parameterization lost the structure it
    exists to encode, not that the data is odd.
    """
    fixture = load_fixture(name)
    bars = fixture["bars"]
    for t in range(1, len(bars["ts"])):
        body = math.log(bars["close"][t] / bars["open"][t])
        upper = math.log(bars["high"][t] / bars["open"][t])
        lower = math.log(bars["low"][t] / bars["open"][t])
        assert upper >= max(0.0, body) - 1e-12
        assert lower <= min(0.0, body) + 1e-12


@pytest.mark.parametrize("name", FIXTURES)
def test_return_identities_hold(name: str) -> None:
    fixture = load_fixture(name)
    bars = fixture["bars"]
    for t in range(1, len(bars["ts"])):
        prev = bars["close"][t - 1]
        r = {k: math.log(bars[k][t] / prev) for k in ("open", "high", "low", "close")}
        assert r["high"] >= max(r["open"], r["close"]) - 1e-12
        assert r["low"] <= min(r["open"], r["close"]) + 1e-12
