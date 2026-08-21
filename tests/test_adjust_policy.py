"""Total-return arithmetic, checked against numbers computed by hand.

The accumulation branch is dead under the recorded verdict (ADR-0019) and is tested anyway. It is
the branch a future re-audit would switch on, and a branch nobody has run is a branch nobody knows
is right.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from axiom.adjust.policy import (
    POLICY_NONE,
    POLICY_SPLIT_AND_DIVIDEND,
    POLICY_SPLIT_ONLY,
    POLICY_UNADJUSTED,
    TR_SCHEMA,
    UnknownAdjustmentPolicy,
    dividends_on_grid,
    needs_events,
    tr_available,
    tr_close,
)
from axiom.schema.bars import BARS_SCHEMA_V1

DAY = 86_400_000


def bars(closes: list[float], *, start_day: int = 0, step: int = DAY) -> pa.Table:
    n = len(closes)
    ts = [start_day * DAY + i * step for i in range(n)]
    return pa.table(
        {
            "ts": pa.array(ts, pa.int64()),
            "open": pa.array(closes, pa.float64()),
            "high": pa.array(closes, pa.float64()),
            "low": pa.array(closes, pa.float64()),
            "close": pa.array(closes, pa.float64()),
            "volume": pa.array([1.0] * n, pa.float64()),
            "amount": pa.array(closes, pa.float64()),
            "n_trades": pa.nulls(n, pa.int64()),
            "taker_buy_volume": pa.nulls(n, pa.float64()),
            "taker_buy_quote_volume": pa.nulls(n, pa.float64()),
        },
        schema=BARS_SCHEMA_V1,
    )


def bars_at(ts: list[int], closes: list[float]) -> pa.Table:
    table = bars(closes)
    return table.set_column(0, "ts", pa.array(ts, pa.int64()))


def events(rows: list[tuple[int, str, float]]) -> pa.Table:
    return pa.table(
        {
            "ts": pa.array([r[0] for r in rows], pa.int64()),
            "event_type": pa.array([r[1] for r in rows], pa.string()),
            "value": pa.array([r[2] for r in rows], pa.float64()),
        }
    )


# --- branch selection -------------------------------------------------------------------


def test_identity_verdicts_need_no_events() -> None:
    assert needs_events(POLICY_SPLIT_AND_DIVIDEND) is False
    assert needs_events(POLICY_NONE) is False
    assert needs_events(POLICY_SPLIT_ONLY) is True


@pytest.mark.parametrize("verdict", [POLICY_UNADJUSTED, "vendor_adjusted_unverified", "nonsense"])
def test_a_verdict_with_no_defined_branch_refuses(verdict: str) -> None:
    """An unadjusted series still has its splits in it; dividends alone cannot fix that."""
    with pytest.raises(UnknownAdjustmentPolicy):
        needs_events(verdict)


def test_identity_branch_returns_close_unchanged() -> None:
    table = bars([10.0, 11.0, 12.5])
    out = tr_close(table, None, POLICY_SPLIT_AND_DIVIDEND)
    assert out.schema == TR_SCHEMA
    assert out["tr_close"].to_pylist() == [10.0, 11.0, 12.5]
    assert out["ts"].to_pylist() == table["ts"].to_pylist()


def test_identity_branch_ignores_events_it_is_handed() -> None:
    """The vendor already applied them. Applying them again would double-count every payout."""
    table = bars([10.0, 11.0])
    with_events = tr_close(table, events([(DAY, "dividend", 1.0)]), POLICY_SPLIT_AND_DIVIDEND)
    assert with_events["tr_close"].to_pylist() == [10.0, 11.0]


# --- accumulation arithmetic ------------------------------------------------------------


def test_zero_dividends_is_the_identity() -> None:
    table = bars([10.0, 11.0, 9.0, 12.0])
    out = tr_close(table, events([]), POLICY_SPLIT_ONLY)
    assert out["tr_close"].to_pylist() == pytest.approx([10.0, 11.0, 9.0, 12.0])


def test_one_dividend_compounds_onto_the_path() -> None:
    """tr_1 = 10 * (11 + 1) / 10 = 12; tr_2 = 12 * 12 / 11 = 13.0909..."""
    table = bars([10.0, 11.0, 12.0])
    out = tr_close(table, events([(DAY, "dividend", 1.0)]), POLICY_SPLIT_ONLY)
    assert out["tr_close"].to_pylist() == pytest.approx([10.0, 12.0, 12.0 * 12.0 / 11.0])


def test_consecutive_dividends_both_land() -> None:
    """tr_1 = 10 * 11/10 = 11; tr_2 = 11 * (12+1)/11 = 13; tr_3 = 13 * (13+2)/12 = 16.25"""
    table = bars([10.0, 11.0, 12.0, 13.0])
    out = tr_close(
        table, events([(2 * DAY, "dividend", 1.0), (3 * DAY, "dividend", 2.0)]), POLICY_SPLIT_ONLY
    )
    assert out["tr_close"].to_pylist() == pytest.approx([10.0, 11.0, 13.0, 16.25])


def test_a_dividend_on_a_day_with_no_bar_is_carried_forward() -> None:
    """Dropping it would understate the return by exactly that dividend, silently."""
    table = bars_at([0, DAY, 3 * DAY], [10.0, 11.0, 12.0])
    out = tr_close(table, events([(2 * DAY, "dividend", 1.0)]), POLICY_SPLIT_ONLY)
    # The dividend lands on the 3*DAY bar: tr = 11 * (12+1)/11 = 13.
    assert out["tr_close"].to_pylist() == pytest.approx([10.0, 11.0, 13.0])


def test_a_dividend_after_the_last_bar_is_discarded() -> None:
    table = bars([10.0, 11.0])
    out = tr_close(table, events([(99 * DAY, "dividend", 5.0)]), POLICY_SPLIT_ONLY)
    assert out["tr_close"].to_pylist() == pytest.approx([10.0, 11.0])


def test_a_dividend_before_a_split_is_scaled_into_post_split_money() -> None:
    """The closes are split-adjusted; the dividend as paid is not.

    A 4:1 split after the payout means every pre-split close was divided by four, so a $1
    dividend paid then is $0.25 against those prices. Adding the raw dollar would overstate the
    day's return fourfold.
    """
    table = bars([10.0, 11.0, 12.0])
    payload = events([(DAY, "dividend", 1.0), (2 * DAY, "split", 4.0)])
    out = tr_close(table, payload, POLICY_SPLIT_ONLY)
    assert out["tr_close"].to_pylist() == pytest.approx(
        [10.0, 10.0 * 11.25 / 10.0, 10.0 * 11.25 / 10.0 * 12.0 / 11.0]
    )


def test_a_dividend_after_every_split_is_not_scaled() -> None:
    table = bars([10.0, 11.0, 12.0])
    payload = events([(DAY, "split", 4.0), (2 * DAY, "dividend", 1.0)])
    grid = dividends_on_grid(payload, table["ts"].to_numpy(zero_copy_only=False))
    assert list(grid) == pytest.approx([0.0, 0.0, 1.0])


def test_a_dividend_and_a_split_on_the_same_day() -> None:
    """The split is not "after" the dividend, so the dividend is already in post-split money."""
    table = bars([10.0, 11.0, 12.0])
    payload = events([(DAY, "dividend", 1.0), (DAY, "split", 2.0)])
    grid = dividends_on_grid(payload, table["ts"].to_numpy(zero_copy_only=False))
    assert list(grid) == pytest.approx([0.0, 1.0, 0.0])


def test_a_zero_close_refuses_rather_than_returning_inf() -> None:
    table = bars([10.0, 0.0, 12.0])
    with pytest.raises(ValueError, match="close is zero"):
        tr_close(table, events([]), POLICY_SPLIT_ONLY)


def test_empty_series() -> None:
    for verdict in (POLICY_SPLIT_ONLY, POLICY_SPLIT_AND_DIVIDEND):
        out = tr_close(bars([]), events([]), verdict)
        assert out.num_rows == 0
        assert out.schema == TR_SCHEMA


# --- availability -----------------------------------------------------------------------


def test_tr_is_always_available_under_an_identity_verdict() -> None:
    assert tr_available(None, POLICY_SPLIT_AND_DIVIDEND) is True
    assert tr_available(None, POLICY_NONE) is True


def test_missing_events_under_accumulation_means_unavailable_not_zero() -> None:
    """Unknown dividends are not no dividends, and conflating them would hit the payers hardest."""
    assert tr_available(None, POLICY_SPLIT_ONLY) is False
    assert tr_available(events([]), POLICY_SPLIT_ONLY) is True
