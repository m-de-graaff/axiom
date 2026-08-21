"""Total-return closes for evaluation labels (ADR-0019).

Two price series, two jobs. The tokenizer sees whatever the vendor publishes; an evaluation label
that measures "what did holding this return" needs total return, dividends included. Conflating
them is how a backtest quietly measures the wrong thing.

Which branch applies is a **measurement**, not an assumption: `docs/reports/v0.2-adjustment-audit`
recorded `split_and_dividend_adjusted` for Stooq, so `tr_close` is an identity for every source we
carry. The dividend-accumulation branch below is not kept for symmetry -- the audit re-runs on any
future pull, and a vendor that changes convention has to flip a branch rather than provoke a
rewrite.

Pure. The exact arithmetic is unit-tested against hand-computed cases; nothing here reads a file.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

#: The verdict values `axiom.raw.adjustments` can produce, and what each implies for `tr_close`.
POLICY_SPLIT_ONLY = "split_adjusted"
POLICY_SPLIT_AND_DIVIDEND = "split_and_dividend_adjusted"
POLICY_UNADJUSTED = "unadjusted"
POLICY_NONE = "none"
POLICY_UNKNOWN = "vendor_adjusted_unverified"

#: Verdicts for which the vendor's close already *is* the total-return path.
IDENTITY_VERDICTS = frozenset({POLICY_SPLIT_AND_DIVIDEND, POLICY_NONE})

#: Verdicts that need dividend events accumulated onto the price path.
ACCUMULATE_VERDICTS = frozenset({POLICY_SPLIT_ONLY})

TR_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("tr_close", pa.float64(), nullable=False),
    ]
)


class UnknownAdjustmentPolicy(ValueError):
    """Raised for a verdict with no defined total-return meaning.

    An unadjusted or unverified series is not a total-return path and cannot be turned into one
    by arithmetic over dividends alone -- the splits are still in it. Approximating would produce
    a label that looks fine and is wrong, so this refuses instead.
    """


def needs_events(verdict: str) -> bool:
    """Does this verdict require dividend events to produce a total-return close?"""
    if verdict in IDENTITY_VERDICTS:
        return False
    if verdict in ACCUMULATE_VERDICTS:
        return True
    raise UnknownAdjustmentPolicy(
        f"no total-return branch for adjustment_policy={verdict!r}; "
        f"identity for {sorted(IDENTITY_VERDICTS)}, accumulation for {sorted(ACCUMULATE_VERDICTS)}"
    )


def dividends_on_grid(events: pa.Table | None, ts: np.ndarray) -> np.ndarray:
    """Dividend value per bar, aligned to ``ts``.

    A dividend stamped on a day the series has no bar -- a suspension, a vendor omission -- is
    **carried to the next bar that exists** rather than dropped. Dropping it would silently
    understate the return by exactly that dividend, and a silent understatement is the failure
    mode this whole module is here to avoid. A dividend after the last bar has nowhere to go and
    is discarded; there is no return left to attribute it to.
    """
    out = np.zeros(ts.size, dtype=np.float64)
    if events is None or events.num_rows == 0 or ts.size == 0:
        return out
    types = np.asarray(events["event_type"].to_pylist())
    event_ts = np.asarray(events["ts"].to_numpy(zero_copy_only=False))
    values = np.asarray(events["value"].to_numpy(zero_copy_only=False))

    dividends = types == "dividend"
    if not dividends.any():
        return out

    scaled = values[dividends] / _forward_split_factor(
        event_ts[dividends], event_ts[types == "split"], values[types == "split"]
    )
    # searchsorted with side="left": a dividend on a bar's own timestamp belongs to that bar,
    # and one on a day with no bar is carried forward to the next bar that exists.
    idx = np.searchsorted(ts, event_ts[dividends], side="left")
    inside = idx < ts.size
    np.add.at(out, idx[inside], scaled[inside])
    return out


def _forward_split_factor(
    dividend_ts: np.ndarray, split_ts: np.ndarray, split_ratios: np.ndarray
) -> np.ndarray:
    """Cumulative split ratio applied *after* each dividend date.

    A split-adjusted close series divides every pre-split price by the ratio. A dividend recorded
    as the amount actually paid at the time is in pre-split money, and adding it to a post-adjusted
    close would overstate the payout by exactly the split ratio -- 4x for AAPL's 2020 split.

    ponytail: assumes vendor dividend values are as-paid, not already back-adjusted. yfinance has
    shipped both conventions across versions. This branch is dead under the recorded verdict
    (ADR-0019), and the convention must be re-measured before it is ever used in anger -- against
    a ticker with a known dividend and a known split, comparing against the vendor's own adjusted
    close.
    """
    factor = np.ones(dividend_ts.size, dtype=np.float64)
    if split_ts.size == 0:
        return factor
    order = np.argsort(split_ts)
    later = np.cumprod(split_ratios[order][::-1])[::-1]
    # For each dividend, the product of every split ratio strictly after it.
    idx = np.searchsorted(split_ts[order], dividend_ts, side="right")
    has_later = idx < split_ts.size
    factor[has_later] = later[idx[has_later]]
    return factor


def tr_close(
    bars: pa.Table,
    events: pa.Table | None,
    verdict: str,
) -> pa.Table:
    """The total-return close series for one instrument.

    Under an identity verdict this is ``close`` under another name, and it is still returned as a
    table so that nothing downstream has to know which branch ran.

    Under ``split_adjusted`` it accumulates dividends onto the price path::

        tr_0 = close_0
        tr_t = tr_{t-1} * (close_t + div_t) / close_{t-1}

    A zero close would make that undefined. The schema forbids NaN and the universe screen
    forbids junk, but a zero print is not impossible in a delisting tail, so it is caught and
    named rather than propagated as inf.
    """
    ts = bars["ts"].to_numpy(zero_copy_only=False).astype(np.int64)
    close = bars["close"].to_numpy(zero_copy_only=False).astype(np.float64)

    if not needs_events(verdict):
        return pa.table(
            {"ts": pa.array(ts, pa.int64()), "tr_close": pa.array(close, pa.float64())},
            schema=TR_SCHEMA,
        )

    if ts.size == 0:
        return pa.table(
            {"ts": pa.array([], pa.int64()), "tr_close": pa.array([], pa.float64())},
            schema=TR_SCHEMA,
        )

    zero = np.flatnonzero(close[:-1] == 0.0)
    if zero.size:
        raise ValueError(
            f"close is zero at row {int(zero[0])} (ts={int(ts[zero[0]])}); a total-return path "
            "cannot be continued through a zero price"
        )

    div = dividends_on_grid(events, ts)
    ratios = np.empty(ts.size, dtype=np.float64)
    ratios[0] = 1.0
    ratios[1:] = (close[1:] + div[1:]) / close[:-1]
    tr = close[0] * np.cumprod(ratios)
    return pa.table(
        {"ts": pa.array(ts, pa.int64()), "tr_close": pa.array(tr, pa.float64())},
        schema=TR_SCHEMA,
    )


def tr_available(events: object | None, verdict: str) -> bool:
    """Can a truthful total-return series be built for this instrument?

    Under an identity verdict: always, and without events. Under accumulation: only if the event
    series was actually captured. A ticker whose yfinance fetch failed has *unknown* dividends,
    which is not the same as *no* dividends, and treating the two alike would approximate exactly
    the names that pay the most.

    ``events`` is only tested for presence, so a registry row standing in for the event series is
    as good an answer as the table itself -- and it is the cheaper one, because establishing that
    a file exists should not require downloading it.
    """
    if not needs_events(verdict):
        return True
    return events is not None
