"""Which missing grid steps are expected, per session.

This is the adaptation Kronos glosses over. The paper's Stage 1 treats any missing bar as a hard
boundary, which is right for a 24/7 exchange and wrong for everything else: a market that is shut
on Saturday has not had an outage on Saturday. Applying the strict rule to FX would partition
every series into weeks, and applying it to US equities would partition every series into days.

So the question a gap stage asks is not "is a bar missing" but **"is a bar missing that should
have been there"**, and the answer depends on `session_id` (ADR-0014, ADR-0018).

Nothing here reads a file or opens a socket. The one external dependency is the
`exchange_calendars` package, imported lazily so that a crypto-only run never pays for it.
"""

from __future__ import annotations

import functools

import numpy as np

from axiom.clean.config import SessionRule
from axiom.schema.bars import grid_step_ms, weekday_utc

MS_PER_DAY = 86_400_000
MS_PER_HOUR = 3_600_000

#: Bounds for the exchange calendars we materialize. Wide enough for the oldest Stooq history
#: (the bulk archive reaches the 1960s for a handful of names) and far enough forward that a pull
#: does not walk off the end mid-decade. Widening it is free; the calendar is built once.
CALENDAR_START = "1960-01-01"
CALENDAR_END = "2035-12-31"


@functools.lru_cache(maxsize=4)
def _session_days(calendar: str) -> np.ndarray:
    """Sorted day numbers (days since epoch) on which ``calendar`` holds a session.

    Cached because building XNYS over seventy years costs about a second and the corpus has
    twelve thousand equity series that all want the same answer.
    """
    import exchange_calendars as xcals

    sessions = xcals.get_calendar(calendar, start=CALENDAR_START, end=CALENDAR_END).sessions
    return (sessions.view("int64") // (MS_PER_DAY * 1_000_000)).astype(np.int64)


def calendar_bounds_days(calendar: str) -> tuple[int, int]:
    """First and last day number the calendar covers."""
    days = _session_days(calendar)
    return int(days[0]), int(days[-1])


def open_slot_mask(slots: np.ndarray, *, frequency: str, session: SessionRule) -> np.ndarray:
    """True at every grid slot where ``session`` says the market should be trading.

    ``slots`` is an int64 array of bar-open timestamps in milliseconds, on the frequency grid.
    The mask is about *expectation*, not about what the file contains: a slot that is open here
    and absent from the file is an unexplained gap, and that is the whole point of the function.
    """
    slots = np.asarray(slots, dtype=np.int64)
    if session.kind == "strict":
        return np.ones(slots.size, dtype=bool)

    if session.kind == "weekend":
        dow = weekday_utc(slots)
        if grid_step_ms(frequency) >= MS_PER_DAY:
            # A daily bar is stamped 00:00 UTC of its date, so its hour carries no information
            # about the session. The rule degrades to the weekday, which is what the vendor
            # publishes: Monday to Friday, plus an occasional Sunday stub we never require.
            return dow < 5
        hour = (slots % MS_PER_DAY) // MS_PER_HOUR
        shut = (
            (dow == 5)
            | ((dow == 4) & (hour >= session.friday_close_hour_utc))
            | ((dow == 6) & (hour < session.sunday_open_hour_utc))
        )
        return ~shut

    if session.kind == "exchange_calendar":
        if grid_step_ms(frequency) < MS_PER_DAY:
            raise NotImplementedError(
                f"{session.calendar} at {frequency}: intraday exchange sessions need open/close "
                "times and half-day handling, which no v0.3 source requires. The predicate takes "
                "a frequency so that adding them is a change here and nowhere else."
            )
        days = slots // MS_PER_DAY
        sessions = _session_days(session.calendar)
        if days.size and (days[0] < sessions[0] or days[-1] > sessions[-1]):
            raise ValueError(
                f"series spans days {int(days[0])}..{int(days[-1])} but the {session.calendar} "
                f"calendar is materialized for {int(sessions[0])}..{int(sessions[-1])}; widen "
                "CALENDAR_START/CALENDAR_END rather than guessing at the edges"
            )
        # searchsorted rather than isin: the session array is sorted and ~18k long, and isin
        # would build a hash set per series across twelve thousand series.
        idx = np.searchsorted(sessions, days)
        idx = np.clip(idx, 0, sessions.size - 1)
        return sessions[idx] == days

    raise ValueError(f"unknown session kind {session.kind!r}")


def missing_open_slots(ts: np.ndarray, *, frequency: str, session: SessionRule) -> np.ndarray:
    """Per adjacent pair of bars, how many expected-open grid slots hold no bar.

    Returns an array of length ``len(ts) - 1``. A positive entry is an unexplained gap and the
    gap stage cuts there; a zero is either consecutive bars or a gap the session accounts for.

    Off-grid bars -- Binance really does publish stretches of phase-shifted hourly bars after an
    exchange restart (ADR-0010) -- are mapped to the slot that contains them. A run of bars all
    offset by the same 28 minutes lands one per slot and produces no gap, which is the honest
    reading: those bars traded, in those hours.
    """
    ts = np.asarray(ts, dtype=np.int64)
    if ts.size < 2:
        return np.zeros(max(ts.size - 1, 0), dtype=np.int64)

    step = grid_step_ms(frequency)
    first = int(ts[0])
    # Slot index of each bar, relative to the first bar's slot. Floor division, so a phase-shifted
    # bar counts against the slot it opened in.
    index = (ts - first) // step
    n_slots = int(index[-1]) + 1
    slots = first + np.arange(n_slots, dtype=np.int64) * step

    is_open = open_slot_mask(slots, frequency=frequency, session=session)
    # Bars occupy their own slots, so a slot holding a bar is never "missing" even if the session
    # says the market was shut -- an unexpected bar is data, not a gap (ADR-0010).
    open_cum = np.concatenate(([0], np.cumsum(is_open.astype(np.int64))))

    # Expected-open slots strictly between consecutive bars: index[i-1]+1 .. index[i]-1.
    lo = index[:-1] + 1
    hi = index[1:]
    # Two bars can share a slot when one of them is phase-shifted, which makes the span empty
    # rather than negative.
    return np.maximum(open_cum[hi] - open_cum[np.minimum(lo, hi)], 0)
