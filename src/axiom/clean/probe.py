"""Why did this series fragment? A diagnostic for the drop-stats gate.

The Phase F gate asks for the most-cut series to be inspected one by one, with a verdict each:
data problem, real market pathology, or rule artifact. Telling those apart needs the *shape* of
what was cut, not the count -- an instrument that loses every bar to a thousand one-hour gaps has
a misdeclared session, and one that loses them to a single decade-long hole has a data problem,
and the drop statistics report both as "100 %".

So this reports the shape: how big the unexpected gaps are, when they happen, and whether the
zero-volume bars are inside the window where the market was supposed to be shut anyway.

Pure. It is handed a table and returns numbers.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pyarrow as pa

from axiom.clean.calendars import missing_open_slots, open_slot_mask
from axiom.clean.config import SessionRule
from axiom.schema.bars import grid_step_ms, weekday_utc

MS_PER_DAY = 86_400_000
MS_PER_HOUR = 3_600_000

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def probe_series(bars: pa.Table, *, frequency: str, session: SessionRule, top: int = 8) -> dict:
    """Characterise the gaps and the dead bars in one series.

    ``gap_sizes`` counts unexpected gaps by how many expected-open slots they swallow. A
    distribution dominated by ones and twos is a session declared wrong; a long tail is an
    instrument that genuinely stopped trading for a while.

    ``missing_by_hour`` and ``missing_by_weekday`` say *when* the holes are. A daily maintenance
    break shows up as one hour taking nearly all of them, which no market outage ever does.
    """
    ts = bars["ts"].to_numpy(zero_copy_only=False).astype(np.int64)
    volume = bars["volume"].to_numpy(zero_copy_only=False)
    step = grid_step_ms(frequency)

    missing = missing_open_slots(ts, frequency=frequency, session=session)
    gaps = missing[missing > 0]

    # Which slots are actually absent, so the hour-of-day question can be asked of them.
    index = (ts - int(ts[0])) // step
    slots = int(ts[0]) + np.arange(int(index[-1]) + 1, dtype=np.int64) * step
    present = np.zeros(slots.size, dtype=bool)
    present[np.clip(index, 0, slots.size - 1)] = True
    is_open = open_slot_mask(slots, frequency=frequency, session=session)
    absent = slots[is_open & ~present]

    dead = volume <= 0.0
    dead_ts = ts[dead]
    dead_in_shut = (
        int((~open_slot_mask(dead_ts, frequency=frequency, session=session)).sum())
        if dead_ts.size
        else 0
    )

    return {
        "bars": int(ts.size),
        "first_ts": int(ts[0]),
        "last_ts": int(ts[-1]),
        "expected_open_slots": int(is_open.sum()),
        "unexpected_gaps": int(gaps.size),
        "missing_open_slots": int(absent.size),
        "gap_sizes": _top(Counter(int(g) for g in gaps), top),
        "missing_by_hour": _top(Counter(int(h) for h in (absent % MS_PER_DAY) // MS_PER_HOUR), top),
        "missing_by_weekday": _top(Counter(_WEEKDAYS[int(d)] for d in weekday_utc(absent)), top),
        "zero_volume_bars": int(dead.sum()),
        "zero_volume_inside_shut_window": dead_in_shut,
    }


def _top(counter: Counter, n: int) -> list[tuple]:
    """The n most common entries, ties broken by key so the output is stable."""
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]


def format_probe(name: str, result: dict) -> str:
    """One readable block per series."""
    lines = [
        f"{name}",
        f"  {result['bars']:,} bars, {result['expected_open_slots']:,} expected-open slots, "
        f"{result['missing_open_slots']:,} of them empty",
        f"  {result['unexpected_gaps']:,} unexpected gap(s)",
        "  gap sizes (slots: count): " + ", ".join(f"{k}:{v:,}" for k, v in result["gap_sizes"]),
        "  missing by UTC hour:      "
        + ", ".join(f"{k:02d}h:{v:,}" for k, v in result["missing_by_hour"]),
        "  missing by weekday:       "
        + ", ".join(f"{k}:{v:,}" for k, v in result["missing_by_weekday"]),
        f"  zero-volume bars: {result['zero_volume_bars']:,} "
        f"({result['zero_volume_inside_shut_window']:,} inside the shut window)",
    ]
    return "\n".join(lines)
