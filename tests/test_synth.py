"""The synthetic toolkit's own tests.

Two things are checked, and neither of them is about cleaning: every generator produces a
schema-v1-valid table, and every annotation points at a bar that exists. A generator whose
ground truth is wrong would make the whole v0.3 edge-case suite agree with a lie.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.schema.bars import grid_step_ms, validate_bars, weekday_utc
from axiom.testing import synth

MS_PER_DAY = 86_400_000


def _every_generator() -> dict[str, synth.SynthSeries]:
    """One series per generator, plus a few compositions."""
    crypto = synth.walk("1h", 400, seed=1)
    fx = synth.walk("1h", 400, seed=2, session_id="24x5")
    equity = synth.walk("1d", 400, seed=3, session_id="XNYS-regular", start_ts=946_684_800_000)
    return {
        "walk_24x7": crypto,
        "walk_24x5": fx,
        "walk_xnys": equity,
        "split": synth.with_split(crypto, 4.0, at=200),
        "adjusted_split": synth.with_adjusted_split(crypto, 4.0, at=200),
        "rollover": synth.with_rollover_jump(crypto, at=150),
        "flash_intrabar": synth.with_flash_crash(crypto, at=120, intrabar=True),
        "flash_crossbar": synth.with_flash_crash(crypto, at=120, intrabar=False),
        "gap_unexpected": synth.with_gap(crypto, at=100, n_bars=5),
        "gap_weekend": synth.with_gap(fx, at=10, kind="expected_weekend"),
        "gap_holiday": synth.with_gap(equity, at=10, kind="expected_holiday"),
        "dst": synth.with_dst_weekend(fx),
        "suspension": synth.with_suspension(equity, at=100, n_bars=20),
        "delisting": synth.ends_at(equity, int(equity.ts[300])),
        "truncated": synth.truncate_tail(crypto, 40),
        "illiquid": synth.with_illiquid(crypto, at=250, n=6),
        "stagnant": synth.with_stagnant(crypto, at=250, n=6),
        "limit_lock": synth.with_limit_lock(crypto, at=250, n=6),
        "stacked": synth.with_stagnant(
            synth.with_gap(synth.with_split(crypto, 2.0, at=80), at=200, n_bars=4), at=300, n=8
        ),
    }


@pytest.mark.parametrize("name", sorted(_every_generator()))
def test_generator_output_is_schema_valid(name: str) -> None:
    series = _every_generator()[name]
    report = validate_bars(series.table, series.frequency, session_id=series.session_id)
    assert report.ok, f"{name}: {report.summary()}"


@pytest.mark.parametrize("name", sorted(_every_generator()))
def test_annotations_land_on_real_bars(name: str) -> None:
    """Every annotated cut and excision names a timestamp the series actually carries."""
    series = _every_generator()[name]
    present = set(series.ts.tolist())
    for ts in series.cut_ts:
        assert ts in present, f"{name}: cut at {ts} is not a bar in the series"
    for first, last in series.excised_ts:
        assert first in present and last in present, f"{name}: excision {first}..{last} is off-grid"
        assert first <= last


def test_walk_has_no_jump_that_any_table_4_threshold_would_cut() -> None:
    """The base series must be clean, or every edge-case test is measuring noise.

    0.10 is the tightest jump threshold in Table 4 (the 1m row), so clearing it clears all of
    them. Checked over many seeds because a single seed passing is luck.
    """
    for seed in range(50):
        series = synth.walk("1h", 512, seed=seed)
        open_ = series.column("open")
        close = series.column("close")
        moves = np.abs(open_[1:] / close[:-1] - 1.0)
        assert moves.max() < 0.10, f"seed {seed} drew a {moves.max():.3f} move"


def test_walk_has_no_stagnant_or_illiquid_run() -> None:
    series = synth.walk("1h", 2000, seed=7)
    assert (series.column("volume") > 0).all()
    assert not (np.diff(series.column("close")) == 0.0).any()


def test_24x7_grid_is_contiguous() -> None:
    series = synth.walk("1h", 300)
    assert (np.diff(series.ts) == grid_step_ms("1h")).all()


def test_24x5_grid_skips_the_weekend_and_nothing_else() -> None:
    series = synth.walk("1h", 500, session_id="24x5")
    ts = series.ts
    dow = weekday_utc(ts)
    hour = (ts % MS_PER_DAY) // 3_600_000
    assert not (dow == 5).any(), "a Saturday bar in a 24x5 series"
    assert not ((dow == 6) & (hour < 22)).any(), "a Sunday-morning bar in a 24x5 series"
    # Every gap is a weekend: a jump of more than one step only ever happens over Friday night.
    holes = np.flatnonzero(np.diff(ts) > grid_step_ms("1h"))
    assert holes.size > 0
    assert (weekday_utc(ts[holes]) == 4).all()


def test_xnys_grid_holds_only_sessions() -> None:
    import exchange_calendars as xcals

    series = synth.walk("1d", 300, session_id="XNYS-regular", start_ts=946_684_800_000)
    calendar = xcals.get_calendar("XNYS", start="1990-01-01", end="2035-12-31")
    sessions = set((calendar.sessions.view("int64") // (MS_PER_DAY * 1_000_000)).tolist())
    assert set((series.ts // MS_PER_DAY).tolist()) <= sessions
    # And it really does skip Independence Day rather than merely skipping weekends.
    assert (np.diff(series.ts) > 3 * MS_PER_DAY).any()


def test_unadjusted_split_moves_prices_and_adjusted_one_does_not() -> None:
    base = synth.walk("1d", 300, seed=5)
    unadjusted = synth.with_split(base, 4.0, at=150)
    adjusted = synth.with_adjusted_split(base, 4.0, at=150)

    assert unadjusted.column("open")[150] == pytest.approx(base.column("open")[150] / 4.0)
    assert unadjusted.cut_ts == (int(base.ts[150]),)

    assert adjusted.column("open")[150] == pytest.approx(base.column("open")[150])
    assert adjusted.cut_ts == ()
    # The volume step is real even when the price is continuous.
    assert adjusted.column("volume")[149] == pytest.approx(base.column("volume")[149] * 4.0)


def test_intrabar_crash_leaves_open_and_close_alone() -> None:
    base = synth.walk("1h", 300, seed=9)
    crashed = synth.with_flash_crash(base, at=120, intrabar=True)
    assert crashed.cut_ts == ()
    assert crashed.column("open")[120] == pytest.approx(base.column("open")[120])
    assert crashed.column("close")[120] == pytest.approx(base.column("close")[120])
    assert crashed.column("low")[120] < base.column("low")[120]


def test_limit_lock_freezes_close_and_kills_volume() -> None:
    locked = synth.with_limit_lock(synth.walk("1h", 300, seed=11), at=100, n=5)
    assert (locked.column("volume")[100:105] == 0.0).all()
    assert len(set(locked.column("close")[100:105].tolist())) == 1
    assert locked.excised_ts == ((int(locked.ts[100]), int(locked.ts[104])),)


def test_gap_generator_refuses_a_session_with_no_natural_gap() -> None:
    with pytest.raises(ValueError, match="no natural gap"):
        synth.with_gap(synth.walk("1h", 200), at=10, kind="expected_weekend")


def test_stacking_accumulates_annotations() -> None:
    base = synth.walk("1h", 600, seed=13)
    stacked = synth.with_stagnant(synth.with_split(base, 2.0, at=100), at=400, n=5)
    assert len(stacked.cut_ts) == 1
    assert len(stacked.excised_ts) == 1
    assert len(stacked.notes) >= 3
