"""The v0.3 cleaning engine, against the roadmap's named edge cases.

Every series here comes from :mod:`axiom.testing.synth`, which knows where the cuts belong and
shares no code with the thing under test. A test that built its own weekend calendar would agree
with the engine about weekends by construction and prove nothing.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom.clean.config import CANONICAL_STAGE_ORDER, CleanConfig, load_clean_config
from axiom.clean.engine import (
    CleanResult,
    SeriesIdentity,
    clean_series,
    dropstats_table,
    segments_table,
    usable_windows,
    verify_corpus_invariants,
)
from axiom.clean.stages import Span, run_stages
from axiom.config.hashing import canonical_json
from axiom.testing import synth


def config(**per_frequency: int | dict[str, object]) -> CleanConfig:
    """The real config, with per-frequency thresholds optionally overridden.

    A bare int means ``min_bars``. Table 4's ``min_bars`` is 256 at 1h and 128 at 1d, which would
    drop every segment in a 300-bar synthetic series and turn each structural test into a test of
    the min-length filter. The thresholds a structural test is *about* -- the jump threshold, the
    run limits -- are never relaxed, and the min-length filter has its own test at the real
    number.
    """
    cfg = load_clean_config("clean_v1")
    if not per_frequency:
        return cfg
    payload = cfg.model_dump()
    for frequency, override in per_frequency.items():
        fields = {"min_bars": override} if isinstance(override, int) else override
        payload["frequencies"][frequency].update(fields)
    return CleanConfig.model_validate(payload)


def identity(series: synth.SynthSeries, symbol: str = "TEST") -> SeriesIdentity:
    return SeriesIdentity(
        source="synth",
        market="test",
        asset_class="test",
        symbol=symbol,
        frequency=series.frequency,
        session_id=series.session_id,
        artifact_path=f"raw/synth/test/{series.frequency}/{symbol}.parquet",
        raw_artifact_sha256="0" * 64,
    )


def clean(series: synth.SynthSeries, cfg: CleanConfig | None = None) -> CleanResult:
    return clean_series(series.table, identity(series), cfg or config(**{"1h": 4, "1d": 4}))


def starts(result: CleanResult) -> list[int]:
    return [row["start_ts"] for row in result.segments]


def kept_ts(result: CleanResult, series: synth.SynthSeries) -> set[int]:
    ts = series.ts
    out: set[int] = set()
    for row in result.segments:
        out |= set(ts[(ts >= row["start_ts"]) & (ts <= row["end_ts"])].tolist())
    return out


# --- corporate actions ------------------------------------------------------------------


def test_unadjusted_split_is_cut_at_the_annotated_bar() -> None:
    series = synth.with_split(synth.walk("1d", 300, seed=1), 4.0, at=150)
    result = clean(series)
    assert len(result.segments) == 2
    assert starts(result)[1] == series.cut_ts[0]
    assert result.segments[0]["cut_reason_end"] == "jump"
    assert result.segments[1]["cut_reason_start"] == "jump"
    # A jump partitions; it does not delete. Every bar is still in some segment.
    assert result.kept_bars == result.total_bars


def test_adjusted_split_is_not_cut() -> None:
    series = synth.with_adjusted_split(synth.walk("1d", 300, seed=1), 4.0, at=150)
    result = clean(series)
    assert len(result.segments) == 1
    assert result.segments[0]["cut_reason_start"] == "series_start"
    assert result.segments[0]["cut_reason_end"] == "series_end"


def test_rollover_jump_is_cut() -> None:
    series = synth.with_rollover_jump(synth.walk("1d", 300, seed=2), at=100)
    result = clean(series)
    assert starts(result) == [int(series.ts[0]), series.cut_ts[0]]


# --- flash crashes ----------------------------------------------------------------------


def test_cross_bar_flash_crash_is_cut() -> None:
    series = synth.with_flash_crash(synth.walk("1h", 300, seed=3), at=120, intrabar=False)
    result = clean(series)
    assert starts(result) == [int(series.ts[0]), series.cut_ts[0]]


def test_intrabar_flash_crash_is_not_cut() -> None:
    """Documented Kronos-consistent behaviour: the rule reads open against the previous close."""
    series = synth.with_flash_crash(synth.walk("1h", 300, seed=3), at=120, intrabar=True)
    result = clean(series)
    assert len(result.segments) == 1


# --- gaps -------------------------------------------------------------------------------


def test_expected_weekend_gap_is_not_a_boundary() -> None:
    series = synth.walk("1h", 600, seed=4, session_id="24x5")
    # The series really does skip weekends -- otherwise this asserts nothing.
    assert (np.diff(series.ts) > 3_600_000).any()
    result = clean(series)
    assert len(result.segments) == 1


def test_expected_holiday_gap_is_not_a_boundary() -> None:
    series = synth.walk("1d", 400, seed=5, session_id="XNYS-regular", start_ts=946_684_800_000)
    # There has to be a real holiday in the window, or this test is about weekends. A mid-week
    # holiday shows up as a gap that skips at least one weekday.
    assert _skips_a_weekday(series.ts), "no exchange holiday in this window"
    result = clean(series)
    assert len(result.segments) == 1


def _skips_a_weekday(ts: np.ndarray) -> bool:
    """True if some gap in a daily series omits a day that is neither Saturday nor Sunday."""
    from axiom.schema.bars import weekday_utc

    day = ts // 86_400_000
    present = set(day.tolist())
    missing = set(range(int(day[0]), int(day[-1]) + 1)) - present
    return any(weekday_utc(np.array([d * 86_400_000]))[0] < 5 for d in missing)


def test_dst_shifted_weekend_is_not_a_boundary() -> None:
    series = synth.with_dst_weekend(synth.walk("1h", 800, seed=6, session_id="24x5"))
    result = clean(series)
    assert len(result.segments) == 1, "a DST shift partitioned an FX series"


def test_missing_crypto_hour_is_a_boundary() -> None:
    series = synth.with_gap(synth.walk("1h", 300, seed=7), at=100, n_bars=1)
    result = clean(series)
    assert starts(result) == [int(series.ts[0]), series.cut_ts[0]]
    assert result.segments[0]["cut_reason_end"] == "gap"


def test_equity_suspension_gives_boundaries_on_both_sides() -> None:
    base = synth.walk("1d", 400, seed=8, session_id="XNYS-regular", start_ts=946_684_800_000)
    series = synth.with_suspension(base, at=200, n_bars=20)
    result = clean(series)
    assert len(result.segments) == 2
    assert result.segments[0]["cut_reason_end"] == "gap"
    assert result.segments[1]["cut_reason_start"] == "gap"


def test_delisting_ends_the_last_segment_cleanly() -> None:
    base = synth.walk("1d", 400, seed=9, session_id="XNYS-regular", start_ts=946_684_800_000)
    series = synth.ends_at(base, int(base.ts[250]))
    result = clean(series)
    assert len(result.segments) == 1
    assert result.segments[0]["cut_reason_end"] == "series_end"
    assert result.segments[0]["end_ts"] == int(base.ts[250])


# --- run rules --------------------------------------------------------------------------


@pytest.mark.parametrize("run_length,excised", [(1, False), (2, True)])
def test_illiquid_run_boundary_is_exactly_table_4(run_length: int, excised: bool) -> None:
    """1h carries ``max_illiquid = 1``: a run of one survives, a run of two does not."""
    series = synth.with_illiquid(synth.walk("1h", 300, seed=10), at=100, n=run_length)
    result = clean(series)
    dead = set(series.ts[100 : 100 + run_length].tolist())
    kept = kept_ts(result, series)
    assert bool(dead - kept) is excised
    assert (result.total_bars - result.kept_bars) == (run_length if excised else 0)


@pytest.mark.parametrize("frozen_bars,excised", [(2, False), (3, True)])
def test_stagnant_run_boundary_is_exactly_table_4(frozen_bars: int, excised: bool) -> None:
    """1h carries ``max_stagnant = 3``: three equal closes survive, four do not.

    The generator freezes the close at the *previous* bar's value, so freezing two bars makes a
    run of three. The run length is measured rather than assumed.
    """
    base = synth.walk("1h", 300, seed=11)
    series = synth.with_stagnant(base, at=100, n=frozen_bars)
    close = series.column("close")
    run = int((close == close[100]).sum())
    assert run == frozen_bars + 1

    result = clean(series, config(**{"1h": 4}))
    dropped = result.total_bars - result.kept_bars
    assert (dropped > 0) is excised, f"run of {run} bars dropped {dropped}"
    if excised:
        assert dropped == run


def test_limit_lock_is_excised_by_both_rules_to_the_same_span() -> None:
    """A halt freezes the close and kills volume. Either rule alone would excise the same bars."""
    base = synth.walk("1h", 300, seed=12)
    series = synth.with_limit_lock(base, at=100, n=8)
    result = clean(series)
    first, last = series.excised_ts[0]
    kept = kept_ts(result, series)
    assert not (set(series.ts[(series.ts >= first) & (series.ts <= last)].tolist()) & kept)
    assert len(result.segments) == 2


def test_short_tail_is_dropped_by_min_length() -> None:
    """A jump near the end leaves a stub that cannot yield a context-512 window."""
    base = synth.walk("1h", 300, seed=13)
    series = synth.with_split(base, 4.0, at=290)
    result = clean(series, config(**{"1h": 100}))
    assert len(result.segments) == 1
    assert result.segments[0]["n_bars"] == 290
    dropstats = {row["rule"]: row for row in result.dropstats}
    assert dropstats["min_length"]["segments_dropped"] == 1
    assert dropstats["min_length"]["bars_dropped"] == 10


def test_a_series_can_clean_to_nothing() -> None:
    series = synth.truncate_tail(synth.walk("1h", 300, seed=14), 40)
    result = clean(series, config())  # real min_bars = 256 at 1h
    assert result.segments == []
    assert result.kept_bars == 0


# --- order of operations ----------------------------------------------------------------


def test_stage_order_changes_the_answer() -> None:
    """Lock ADR-0018's order with a series the wrong order cleans differently.

    Eleven zero-volume bars, with the sixth of them missing from the feed. Partitioning first
    makes two dead runs of five, and at ``max_illiquid = 5`` a run of five survives -- so every
    dead bar is kept. Excising first sees one run of ten and deletes all of it.
    """
    base = synth.walk("1h", 300, seed=15)
    series = synth.with_gap(synth.with_illiquid(base, at=100, n=11), at=105, n_bars=1)
    cfg = config(**{"1h": {"min_bars": 1, "max_illiquid": 5}})

    columns = {
        name: series.table[name].to_numpy(zero_copy_only=False)
        for name in ("ts", "open", "high", "low", "close", "volume")
    }
    kwargs = {
        "config": cfg,
        "rule": cfg.rule_for("1h"),
        "session": cfg.session_for("24x7"),
        "frequency": "1h",
    }
    canonical, _ = run_stages(columns, stage_order=CANONICAL_STAGE_ORDER, **kwargs)
    swapped, _ = run_stages(
        columns, stage_order=("illiquid", "gap", "jump", "stagnant", "min_length"), **kwargs
    )
    assert _spans(canonical) != _spans(swapped), (
        "the two stage orders produced identical segments, so this series does not lock the order"
    )
    assert sum(s.n_bars for s in canonical) > sum(s.n_bars for s in swapped)


def _spans(spans: list[Span]) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in spans]


def test_config_refuses_a_reordered_stage_list() -> None:
    payload = load_clean_config("clean_v1").model_dump()
    payload["stage_order"] = ["jump", "gap", "illiquid", "stagnant", "min_length"]
    with pytest.raises(ValueError, match="ADR-0018"):
        CleanConfig.model_validate(payload)


# --- config identity --------------------------------------------------------------------


def test_config_hash_is_sensitive_to_every_threshold() -> None:
    base = load_clean_config("clean_v1")
    for field_name in ("min_bars", "jump_threshold", "max_illiquid", "max_stagnant"):
        payload = base.model_dump()
        current = payload["frequencies"]["1h"][field_name]
        payload["frequencies"]["1h"][field_name] = current + 1
        assert CleanConfig.model_validate(payload).config_hash != base.config_hash, field_name

    payload = base.model_dump()
    payload["illiquid_eps"] = 1e-9
    assert CleanConfig.model_validate(payload).config_hash != base.config_hash


def test_the_verified_flag_is_not_in_the_config_hash() -> None:
    """It records confidence in a row, not what the row says.

    Hashing it would mean that re-reading the 2H threshold against the paper invalidates every
    1h and 1d segment in the corpus — a full recompute for a change that cannot have altered a
    single cut.
    """
    base = load_clean_config("clean_v1")
    payload = base.model_dump()
    payload["frequencies"]["2h"]["verified"] = True
    assert CleanConfig.model_validate(payload).config_hash == base.config_hash


def test_unverified_table_4_rows_refuse_to_run() -> None:
    cfg = load_clean_config("clean_v1")
    for frequency in ("10m", "20m", "40m", "2h"):
        with pytest.raises(ValueError, match="verified: false"):
            cfg.rule_for(frequency)
    for frequency in ("1h", "1d"):
        cfg.rule_for(frequency)


# --- properties -------------------------------------------------------------------------

_PATHOLOGIES = st.sampled_from(
    ["split", "gap", "illiquid", "stagnant", "limit_lock", "flash", "rollover", "none"]
)


def _compose(seed: int, kinds: list[str], n: int) -> synth.SynthSeries:
    series = synth.walk("1h", n, seed=seed)
    at = 20
    for kind in kinds:
        at += 25
        if at + 15 >= len(series.ts):
            break
        if kind == "split":
            series = synth.with_split(series, 2.0, at=at)
        elif kind == "gap":
            series = synth.with_gap(series, at=at, n_bars=2)
            n -= 2
        elif kind == "illiquid":
            series = synth.with_illiquid(series, at=at, n=4)
        elif kind == "stagnant":
            series = synth.with_stagnant(series, at=at, n=5)
        elif kind == "limit_lock":
            series = synth.with_limit_lock(series, at=at, n=6)
        elif kind == "flash":
            series = synth.with_flash_crash(series, at=at, intrabar=bool(at % 2))
        elif kind == "rollover":
            series = synth.with_rollover_jump(series, at=at)
    return series


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    kinds=st.lists(_PATHOLOGIES, min_size=0, max_size=5),
    n=st.integers(120, 400),
)
def test_invariants_hold_over_random_compositions(seed: int, kinds: list[str], n: int) -> None:
    series = _compose(seed, kinds, n)
    result = clean(series, config(**{"1h": 4}))
    ts = series.ts

    # Ordered, non-overlapping, inside the series.
    last = -1
    total_in_segments = 0
    for row in result.segments:
        assert row["start_ts"] > last
        assert row["start_ts"] <= row["end_ts"]
        assert row["start_ts"] in set(ts.tolist()) and row["end_ts"] in set(ts.tolist())
        n_in = int(((ts >= row["start_ts"]) & (ts <= row["end_ts"])).sum())
        assert n_in == row["n_bars"], "a segment's bar count disagrees with its timestamp span"
        total_in_segments += n_in
        last = row["end_ts"]

    # Every kept bar is in exactly one segment, and the books balance.
    assert total_in_segments == result.kept_bars
    assert result.kept_bars + result.dropped_bars == result.total_bars
    assert result.total_bars == len(ts)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), kinds=st.lists(_PATHOLOGIES, max_size=4))
def test_cleaning_is_deterministic(seed: int, kinds: list[str]) -> None:
    series = _compose(seed, kinds, 300)
    cfg = config(**{"1h": 4})
    first = segments_table(clean(series, cfg).segments)
    second = segments_table(clean(series, cfg).segments)
    assert canonical_json(first.to_pylist()) == canonical_json(second.to_pylist())


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), kinds=st.lists(_PATHOLOGIES, max_size=4))
def test_cleaning_is_idempotent(seed: int, kinds: list[str]) -> None:
    """Re-cleaning the bars of a surviving segment yields that segment back, unchanged.

    Cut reasons are allowed to differ -- a segment sliced out of its series legitimately begins
    with `series_start` -- but its extent may not.
    """
    series = _compose(seed, kinds, 300)
    cfg = config(**{"1h": 4})
    result = clean(series, cfg)
    ts = series.ts
    for row in result.segments:
        window = series.table.filter(pa.array((ts >= row["start_ts"]) & (ts <= row["end_ts"])))
        again = clean_series(window, identity(series), cfg)
        assert len(again.segments) == 1, f"segment {row['segment_id']} re-split on a second pass"
        assert again.segments[0]["start_ts"] == row["start_ts"]
        assert again.segments[0]["end_ts"] == row["end_ts"]
        assert again.segments[0]["n_bars"] == row["n_bars"]


# --- table assembly ---------------------------------------------------------------------


def test_tables_build_and_pass_the_corpus_invariants() -> None:
    results = [
        clean(synth.with_split(synth.walk("1h", 300, seed=s), 2.0, at=100), config(**{"1h": 4}))
        for s in range(3)
    ]
    # Distinct symbols, or three copies of the same series really would collide.
    rows: list[dict] = []
    drops: list[dict] = []
    for i, result in enumerate(results):
        for row in result.segments:
            rows.append(
                {
                    **row,
                    "symbol": f"S{i}",
                    "segment_id": f"S{i}:1h:{row['start_ts']}",
                    "artifact_path": f"raw/synth/S{i}.parquet",
                }
            )
        drops.extend(result.dropstats)
    segments = segments_table(rows)
    assert segments.num_rows == 6
    assert verify_corpus_invariants(segments) == []
    assert dropstats_table(drops).num_rows == 15


def test_segment_ids_separate_the_same_ticker_on_two_markets() -> None:
    """Binance lists ACEUSDT on spot and on USDT-M futures, and both begin the same day.

    Thirty-two ids collided on the first full corpus run, because the id held only symbol,
    frequency and start. Source and market are what tell the two listings apart.
    """
    series = synth.walk("1d", 300, seed=21)
    cfg = config(**{"1d": 4})
    ids = {
        row["segment_id"]
        for market in ("spot", "um")
        for row in clean_series(
            series.table, replace(identity(series, "ACEUSDT"), market=market), cfg
        ).segments
    }
    assert len(ids) == 2, ids


def test_corpus_invariants_catch_an_overlap() -> None:
    result = clean(synth.walk("1h", 300, seed=1), config(**{"1h": 4}))
    row = result.segments[0]
    twisted = [row, {**row, "segment_id": "TEST:1h:other", "start_ts": row["start_ts"] + 3_600_000}]
    problems = verify_corpus_invariants(segments_table(twisted))
    assert any("overlap" in p for p in problems)


def test_corpus_invariants_catch_a_mixed_config_hash() -> None:
    result = clean(synth.walk("1h", 300, seed=1), config(**{"1h": 4}))
    row = result.segments[0]
    mixed = [row, {**row, "segment_id": "OTHER:1h:1", "clean_config_hash": "deadbeefcafe"}]
    problems = verify_corpus_invariants(segments_table(mixed))
    assert any("config hashes" in p for p in problems)


def test_usable_windows_counts_what_a_context_512_model_can_train_on() -> None:
    assert usable_windows([511], 512) == 0
    assert usable_windows([512], 512) == 1
    assert usable_windows([600, 1000, 10], 512) == 89 + 489


# --- purity -----------------------------------------------------------------------------

#: What the cleaning engine and the synthetic toolkit are allowed to import at module scope.
_ALLOWED_ROOTS = {
    "__future__",
    "ast",
    "collections",
    "hashlib",
    "dataclasses",
    "functools",
    "itertools",
    "math",
    "numpy",
    "pathlib",
    "pyarrow",
    "pydantic",
    "typing",
    "yaml",
    "axiom",
}

#: Modules that would make the engine impure, named so a failure says what went wrong.
_FORBIDDEN = {"huggingface_hub", "httpx", "requests", "urllib", "socket", "subprocess"}


def _toplevel_imports(path: Path) -> set[str]:
    """Module roots imported at module scope. A lazy import inside a function does not count."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


#: The driver is not part of the pure engine: it reads bytes somebody hands it and writes files.
#: Purity is a claim about the rules, and `run.py` contains none of them.
_IMPURE_BY_DESIGN = {"__init__.py", "run.py"}


def _module_paths(package: str) -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "src" / "axiom" / package
    return sorted(p for p in root.glob("*.py") if p.name not in _IMPURE_BY_DESIGN)


@pytest.mark.parametrize("path", _module_paths("clean"), ids=lambda p: p.name)
def test_cleaning_engine_imports_nothing_that_could_do_io(path: Path) -> None:
    imported = _toplevel_imports(path)
    assert not (imported & _FORBIDDEN), f"{path.name} imports {sorted(imported & _FORBIDDEN)}"
    assert imported <= _ALLOWED_ROOTS, f"{path.name} imports {sorted(imported - _ALLOWED_ROOTS)}"


def test_engine_does_not_reach_into_the_raw_tier() -> None:
    """The rules never fetch their own input. `run.py` is handed bytes and is exempt above.

    An import, not a mention: a docstring is allowed to say what a module deliberately is not
    next to, and an earlier version of this test failed on exactly that sentence.
    """
    for path in _module_paths("clean"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            offenders = [n for n in names if n.startswith(("axiom.raw", "axiom.registry"))]
            assert not offenders, f"{path.name} imports {offenders}"


def test_synthetic_toolkit_shares_no_code_with_the_engine() -> None:
    """Independence, asserted rather than intended.

    The toolkit builds its own session grids. If it ever imported the engine's calendars, every
    session test in this file would be checking that a function agrees with itself.
    """
    path = Path(__file__).resolve().parents[1] / "src/axiom/testing/synth.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        name
        for node in ast.walk(tree)
        for name in (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert not any(name.startswith("axiom.clean") for name in imported), sorted(imported)
