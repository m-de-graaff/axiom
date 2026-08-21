"""The five stages of Kronos Algorithm 1, as pure functions over numpy columns.

No I/O, no Hub, no globals, no logging that carries state. A stage takes the spans it was given,
applies one rule, and returns the spans that survive plus what it did. The order they compose in
is fixed by ADR-0018 and enforced by :data:`axiom.clean.config.CANONICAL_STAGE_ORDER`.

**Spans are row indices, not timestamps.** Inside one series the bars table never moves, so an
index is the cheapest thing that identifies a bar, and the conversion to `[start_ts, end_ts]`
happens once at the end in :mod:`axiom.clean.engine`. Annotations that leave this module are in
timestamp space; annotations inside it are not.

The signature the plan named -- ``(bars, config, session) -> spans + stats`` -- gained a ``spans``
argument, because a stage that could not see what the previous stage decided could not compose
with it. That is the only change to the shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from axiom.clean.calendars import missing_open_slots
from axiom.clean.config import CleanConfig, FrequencyRule, SessionRule

#: Why a segment starts or ends. `series_start`/`series_end` mean the data simply began or ran
#: out -- a listing, a delisting, the edge of the pull -- and are not a cut at all.
CUT_REASONS = frozenset({"series_start", "series_end", "gap", "jump", "illiquid", "stagnant"})


@dataclass(frozen=True)
class Span:
    """A contiguous run of kept bars, as inclusive row indices."""

    start: int
    end: int
    reason_start: str = "series_start"
    reason_end: str = "series_end"

    @property
    def n_bars(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class StageStats:
    """What one rule did to one series."""

    rule: str
    #: Bars removed outright. Only the excision rules and the min-length filter remove bars;
    #: the partition rules cut between bars and remove none.
    bars_dropped: int = 0
    #: Maximal runs excised. Distinct from `bars_dropped`: one run of forty bars is not forty
    #: findings, and a report that conflated them would read as forty separate halts.
    runs_excised: int = 0
    #: Extra segments this stage produced. A partition that fires twice adds two.
    segments_created: int = 0
    #: Segments this stage removed entirely.
    segments_dropped: int = 0


def _split_at(spans: list[Span], boundaries: np.ndarray, reason: str) -> list[Span]:
    """Cut every span wherever ``boundaries`` names a bar that must start a new segment.

    ``boundaries`` holds row indices of *first* bars of new segments. A boundary that falls
    outside a span, or on a span's own first bar, changes nothing -- the cut is already there.
    """
    if boundaries.size == 0:
        return spans
    out: list[Span] = []
    for span in spans:
        inner = boundaries[(boundaries > span.start) & (boundaries <= span.end)]
        if inner.size == 0:
            out.append(span)
            continue
        cursor = span.start
        reason_in = span.reason_start
        for cut in inner.tolist():
            out.append(Span(cursor, cut - 1, reason_in, reason))
            cursor, reason_in = cut, reason
        out.append(Span(cursor, span.end, reason_in, span.reason_end))
    return out


def _runs_in(mask: np.ndarray, span: Span) -> list[tuple[int, int]]:
    """Maximal runs of True inside ``span``, as inclusive row indices.

    Restricted to the span rather than computed globally: a run interrupted by a segment boundary
    is two runs, and that is the whole reason partitioning happens before excision (ADR-0018).
    """
    window = mask[span.start : span.end + 1]
    if not window.any():
        return []
    padded = np.concatenate(([False], window, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(span.start + int(a), span.start + int(b)) for a, b in zip(starts, ends, strict=True)]


def _equal_value_runs(values: np.ndarray, span: Span) -> list[tuple[int, int]]:
    """Maximal groups of consecutive equal values inside ``span``, length two or more.

    Grouping by *value*, not by a boolean neighbour test. Closes ``[5, 5, 7, 7]`` are two runs of
    two, and a neighbour mask would report one run of four -- which at ``max_stagnant = 3`` is the
    difference between keeping every bar and deleting all of them.
    """
    window = values[span.start : span.end + 1]
    if window.size < 2:
        return []
    changes = np.flatnonzero(window[1:] != window[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes - 1, [window.size - 1]))
    return [
        (span.start + int(a), span.start + int(b))
        for a, b in zip(starts, ends, strict=True)
        if b > a
    ]


def _excise(
    spans: list[Span],
    runs_of,
    max_run: int,
    reason: str,
) -> tuple[list[Span], StageStats]:
    """Remove runs longer than ``max_run``, splitting whatever they were inside.

    ``runs_of`` maps a span to the maximal runs inside it, so the two excision rules differ only
    in what they call a run. A run of exactly ``max_run`` bars survives; strictly longer does not
    (ADR-0018).
    """
    out: list[Span] = []
    dropped = excised = 0
    for span in spans:
        runs = [r for r in runs_of(span) if (r[1] - r[0] + 1) > max_run]
        if not runs:
            out.append(span)
            continue
        excised += len(runs)
        cursor = span.start
        reason_in = span.reason_start
        for first, last in runs:
            if first > cursor:
                out.append(Span(cursor, first - 1, reason_in, reason))
            dropped += last - first + 1
            cursor, reason_in = last + 1, reason
        if cursor <= span.end:
            out.append(Span(cursor, span.end, reason_in, span.reason_end))
    created = len(out) - len(spans)
    return out, StageStats(
        rule=reason,
        bars_dropped=dropped,
        runs_excised=excised,
        segments_created=max(created, 0),
        segments_dropped=max(-created, 0),
    )


# --- the five stages --------------------------------------------------------------------


def stage_gap(
    spans: list[Span],
    ts: np.ndarray,
    *,
    frequency: str,
    session: SessionRule,
) -> tuple[list[Span], StageStats]:
    """Cut wherever the grid skips a step the series' session says should have been there.

    Removes no bars. A gap is an absence, and an absence cannot be deleted.
    """
    missing = missing_open_slots(ts, frequency=frequency, session=session)
    boundaries = np.flatnonzero(missing > 0) + 1
    out = _split_at(spans, boundaries, "gap")
    return out, StageStats(rule="gap", segments_created=len(out) - len(spans))


def stage_jump(
    spans: list[Span],
    open_: np.ndarray,
    close: np.ndarray,
    *,
    threshold: float,
) -> tuple[list[Span], StageStats]:
    """Cut wherever ``|open_t / close_{t-1} - 1|`` exceeds the frequency's threshold.

    Open against the *previous* close, so the rule sees only cross-bar discontinuity. A crash
    that happens and recovers inside one bar moves the low and leaves both ends alone, and is
    correctly not a cut (ADR-0018).
    """
    if close.size < 2:
        return spans, StageStats(rule="jump")
    with np.errstate(divide="ignore", invalid="ignore"):
        moves = np.abs(open_[1:] / close[:-1] - 1.0)
    # A zero or non-finite previous close is not a price, and comparing against it would either
    # divide by zero or silently compare NaN (which is False, i.e. "no cut"). The schema forbids
    # NaN, so this only fires on a genuinely zero close -- and a price that reached zero is a
    # discontinuity by any reading.
    moves = np.where(np.isfinite(moves), moves, np.inf)
    boundaries = np.flatnonzero(moves > threshold) + 1
    out = _split_at(spans, boundaries, "jump")
    return out, StageStats(rule="jump", segments_created=len(out) - len(spans))


def stage_illiquid(
    spans: list[Span], volume: np.ndarray, *, eps: float, max_run: int
) -> tuple[list[Span], StageStats]:
    """Excise runs of bars in which nothing traded."""
    mask = volume <= eps
    return _excise(spans, lambda span: _runs_in(mask, span), max_run, "illiquid")


def stage_stagnant(
    spans: list[Span], close: np.ndarray, *, max_run: int
) -> tuple[list[Span], StageStats]:
    """Excise runs of bars whose closes are all exactly equal.

    Float equality is meaningful here: closes come from a parse, not from arithmetic, so two bars
    that printed the same price produce the same double and two that did not, do not (ADR-0018).

    A bar is part of a stagnant run only if it has a neighbour it matches, so an isolated bar is
    never a run of one.
    """
    if close.size < 2:
        return spans, StageStats(rule="stagnant")
    return _excise(spans, lambda span: _equal_value_runs(close, span), max_run, "stagnant")


def stage_min_length(spans: list[Span], *, min_bars: int) -> tuple[list[Span], StageStats]:
    """Drop what is left that is too short to yield a training window."""
    kept = [s for s in spans if s.n_bars >= min_bars]
    dropped = [s for s in spans if s.n_bars < min_bars]
    return kept, StageStats(
        rule="min_length",
        bars_dropped=sum(s.n_bars for s in dropped),
        segments_dropped=len(dropped),
    )


# --- composition ------------------------------------------------------------------------

#: Stage name -> the callable that runs it, bound in :func:`run_stages`.
STAGE_NAMES = ("gap", "jump", "illiquid", "stagnant", "min_length")


def run_stages(
    columns: dict[str, np.ndarray],
    *,
    config: CleanConfig,
    rule: FrequencyRule,
    session: SessionRule,
    frequency: str,
    stage_order: tuple[str, ...] | None = None,
) -> tuple[list[Span], list[StageStats]]:
    """Compose the five stages over one series.

    ``stage_order`` exists so a test can prove the order matters. Production always passes None
    and gets the config's order, which :class:`CleanConfig` will not let differ from ADR-0018.
    """
    n = len(columns["ts"])
    if n == 0:
        return [], []
    spans = [Span(0, n - 1)]
    stats: list[StageStats] = []

    for name in stage_order or tuple(config.stage_order):
        if name == "gap":
            spans, s = stage_gap(spans, columns["ts"], frequency=frequency, session=session)
        elif name == "jump":
            spans, s = stage_jump(
                spans, columns["open"], columns["close"], threshold=rule.jump_threshold
            )
        elif name == "illiquid":
            spans, s = stage_illiquid(
                spans, columns["volume"], eps=config.illiquid_eps, max_run=rule.max_illiquid
            )
        elif name == "stagnant":
            spans, s = stage_stagnant(spans, columns["close"], max_run=rule.max_stagnant)
        elif name == "min_length":
            spans, s = stage_min_length(spans, min_bars=rule.min_bars)
        else:
            raise ValueError(f"unknown stage {name!r}; expected one of {STAGE_NAMES}")
        stats.append(s)

    # The first and last surviving segments still carry whatever reason the stage that produced
    # them wrote. That is correct except at the very edges of the series, where nothing cut.
    if spans:
        if spans[0].start == 0:
            spans[0] = replace(spans[0], reason_start="series_start")
        if spans[-1].end == n - 1:
            spans[-1] = replace(spans[-1], reason_end="series_end")
    return spans, stats
