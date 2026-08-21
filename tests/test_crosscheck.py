"""The Stooq-versus-Yahoo comparison.

The whole value of this module is that its numbers mean something, so the tests pin what each
number says: an adjusted-versus-unadjusted pair must show a large difference, two vendors that
agree must show a small one, and a calendar difference must not be mistaken for either.
"""

from __future__ import annotations

from datetime import UTC, datetime

from axiom.raw.crosscheck import compare_closes, crosscheck_markdown

DAY_MS = 86_400_000
START = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)


def path(n: int, *, start: float = 100.0, step: float = 0.5, skip: set[int] | None = None):
    skip = skip or set()
    return {START + i * DAY_MS: start + i * step for i in range(n) if i not in skip}


def test_two_vendors_that_agree_show_a_tiny_difference():
    ours = path(200)
    theirs = {t: v * 1.0001 for t, v in ours.items()}
    result = compare_closes("AAPL", ours, theirs)
    assert result.status == "compared"
    assert result.max_abs_rel_diff < 0.001
    assert result.correlation > 0.999


def test_an_unadjusted_versus_adjusted_pair_shows_a_large_difference():
    """A 4:1 split one vendor applied and the other did not is a 75% gap, not a rounding error."""
    ours = path(200)
    theirs = {t: v / 4.0 for t, v in ours.items()}
    result = compare_closes("AAPL", ours, theirs)
    assert result.median_abs_rel_diff > 2.5
    # Correlation stays perfect: a constant factor is not a shape change, which is exactly why
    # correlation alone would have missed this and the relative difference is the headline.
    assert result.correlation > 0.999


def test_only_shared_dates_are_compared():
    """A holiday one vendor observes is a calendar difference, not an adjustment difference."""
    ours = path(200)
    theirs = {t: v for t, v in path(200).items() if t != START + 5 * DAY_MS}
    result = compare_closes("AAPL", ours, theirs)
    assert result.overlap_days == 199
    assert result.max_abs_rel_diff == 0.0


def test_too_little_overlap_is_skipped_rather_than_reported_thinly():
    result = compare_closes("AAPL", path(10), path(10))
    assert result.status == "skipped"
    assert "need 60" in result.detail


def test_non_positive_closes_are_excluded():
    ours = {**path(200), START: -1.0}
    result = compare_closes("AAPL", ours, path(200))
    assert result.status == "compared"
    assert result.overlap_days == 199


def test_the_markdown_says_so_when_nothing_could_be_compared():
    """Absence of a comparison must not read as a comparison that found agreement."""
    text = crosscheck_markdown([], as_of="2026-08-20")
    assert "No ticker could be compared" in text
    assert "stand independently" in text


def test_the_markdown_ranks_the_worst_ticker_first():
    small = compare_closes("AAA", path(200), {t: v * 1.0001 for t, v in path(200).items()})
    large = compare_closes("ZZZ", path(200), {t: v / 4.0 for t, v in path(200).items()})
    text = crosscheck_markdown([small, large], as_of="2026-08-20")
    assert text.index("| ZZZ |") < text.index("| AAA |")


def test_yahoo_is_asked_for_our_symbol_not_the_vendors_spelling():
    """Stooq writes `aapl.us`; Yahoo 404s on it. Asking wrong looks like Yahoo being down."""
    from axiom.raw.crosscheck import crosscheck_equities
    from tests.test_registry import manifest

    asked: list[str] = []

    def fetcher(symbol, start, end):
        asked.append(symbol)
        return path(200)

    class Store:
        def get(self, artifact_path):
            import io

            import pyarrow as pa
            import pyarrow.parquet as pq

            ts = sorted(path(200))
            buf = io.BytesIO()
            pq.write_table(
                pa.table(
                    {
                        "ts": pa.array(ts, pa.int64()),
                        "close": pa.array([100.0] * len(ts), pa.float64()),
                    }
                ),
                buf,
            )
            return buf.getvalue()

    m = manifest(
        source="stooq",
        market="us",
        asset_class="equity",
        symbol="AAPL",
        frequency="1d",
        source_symbol="aapl.us",
    )
    crosscheck_equities(Store(), [m], fetcher, sample=1, now=datetime(2024, 8, 1, tzinfo=UTC))
    assert asked == ["AAPL"]
