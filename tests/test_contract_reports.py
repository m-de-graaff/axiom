"""The generated half of the v0.4 QA report.

The report is a deliverable — it is what v0.5 designs quantizer ranges against and what the G2
record cites. A renderer that silently drops a red flag would make a gate pass on a number nobody
saw, so the red-flag path is tested from both sides.
"""

from __future__ import annotations

from axiom.contract import load_spec
from axiom.contract.corpus import DryrunResult, quantile_rows
from axiom.contract.reports import CLIP_RED_FLAG, contract_qa_markdown, red_flags, spec_comparison
from axiom.contract.stats import REPORT_QUANTILES
from axiom.testing.contract import constants

SPECS = [load_spec("contract_geo_v1"), load_spec("contract_ret_v1")]


def make_row(**overrides) -> dict:
    row = {
        "spec_id": "geo-v1",
        "asset_class": "crypto",
        "frequency": "1h",
        "feature": "gap",
        "n": 1_000_000,
        "clipped": 100,
        "clip_rate": 0.0001,
        "n_nan": 0,
    }
    row.update({f"q{q:g}": 0.1 * i for i, q in enumerate(REPORT_QUANTILES)})
    row.update(overrides)
    return row


def test_a_slice_over_the_threshold_is_flagged() -> None:
    rows = [make_row(clip_rate=CLIP_RED_FLAG + 1e-6), make_row(feature="body")]

    assert [row["feature"] for row in red_flags(rows)] == ["gap"]


def test_a_slice_exactly_on_the_threshold_is_not_flagged() -> None:
    """The gate reads "above 0.5 %". A boundary that flips either way makes the number arbitrary."""
    assert red_flags([make_row(clip_rate=CLIP_RED_FLAG)]) == []


def test_the_report_names_every_flagged_slice() -> None:
    rows = [
        make_row(clip_rate=0.2, feature="gap"),
        make_row(clip_rate=0.03, feature="volume", asset_class="fx"),
    ]

    markdown = contract_qa_markdown(
        rows,
        DryrunResult(),
        constants=constants(SPECS),
        specs=SPECS,
        usable_windows_512=1,
        snapshot_hashes={},
        failures=[],
    )

    assert "| geo-v1 | crypto | 1h | gap | 20.000 |" in markdown
    assert "| geo-v1 | fx | 1h | volume | 3.000 |" in markdown


def test_the_report_says_so_plainly_when_nothing_is_flagged() -> None:
    markdown = contract_qa_markdown(
        [make_row()],
        DryrunResult(),
        constants=constants(SPECS),
        specs=SPECS,
        usable_windows_512=1,
        snapshot_hashes={},
        failures=[],
    )

    assert "None. Every (spec, class, frequency, feature) clips below" in markdown


def test_a_partial_run_is_labelled_as_one() -> None:
    """A `--limit` run's numbers must never read as the corpus."""
    markdown = contract_qa_markdown(
        [make_row()],
        DryrunResult(),
        constants=constants(SPECS),
        specs=SPECS,
        usable_windows_512=1,
        snapshot_hashes={},
        failures=[],
        partial=True,
    )

    assert "**Partial run.**" in markdown


def test_the_spec_comparison_pairs_the_two_parameterizations_per_slice() -> None:
    rows = [
        make_row(spec_id="geo-v1", feature=name, clipped=10, n=1000)
        for name in ("gap", "body", "upper", "lower")
    ] + [
        make_row(spec_id="ret-v1", feature=name, clipped=20, n=1000)
        for name in ("ret_open", "ret_high", "ret_low", "ret_close")
    ]

    table = spec_comparison(rows)

    assert "| crypto | 1h | 1.000 | 2.000 |" in table


def test_a_slice_present_for_only_one_spec_is_skipped_rather_than_half_reported() -> None:
    rows = [make_row(spec_id="geo-v1", feature="gap")]

    assert spec_comparison(rows).count("\n") == 1  # header and separator only


def test_the_quantile_rows_of_an_empty_run_are_empty() -> None:
    assert quantile_rows(DryrunResult()) == []
