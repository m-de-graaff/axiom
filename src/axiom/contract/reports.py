"""The Phase E report, rendered from the dryrun's sketches.

Machine-generated. `docs/reports/v0.4-contract-qa.md` is this output with a reviewed prose section
added on top, the same split the v0.3 clean report uses: the tables are produced, the reading of
them is written by somebody who looked.
"""

from __future__ import annotations

from typing import Any

from axiom.contract.stats import REPORT_QUANTILES

#: Above this, a clip rate is a red flag the Phase E gate requires investigating in writing
#: rather than noting. Either the constants are wrong for the slice or the slice is pathological,
#: and both are things v0.5 needs to know before it picks quantizer ranges.
CLIP_RED_FLAG = 0.005


def _fmt(value: float) -> str:
    return f"{value:+.4f}"


def distribution_table(rows: list[dict[str, Any]], spec_id: str) -> str:
    header = ["class", "freq", "feature", "n", "clip %"] + [f"q{q:g}" for q in REPORT_QUANTILES]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + "|",
    ]
    for row in rows:
        if row["spec_id"] != spec_id:
            continue
        cells = [
            row["asset_class"],
            row["frequency"],
            row["feature"],
            f"{row['n']:,}",
            f"{100 * row['clip_rate']:.3f}",
        ] + [_fmt(row[f"q{q:g}"]) for q in REPORT_QUANTILES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def red_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["clip_rate"] > CLIP_RED_FLAG]


def spec_comparison(rows: list[dict[str, Any]]) -> str:
    """geo-v1 against ret-v1 on the two things v0.5 has to choose between them on.

    Clip rate says how much of the distribution the frozen scaling fails to hold, and the 1-99
    spread says how much dynamic range a quantizer has to spend. The two flow features are
    identical across specs by construction, so only the four price features are compared.
    """
    by_key = {(r["spec_id"], r["asset_class"], r["frequency"], r["feature"]): r for r in rows}
    geo_price = ("gap", "body", "upper", "lower")
    ret_price = ("ret_open", "ret_high", "ret_low", "ret_close")
    lines = [
        "| class | freq | geo clip % | ret clip % | geo q1-q99 span | ret q1-q99 span |",
        "|---|---|---:|---:|---:|---:|",
    ]
    slices = sorted({(r["asset_class"], r["frequency"]) for r in rows})
    for asset_class, frequency in slices:
        stats = {}
        for spec_id, names in (("geo-v1", geo_price), ("ret-v1", ret_price)):
            found = [by_key.get((spec_id, asset_class, frequency, n)) for n in names]
            found = [f for f in found if f]
            if not found:
                continue
            clip = sum(f["clipped"] for f in found) / max(1, sum(f["n"] for f in found))
            span = sum(f["q0.99"] - f["q0.01"] for f in found) / len(found)
            stats[spec_id] = (clip, span)
        if len(stats) != 2:
            continue
        lines.append(
            f"| {asset_class} | {frequency} | {100 * stats['geo-v1'][0]:.3f} | "
            f"{100 * stats['ret-v1'][0]:.3f} | {stats['geo-v1'][1]:.4f} | "
            f"{stats['ret-v1'][1]:.4f} |"
        )
    return "\n".join(lines)


def contract_qa_markdown(
    rows: list[dict[str, Any]],
    result,
    *,
    constants,
    specs,
    usable_windows_512: int,
    snapshot_hashes: dict[str, str],
    failures: list[str],
    partial: bool = False,
) -> str:
    flags = red_flags(rows)
    audit = f"**{result.audits_passed}/{result.audits_run}**" if result.audits_run else "not run"
    parts = [
        "# v0.4 contract QA (generated)",
        "",
        f"`schema_version` **{constants.schema_version}** · constants **{constants.config_hash}** "
        + " · ".join(f"`{s.spec_id}` **{s.config_hash}**" for s in specs),
        "",
        f"**{result.sketches.segments:,} segment-passes · {result.sketches.bars:,} bars · "
        f"{result.rows:,} feature rows · {result.bars_per_second:,.0f} bars/sec/core · "
        f"{result.n_nan} NaN/Inf.**",
        "",
        f"Corpus prefix-consistency audit: {audit} split points on real segments, bit-exact.",
        "",
        f"Usable windows at context 512, under the anchor-bar rule: **{usable_windows_512:,}**. "
        "This supersedes the v0.3 table, which counted bars rather than feature rows.",
        "",
    ]
    if partial:
        parts += [
            "> **Partial run.** `--limit` was set, so these numbers cover a prefix of the corpus "
            "and must not be read as the corpus.",
            "",
        ]
    parts += [
        "## Red flags",
        "",
    ]
    if flags:
        parts.append(
            f"{len(flags)} (spec, class, frequency, feature) combination(s) clip above "
            f"{100 * CLIP_RED_FLAG:.1f} %. Each needs a written investigation before G2 closes."
        )
        parts += [
            "",
            "| spec | class | freq | feature | clip % | q0.001 | q0.999 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for row in sorted(flags, key=lambda r: -r["clip_rate"]):
            parts.append(
                f"| {row['spec_id']} | {row['asset_class']} | {row['frequency']} | "
                f"{row['feature']} | {100 * row['clip_rate']:.3f} | {_fmt(row['q0.001'])} | "
                f"{_fmt(row['q0.999'])} |"
            )
    else:
        parts.append(
            f"None. Every (spec, class, frequency, feature) clips below "
            f"{100 * CLIP_RED_FLAG:.1f} %."
        )
    parts += ["", "## geo-v1 against ret-v1", "", spec_comparison(rows), ""]
    for spec in specs:
        parts += [
            f"## Distributions — `{spec.spec_id}`",
            "",
            "Quantiles are of the **scaled, clipped** feature, which is what the tokenizer sees.",
            "",
            distribution_table(rows, spec.spec_id),
            "",
        ]
    parts += ["## Pinned regression snapshots", "", "| series / spec | sha256 |", "|---|---|"]
    for key, digest in sorted(snapshot_hashes.items()):
        parts.append(f"| {key} | `{digest}` |")
    parts.append("")
    if failures:
        parts += [
            "## Failures",
            "",
            f"{len(failures)} artifact(s) or segment(s) failed. The first twenty:",
            "",
            *(f"- `{failure}`" for failure in failures[:20]),
            "",
        ]
    return "\n".join(parts)
