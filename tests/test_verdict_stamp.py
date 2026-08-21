"""Stamping a measured verdict into sidecars that already exist (ADR-0019).

The whole design rests on one property: writing the verdict must not change `manifest_sha256`.
If it did, every Parquet's embedded copy of that hash would stop matching its sidecar and a label
fix would become a re-pull of twelve thousand artifacts. That property is asserted here from both
directions -- the hash does not move, and an already-written sidecar still verifies itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from axiom.adjust.policy import POLICY_SPLIT_AND_DIVIDEND
from axiom.provenance.manifest import SIDECAR_SUFFIX, FileManifest
from axiom.raw.store import LocalRawStore
from axiom.raw.verdict import stamp_verdict, stamped
from axiom.registry import build_from_manifests

UNVERIFIED = "vendor_adjusted_unverified"


def manifest(symbol: str = "AAPL", source: str = "stooq", **extra) -> FileManifest:
    return FileManifest(
        schema_version=1,
        source=source,
        market="us",
        asset_class="equity",
        symbol=symbol,
        frequency="1d",
        pull_run_id="pull-1",
        pulled_at="2026-08-21T00:00:00+00:00",
        loader_version="0.2.0+abc",
        source_urls=["https://example/d_us_txt.zip"],
        source_sha256s=["a" * 64],
        artifact_path=f"raw/{source}/us/1d/{symbol[0]}/{symbol}.parquet",
        artifact_sha256="c" * 64,
        row_count=1000,
        first_ts=0,
        last_ts=86_400_000_000,
        gap_count=0,
        adjustment_policy=UNVERIFIED,
        universe_hash="u" * 12,
        **extra,
    )


def test_stamping_does_not_move_the_identity_hash() -> None:
    """The property the whole approach depends on."""
    before = manifest()
    after = stamped(before, POLICY_SPLIT_AND_DIVIDEND)
    assert after.adjustment_policy_verified == POLICY_SPLIT_AND_DIVIDEND
    assert after.manifest_sha256 == before.manifest_sha256
    # And the belief the loader recorded is still there, untouched.
    assert after.adjustment_policy == UNVERIFIED


def test_a_stamped_sidecar_still_verifies_itself() -> None:
    """`from_json` recomputes the hash and rejects an edited sidecar. A stamp is not an edit."""
    text = stamped(manifest(), POLICY_SPLIT_AND_DIVIDEND).to_json()
    reloaded = FileManifest.from_json(text)
    assert reloaded.adjustment_policy_verified == POLICY_SPLIT_AND_DIVIDEND
    assert reloaded.manifest_sha256 == manifest().manifest_sha256


def test_an_unstamped_sidecar_loads_unchanged() -> None:
    """A v0.1 or v0.2 sidecar has no such key at all and must hash exactly as it always did."""
    payload = json.loads(manifest().to_json())
    recorded = payload.pop("manifest_sha256")
    payload.pop("adjustment_policy_verified")
    reloaded = FileManifest.model_validate(payload)
    assert reloaded.adjustment_policy_verified == ""
    assert reloaded.manifest_sha256 == recorded


def test_stamping_rewrites_only_the_matching_source(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path)
    manifests = [
        manifest("AAPL"),
        manifest("MSFT"),
        manifest("BTCUSDT", source="binance_vision"),
    ]
    for m in manifests:
        store.put(m.artifact_path, b"not really parquet", m)

    run = stamp_verdict(store, manifests, source="stooq", verdict=POLICY_SPLIT_AND_DIVIDEND)
    assert (run.stamped, run.already, run.skipped, run.failures) == (2, 0, 1, [])

    for m in manifests:
        written = FileManifest.from_json(
            (tmp_path / (m.artifact_path + SIDECAR_SUFFIX)).read_text(encoding="utf-8")
        )
        expected = POLICY_SPLIT_AND_DIVIDEND if m.source == "stooq" else ""
        assert written.adjustment_policy_verified == expected
        assert written.manifest_sha256 == m.manifest_sha256, "a stamp moved the identity hash"


def test_stamping_twice_writes_nothing_the_second_time(tmp_path: Path) -> None:
    """Idempotent, so a re-run after a partial pass costs only what it did not reach."""
    store = LocalRawStore(tmp_path)
    manifests = [manifest("AAPL")]
    store.put(manifests[0].artifact_path, b"x", manifests[0])

    first = stamp_verdict(store, manifests, source="stooq", verdict=POLICY_SPLIT_AND_DIVIDEND)
    assert first.stamped == 1
    fresh = store.list_manifests()
    second = stamp_verdict(store, fresh, source="stooq", verdict=POLICY_SPLIT_AND_DIVIDEND)
    assert (second.stamped, second.already) == (0, 1)


def test_a_dry_run_writes_nothing(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path)
    manifests = [manifest("AAPL")]
    store.put(manifests[0].artifact_path, b"x", manifests[0])

    run = stamp_verdict(
        store, manifests, source="stooq", verdict=POLICY_SPLIT_AND_DIVIDEND, dry_run=True
    )
    assert run.stamped == 1
    assert store.list_manifests()[0].adjustment_policy_verified == ""


def test_the_registry_carries_the_verdict() -> None:
    """So `axiom derive tr` can read it without downloading twelve thousand sidecars."""
    table = build_from_manifests(
        [stamped(manifest("AAPL"), POLICY_SPLIT_AND_DIVIDEND), manifest("MSFT")]
    ).table
    verified = dict(
        zip(
            table["symbol"].to_pylist(),
            table["adjustment_policy_verified"].to_pylist(),
            strict=True,
        )
    )
    assert verified == {"AAPL": POLICY_SPLIT_AND_DIVIDEND, "MSFT": ""}
    # The belief at pull time survives beside it.
    assert set(table["adjustment_policy"].to_pylist()) == {UNVERIFIED}
