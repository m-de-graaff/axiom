"""Provenance manifests: canonical serialization, identity hashing, and the idempotence test."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from axiom.provenance.manifest import (
    VOLATILE_MANIFEST_FIELDS,
    FileManifest,
    PullFailure,
    PullRunManifest,
    is_current,
    sidecar_path,
    write_sidecar,
)


def make_manifest(**overrides) -> FileManifest:
    return FileManifest(
        **{
            "schema_version": 1,
            "source": "binance_vision",
            "market": "spot",
            "asset_class": "crypto",
            "symbol": "BTCUSDT",
            "frequency": "1h",
            "pull_run_id": "pull-001",
            "pulled_at": "2026-08-20T12:00:00+00:00",
            "loader_version": "0.1.0+abc1234",
            "source_urls": ["https://example/BTCUSDT-1h-2024-01.zip"],
            "source_sha256s": ["a" * 64],
            "artifact_path": "raw/binance/spot/1h/BTCUSDT.parquet",
            "artifact_sha256": "b" * 64,
            "row_count": 744,
            "first_ts": 1_704_067_200_000,
            "last_ts": 1_706_745_600_000,
            "gap_count": 0,
            "universe_hash": "0123456789ab",
            **overrides,
        }
    )


def test_serialization_is_byte_stable():
    manifest = make_manifest()
    assert manifest.to_json() == make_manifest().to_json()
    assert manifest.to_json().endswith("\n")
    payload = json.loads(manifest.to_json())
    assert list(payload) == sorted(payload)


def test_round_trip_through_json():
    manifest = make_manifest()
    assert FileManifest.from_json(manifest.to_json()) == manifest


def test_hash_changes_when_a_substantive_field_changes():
    base = make_manifest().manifest_sha256
    assert make_manifest(row_count=745).manifest_sha256 != base
    assert make_manifest(source_sha256s=["c" * 64]).manifest_sha256 != base
    assert make_manifest(universe_hash="ffffffffffff").manifest_sha256 != base
    assert make_manifest(amount_synthesized=True).manifest_sha256 != base


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pull_run_id", "pull-999"),
        ("pulled_at", "2030-01-01T00:00:00+00:00"),
        ("artifact_sha256", "f" * 64),
        ("loader_version", "9.9.9+deadbee"),
    ],
)
def test_hash_ignores_run_identity(field, value):
    # This is what makes a re-pull byte-identical: the hash goes into the Parquet metadata, so
    # anything that varies per run must stay out of it.
    assert field in VOLATILE_MANIFEST_FIELDS
    assert make_manifest(**{field: value}).manifest_sha256 == make_manifest().manifest_sha256


def test_tampered_sidecar_is_rejected():
    payload = json.loads(make_manifest().to_json())
    payload["row_count"] = 10_000
    with pytest.raises(ValueError, match="has been edited"):
        FileManifest.from_json(json.dumps(payload))


def test_sidecar_without_a_recorded_hash_still_parses():
    payload = json.loads(make_manifest().to_json())
    del payload["manifest_sha256"]
    assert FileManifest.from_json(json.dumps(payload)).symbol == "BTCUSDT"


def test_unknown_field_is_refused():
    with pytest.raises(ValidationError):
        make_manifest(exchange="binance")


def test_parallel_lists_must_pair_up():
    with pytest.raises(ValidationError, match="parallel lists"):
        make_manifest(source_sha256s=["a" * 64, "b" * 64])


# --- the idempotence primitive -----------------------------------------------------------


def test_is_current_matches_only_on_identical_checksums():
    manifest = make_manifest(source_sha256s=["a" * 64, "b" * 64], source_urls=["u1", "u2"])
    assert is_current(manifest, ["a" * 64, "b" * 64])
    assert not is_current(manifest, ["a" * 64])
    assert not is_current(manifest, ["a" * 64, "b" * 64, "c" * 64])  # a grown daily tail
    assert not is_current(manifest, ["b" * 64, "a" * 64])  # reordered enumeration
    assert not is_current(None, ["a" * 64, "b" * 64])


def test_is_current_ignores_when_the_pull_happened():
    manifest = make_manifest(pulled_at="1999-01-01T00:00:00+00:00", pull_run_id="ancient")
    assert is_current(manifest, ["a" * 64])


# --- sidecar writing ---------------------------------------------------------------------


def test_write_sidecar_lands_next_to_the_artifact(tmp_path):
    artifact = tmp_path / "BTCUSDT.parquet"
    path = write_sidecar(make_manifest(), artifact)
    assert path == sidecar_path(artifact)
    assert path.name == "BTCUSDT.parquet.manifest.json"
    assert FileManifest.from_json(path.read_text(encoding="utf-8")).symbol == "BTCUSDT"


# --- the run manifest --------------------------------------------------------------------


def make_run(**overrides) -> PullRunManifest:
    return PullRunManifest(
        **{
            "pull_run_id": "pull-001",
            "started_at": "2026-08-20T12:00:00+00:00",
            "loader_version": "0.1.0+abc1234",
            "backend_tag": "github-actions",
            "universe_hash": "0123456789ab",
            "universe_path": "universe_v1.yaml",
            "markets": ["spot", "um"],
            "frequencies": ["1h", "1d"],
            **overrides,
        }
    )


def test_a_full_run_is_not_partial():
    assert not make_run().is_partial


@pytest.mark.parametrize("narrowing", [{"limit": 40}, {"symbols_filter": ["BTCUSDT"]}])
def test_narrowed_runs_are_marked_partial(narrowing):
    assert make_run(**narrowing).is_partial


def test_run_manifest_carries_its_failures():
    run = make_run(
        failed=1,
        failures=[PullFailure(market="um", symbol="XYZUSDT", frequency="1h", error="404")],
    )
    parsed = json.loads(run.to_json())
    assert parsed["failures"][0]["symbol"] == "XYZUSDT"
    assert parsed["failed"] == 1
