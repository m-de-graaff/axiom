"""The corpus registry: one queryable table over every sidecar in `axiom-raw`.

A registry over one source is a list, which is why v0.1 did not build one. With four sources and
eighteen thousand artifacts, "what do we have, from where, pulled when" stops being answerable by
looking, and `list_repo_files` plus a few hundred small downloads is too slow to run every time
somebody wonders.

So the registry is a **cache with no authority**. Every row is derived from a sidecar, the
sidecars remain the truth, and a rebuild is idempotent: same tier, same bytes, same hash. Nothing
downstream may write to it, and nothing may read it in place of a sidecar when correctness
matters -- it exists so that *questions* are cheap, not so that facts live in two places.

A sidecar that cannot be read is **reported, never skipped**. A registry that silently omits what
it could not parse is worse than no registry, because the omission looks exactly like absence.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from axiom.config.hashing import SHORT_LEN, canonical_json
from axiom.provenance.manifest import SIDECAR_SUFFIX, FileManifest

log = logging.getLogger("axiom.registry")

REGISTRY_PATH = "registry/registry.parquet"
SUMMARY_PATH = "registry/summary.md"

MS_PER_DAY = 86_400_000

#: Manifest fields that are lists. They are the provenance detail a sidecar exists to carry, and
#: they do not belong in a table meant for group-bys -- so the registry keeps their *count* and
#: points at the sidecar for the rest.
_LIST_FIELDS = ("source_urls", "source_sha256s")

REGISTRY_SCHEMA = pa.schema(
    [
        pa.field("artifact_path", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("asset_class", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("source_symbol", pa.string(), nullable=False),
        pa.field("frequency", pa.string(), nullable=False),
        pa.field("schema_version", pa.int32(), nullable=False),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("first_ts", pa.int64(), nullable=False),
        pa.field("last_ts", pa.int64(), nullable=False),
        pa.field("history_days", pa.float64(), nullable=False),
        pa.field("gap_count", pa.int64(), nullable=False),
        pa.field("off_grid_count", pa.int64(), nullable=False),
        pa.field("closed_window_count", pa.int64(), nullable=False),
        pa.field("median_dollar_volume", pa.float64(), nullable=False),
        pa.field("price_side", pa.string(), nullable=False),
        pa.field("volume_convention", pa.string(), nullable=False),
        pa.field("amount_synthesized", pa.bool_(), nullable=False),
        pa.field("adjustment_policy", pa.string(), nullable=False),
        pa.field("adjustment_policy_verified", pa.string(), nullable=False),
        pa.field("redistribution_class", pa.string(), nullable=False),
        pa.field("universe_hash", pa.string(), nullable=False),
        pa.field("pull_run_id", pa.string(), nullable=False),
        pa.field("pulled_at", pa.string(), nullable=False),
        pa.field("loader_version", pa.string(), nullable=False),
        pa.field("manifest_sha256", pa.string(), nullable=False),
        pa.field("artifact_sha256", pa.string(), nullable=False),
        pa.field("source_file_count", pa.int32(), nullable=False),
        # Filled from the repo tree rather than the manifest, which does not record it.
        pa.field("artifact_bytes", pa.int64(), nullable=False),
    ]
)

#: The sort key. Deterministic ordering is what makes a rebuild byte-identical, and a hash over
#: an arbitrarily ordered table would change every time the Hub listed files in a new order.
SORT_KEY = ("source", "asset_class", "market", "frequency", "symbol")


@dataclass
class BadSidecar:
    """A sidecar that could not be turned into a row, and why."""

    path: str
    error: str


@dataclass
class RegistryBuild:
    """The result of one build: the table, its hash, and everything that would not parse."""

    table: pa.Table
    registry_hash: str
    bad: list[BadSidecar] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.bad


def manifest_row(manifest: FileManifest, *, artifact_bytes: int = 0) -> dict[str, Any]:
    """One registry row from one sidecar."""
    payload = manifest.model_dump(mode="json")
    row = {name: payload.get(name) for name in REGISTRY_SCHEMA.names if name in payload}
    row["source_symbol"] = manifest.source_symbol or manifest.symbol
    row["source_file_count"] = len(manifest.source_urls)
    row["manifest_sha256"] = manifest.manifest_sha256
    row["history_days"] = round((manifest.last_ts - manifest.first_ts) / MS_PER_DAY, 3)
    row["artifact_bytes"] = int(artifact_bytes)
    row["closed_window_count"] = int(manifest.closed_window_count)
    row["adjustment_policy_verified"] = manifest.adjustment_policy_verified
    row["median_dollar_volume"] = float(manifest.median_dollar_volume)
    for name in _LIST_FIELDS:
        row.pop(name, None)
    return row


def rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Sort into the canonical order and build the table."""
    ordered = sorted(rows, key=lambda r: tuple(str(r[k]) for k in SORT_KEY))
    columns = {
        field.name: pa.array([r.get(field.name) for r in ordered], field.type)
        for field in REGISTRY_SCHEMA
    }
    return pa.table(columns, schema=REGISTRY_SCHEMA)


def registry_hash(table: pa.Table) -> str:
    """Content identity of the whole registry.

    Taken over the rows rather than the Parquet bytes: two writers with different pyarrow
    versions should agree that the corpus is unchanged, and they will not agree on compression
    block boundaries.
    """
    payload = canonical_json(table.to_pylist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:SHORT_LEN]


def write_registry_parquet(table: pa.Table, *, registry_hash_value: str) -> bytes:
    """Serialize the registry, stamping its own hash into the file's metadata."""
    buffer = io.BytesIO()
    stamped = table.replace_schema_metadata(
        {
            b"axiom_registry_hash": registry_hash_value.encode("utf-8"),
            b"axiom_registry_rows": str(table.num_rows).encode("utf-8"),
        }
    )
    pq.write_table(stamped, buffer, compression="zstd")
    return buffer.getvalue()


def build_from_manifests(
    manifests: list[FileManifest],
    *,
    sizes: dict[str, int] | None = None,
    bad: list[BadSidecar] | None = None,
) -> RegistryBuild:
    """Assemble a registry from already-parsed sidecars. Pure; no network."""
    sizes = sizes or {}
    rows = [manifest_row(m, artifact_bytes=sizes.get(m.artifact_path, 0)) for m in manifests]
    table = rows_to_table(rows)
    return RegistryBuild(table=table, registry_hash=registry_hash(table), bad=list(bad or []))


def _sidecar_names(api: Any, repo_id: str) -> list[str]:
    return sorted(
        name
        for name in api.list_repo_files(repo_id, repo_type="dataset")
        if name.endswith(SIDECAR_SUFFIX)
    )


def _artifact_sizes(api: Any, repo_id: str) -> dict[str, int]:
    """Byte size per artifact, from the repo tree.

    The manifest does not record it -- `artifact_sha256` identifies the bytes without measuring
    them -- and "how much storage is this source costing" is one of the four questions the
    registry exists to answer, so it is read from the Hub rather than inferred.
    """
    try:
        entries = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    except Exception as exc:  # an older hub client, or a repo the tree API dislikes
        log.warning("could not read the repo tree (%s); sizes will be zero", exc)
        return {}
    sizes: dict[str, int] = {}
    for entry in entries:
        size = getattr(entry, "size", None)
        path = getattr(entry, "path", None)
        if size is not None and path and path.endswith(".parquet"):
            sizes[path] = int(size)
    return sizes


def build_registry(
    api: Any,
    repo_id: str,
    *,
    token: str | None = None,
    concurrency: int = 16,
) -> RegistryBuild:
    """Download every sidecar in the dataset and reduce them to one table.

    Threaded because this is eighteen thousand HTTPS round trips for a few hundred bytes each,
    where the cost is entirely latency. The Hub's own cache makes a rebuild after a pull cheap:
    only the sidecars that changed are fetched again.
    """
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    names = _sidecar_names(api, repo_id)
    log.info("%d sidecar(s) in %s", len(names), repo_id)

    manifests: list[FileManifest] = []
    bad: list[BadSidecar] = []

    def read(name: str) -> tuple[str, FileManifest | None, str]:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=name, repo_type="dataset", token=token)
            return name, FileManifest.from_json(Path(path).read_text(encoding="utf-8")), ""
        except Exception as exc:
            return name, None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="registry") as pool:
        for name, manifest, error in pool.map(read, names):
            if manifest is None:
                log.warning("unreadable sidecar %s: %s", name, error)
                bad.append(BadSidecar(name, error))
            else:
                manifests.append(manifest)

    return build_from_manifests(manifests, sizes=_artifact_sizes(api, repo_id), bad=bad)


def read_registry(data: bytes) -> pa.Table:
    """Load a registry Parquet back into a table."""
    return pq.read_table(io.BytesIO(data))


def registry_metadata(table: pa.Table) -> dict[str, str]:
    """The hash and row count a registry file carries about itself."""
    raw = table.schema.metadata or {}
    return {k.decode("utf-8"): v.decode("utf-8") for k, v in raw.items()}


def bad_sidecars_json(bad: list[BadSidecar]) -> str:
    return json.dumps([{"path": b.path, "error": b.error} for b in bad], indent=2) + "\n"
