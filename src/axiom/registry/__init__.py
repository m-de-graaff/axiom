"""The corpus registry: one queryable table over every sidecar in `axiom-raw`."""

from axiom.registry.build import (
    REGISTRY_PATH,
    REGISTRY_SCHEMA,
    SUMMARY_PATH,
    BadSidecar,
    RegistryBuild,
    build_from_manifests,
    build_registry,
    read_registry,
    registry_hash,
    write_registry_parquet,
)
from axiom.registry.reports import (
    coverage_matrix,
    gappiest,
    m0_verdict,
    staleness,
    storage_by_source,
    summary_markdown,
)

__all__ = [
    "REGISTRY_PATH",
    "REGISTRY_SCHEMA",
    "SUMMARY_PATH",
    "BadSidecar",
    "RegistryBuild",
    "build_from_manifests",
    "build_registry",
    "coverage_matrix",
    "gappiest",
    "m0_verdict",
    "read_registry",
    "registry_hash",
    "staleness",
    "storage_by_source",
    "summary_markdown",
    "write_registry_parquet",
]
