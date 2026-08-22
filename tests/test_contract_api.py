"""The public surface, the refusals, and proof that the audit can actually fail.

Three separate jobs that all guard the same rule: there is one contract, it is these four
functions, and nothing routes around them.
"""

from __future__ import annotations

import ast
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest

import axiom.contract as contract
from axiom.contract import SCHEMA_VERSION, inverse, load_constants, load_spec, transform
from axiom.contract.rolling import strictly_past_median
from axiom.contract.spec import ContractConstants, ContractSpec, firewall_sha256, load_firewall
from axiom.contract.transform import ContractError
from axiom.testing.contract import constants
from tests.test_contract_properties import _table

PUBLIC = {"SCHEMA_VERSION", "inverse", "load_constants", "load_spec", "transform"}


def _bars(n: int = 40):
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = rng.lognormal(5.0, 1.0, n)
    ts = 1_420_416_000_000 + np.arange(n, dtype=np.int64) * 3_600_000
    return _table(ts, open_, high, low, close, volume, volume * close)


def test_the_package_exports_exactly_the_four_contract_functions() -> None:
    """The single-implementation rule, as a test rather than as a paragraph in an ADR."""
    assert set(contract.__all__) == PUBLIC
    assert sum(callable(getattr(contract, name)) for name in contract.__all__) == 4


def test_nothing_public_leaks_out_of_the_package_namespace() -> None:
    exported = {
        name
        for name in dir(contract)
        if not name.startswith("_") and not isinstance(getattr(contract, name), types.ModuleType)
    }

    assert exported == PUBLIC


def test_the_schema_version_is_frozen_at_one() -> None:
    """A bump here is a new tokenizer, new shards and new snapshots (ADR-0020)."""
    assert SCHEMA_VERSION == 1


@pytest.mark.parametrize("name", ["contract_geo_v1", "contract_ret_v1"])
def test_a_packaged_spec_loads_by_bare_name(name: str) -> None:
    """Cloud kernels install a wheel and have no checkout."""
    spec = load_spec(name)

    assert spec.schema_version == SCHEMA_VERSION
    assert len(spec.feature_names) == 6
    assert not spec.leaky


def test_a_spec_from_another_schema_version_is_refused() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        ContractSpec(
            spec_id="future-v9",
            schema_version=SCHEMA_VERSION + 1,
            parameterization="geo",
            volume_window=256,
            clip_low=-5.0,
            clip_high=5.0,
        )


def test_constants_fitted_over_the_firewall_are_refused() -> None:
    """A contaminated constants file must not load at all, not merely warn."""
    payload = constants([load_spec("contract_geo_v1")]).model_dump(mode="json")
    payload["manifest"]["firewall_respected"] = False

    with pytest.raises(ValueError, match="contaminated"):
        ContractConstants.model_validate(payload)


def test_constants_from_a_partial_fit_are_refused() -> None:
    payload = constants([load_spec("contract_geo_v1")]).model_dump(mode="json")
    payload["manifest"]["partial"] = True

    with pytest.raises(ValueError, match="partial fit"):
        ContractConstants.model_validate(payload)


def test_a_missing_slice_names_itself_rather_than_falling_back() -> None:
    """A silent fallback to another asset class's scale is unexplainable four versions later."""
    spec = load_spec("contract_geo_v1")

    with pytest.raises(KeyError, match="no constants for spec"):
        constants([spec]).scaling_for(spec, "equity", "5m")


def test_the_leaky_spec_is_refused_unless_the_caller_asks_for_it() -> None:
    spec = load_spec("contract_kronos_zscore_v0")

    with pytest.raises(ContractError) as caught:
        transform(_bars(), spec, None, asset_class="crypto", frequency="1h")

    assert caught.value.code == "leaky_spec"


def test_the_leaky_spec_cannot_be_inverted() -> None:
    spec = load_spec("contract_kronos_zscore_v0")
    block = transform(_bars(), spec, None, asset_class="crypto", frequency="1h", allow_leaky=True)

    with pytest.raises(ContractError) as caught:
        inverse(block, spec, constants([spec]))

    assert caught.value.code == "leaky_spec"


@pytest.mark.causality
def test_the_leaky_spec_fails_prefix_consistency_by_construction() -> None:
    """Documentation by test: this is what the audit catching a leak looks like.

    `kronos-zscore-v0` normalizes each feature against the mean and standard deviation of the
    window it is normalizing, so extending the window rewrites every row that came before. The
    audit is required to notice.
    """
    spec = load_spec("contract_kronos_zscore_v0")
    bars = _bars(40)
    full = transform(bars, spec, None, asset_class="crypto", frequency="1h", allow_leaky=True)

    prefix = transform(
        bars.slice(0, 21), spec, None, asset_class="crypto", frequency="1h", allow_leaky=True
    )

    assert not np.array_equal(prefix.values, full.values[:20])


@pytest.mark.causality
def test_a_forward_looking_median_window_fails_prefix_consistency() -> None:
    """The live-fire drill, frozen as a test.

    ADR-0020's causality argument is only worth anything if the audit fails when a window reaches
    forward. `_leaky_median` is that reach: it averages the bars **after** t instead of before.
    Extending the series then rewrites rows the honest version had already fixed, and
    prefix-consistency is exactly the property that notices.

    Note what this does and does not catch. A window that merely includes bar t itself stays
    prefix-consistent — a self-normalizing feature leaks nothing from the future. That case is
    forbidden separately, by the strictly-past window, and caught by
    `test_the_window_ends_before_its_own_index` in the rolling tests. Two properties, two leaks.
    """

    def _leaky_median(a: np.ndarray, window: int) -> np.ndarray:
        out = np.full(a.size, np.nan)
        for t in range(1, a.size):
            out[t] = np.median(a[t : t + window])
        return out

    a = np.arange(20.0)

    leaky_prefix = _leaky_median(a[:11], 4)

    np.testing.assert_array_equal(
        strictly_past_median(a[:11], 4)[1:], strictly_past_median(a, 4)[1:11]
    )
    assert not np.array_equal(leaky_prefix[1:], _leaky_median(a, 4)[1:11])


def test_the_firewall_config_hashes_to_what_adr_0021_records() -> None:
    """The hash-commit. A firewall that can be edited without anybody noticing is not a firewall.

    If this fails because the firewall was deliberately moved, ADR-0021 is what has to change
    first, and the constants have to be refitted after it.
    """
    firewall = load_firewall()

    assert firewall.firewall_ts == 1_735_689_600_000
    assert firewall.firewall_date_utc == "2025-01-01T00:00:00Z"
    assert firewall_sha256() == "94dd8b5072b01f746c03537450b6559180f21e87e3031fe22daad6c04719e871"


#: Every (asset class, frequency) the M0 corpus actually holds. US equities are daily only
#: (ADR-0016), so a `equity/1h` row in the constants would mean the fit invented one.
CORPUS_SLICES = (
    ("crypto", "1h"),
    ("crypto", "1d"),
    ("fx", "1h"),
    ("fx", "1d"),
    ("commodity", "1h"),
    ("commodity", "1d"),
    ("equity", "1d"),
)


@pytest.mark.parametrize("spec_name", ["contract_geo_v1", "contract_ret_v1"])
def test_the_committed_constants_cover_every_slice_of_the_corpus(spec_name: str) -> None:
    """The Phase B output, checked as data rather than trusted as a build artifact."""
    spec = load_spec(spec_name)
    table = load_constants()

    for asset_class, frequency in CORPUS_SLICES:
        scaling = table.scaling_for(spec, asset_class, frequency)
        assert len(scaling) == 6
        assert all(s.scale > 0 and math.isfinite(s.scale) for s in scaling)


@pytest.mark.parametrize("frequency", ["1h", "1d"])
def test_the_fitted_wicks_are_one_sided(frequency: str) -> None:
    """`upper` sits above zero and `lower` below it, in every slice.

    It follows from `high >= open >= low`, so a violation means the fit wrote the columns in the
    wrong order — a bug that leaves six plausible numbers per slice and no other symptom.
    """
    spec = load_spec("contract_geo_v1")
    table = load_constants()

    for asset_class, freq in CORPUS_SLICES:
        if freq != frequency:
            continue
        by_name = dict(
            zip(spec.feature_names, table.scaling_for(spec, asset_class, freq), strict=True)
        )
        assert by_name["upper"].center > 0, asset_class
        assert by_name["lower"].center < 0, asset_class


@pytest.mark.parametrize("frequency", ["1h", "1d"])
def test_crypto_is_fitted_wider_than_fx_at_the_same_frequency(frequency: str) -> None:
    """A spot-check on the fit against what anybody who has looked at a chart expects.

    Not a property of the contract — a property of the market, asserted because if it ever fails
    the likely cause is a slice label crossing wires somewhere between the registry and the fit.
    """
    spec = load_spec("contract_geo_v1")
    table = load_constants()

    crypto = dict(
        zip(spec.feature_names, table.scaling_for(spec, "crypto", frequency), strict=True)
    )
    fx = dict(zip(spec.feature_names, table.scaling_for(spec, "fx", frequency), strict=True))

    assert crypto["body"].scale > fx["body"].scale


#: What the contract may import at module scope. Anything that reads a file or opens a socket
#: makes `transform` a function of more than its arguments, which is the one thing it may not be.
_ALLOWED_ROOTS = {
    "__future__",
    "base64",
    "dataclasses",
    "hashlib",
    "math",
    "numpy",
    "pathlib",
    "pyarrow",
    "pydantic",
    "typing",
    "yaml",
    "axiom",
}
_FORBIDDEN = {"huggingface_hub", "httpx", "requests", "urllib", "socket", "subprocess"}

#: `corpus.py` is the driver: it parses bytes somebody else downloaded and writes reports.
#: `spec.py` reads YAML by design -- loading a config is what it is for.
_IMPURE_BY_DESIGN = {"__init__.py", "corpus.py"}


def _toplevel_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize(
    "path",
    sorted(
        p
        for p in (Path(__file__).resolve().parents[1] / "src" / "axiom" / "contract").glob("*.py")
        if p.name not in _IMPURE_BY_DESIGN
    ),
    ids=lambda p: p.name,
)
def test_the_contract_stays_pure(path: Path) -> None:
    """Same rule the v0.3 engine carries: an offline module stays offline."""
    imports = _toplevel_imports(path)

    assert not imports & _FORBIDDEN
    assert imports <= _ALLOWED_ROOTS, f"{path.name} imports {sorted(imports - _ALLOWED_ROOTS)}"


# --- the pinned regression snapshots -------------------------------------------------------

SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "tests" / "snapshots" / "contract_v1.json"


def test_a_snapshot_hash_exists_for_every_pinned_series_and_spec() -> None:
    """The cheapest tripwire in the project: five series, two specs, ten hashes, no data.

    Recomputing them needs the corpus, so this test checks the shape rather than the values. The
    value half is the diff: any contract change that moves a number moves these hashes, and a
    commit that changes them without a `schema_version` bump is a commit to argue with.
    """
    from axiom.contract.corpus import PINNED_SERIES

    hashes = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    expected = {f"{series}/{spec}" for series in PINNED_SERIES for spec in ("geo-v1", "ret-v1")}
    assert set(hashes) == expected
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in hashes.values()
    )


def test_the_snapshots_were_cut_against_the_committed_constants() -> None:
    """A hash cut under different constants describes a contract nobody is running.

    The dryrun that produced the snapshots loads the committed constants file, so the two move
    together or not at all. This asserts they did.
    """
    hashes = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    assert len(set(hashes.values())) == len(hashes)
    assert load_constants().manifest.firewall_respected
