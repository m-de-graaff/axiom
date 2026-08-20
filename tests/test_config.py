"""Config loading and hashing: the identity of an experiment."""

from __future__ import annotations

import re

import pytest
import yaml
from pydantic import ValidationError

from axiom.config.hashing import config_hash
from axiom.config.settings import LoopConfig, load_config


def make_config(**overrides) -> LoopConfig:
    return LoopConfig(
        **{
            "run_id": "test-001",
            "seed": 1337,
            "total_steps": 100,
            "save_every": 10,
            "sleep_s": 0.0,
            "backend_tag": "local",
            "schema_version": 0,
            **overrides,
        }
    )


def write_yaml(tmp_path, payload) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_load_config_rejects_unknown_keys(tmp_path):
    path = write_yaml(
        tmp_path, {"run_id": "x", "total_steps": 10, "save_every": 5, "totl_steps": 10}
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_rejects_a_non_mapping_document(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_config(str(path))


def test_load_config_rejects_zero_total_steps(tmp_path):
    path = write_yaml(tmp_path, {"run_id": "x", "total_steps": 0, "save_every": 5})

    with pytest.raises(ValidationError):
        load_config(path)


def test_config_hash_is_stable_across_yaml_key_order(tmp_path):
    fields = {"run_id": "x", "seed": 7, "total_steps": 100, "save_every": 10}
    forward = load_config(write_yaml(tmp_path, fields))
    reversed_path = tmp_path / "reversed.yaml"
    reversed_path.write_text(yaml.safe_dump(dict(reversed(list(fields.items())))), encoding="utf-8")

    assert config_hash(forward) == config_hash(load_config(str(reversed_path)))


def test_config_hash_ignores_run_id_and_backend_tag():
    local = make_config(run_id="local-run", backend_tag="local")
    kaggle = make_config(run_id="kaggle-run", backend_tag="kaggle")

    assert config_hash(local) == config_hash(kaggle)


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", 42), ("total_steps", 200), ("save_every", 20), ("schema_version", 1)],
)
def test_config_hash_changes_when_a_substantive_field_changes(field, value):
    base = make_config()

    assert config_hash(base) != config_hash(make_config(**{field: value}))


def test_config_hash_short_form_is_a_prefix_of_the_full_digest():
    cfg = make_config()

    assert config_hash(cfg, short=False).startswith(config_hash(cfg))


def test_a_bare_name_resolves_to_the_packaged_config():
    """Cloud kernels install a wheel and have no checkout, so this path must work without one."""
    cfg = load_config("loop_test")

    assert cfg.run_id == "loop-test-001"
    assert cfg.total_steps == 1000


def test_an_explicit_path_still_wins_over_the_packaged_name(tmp_path):
    path = write_yaml(tmp_path, {"run_id": "from-disk", "total_steps": 5, "save_every": 5})

    assert load_config(path).run_id == "from-disk"


def test_an_unknown_config_name_names_both_places_it_looked():
    with pytest.raises(FileNotFoundError, match=re.escape("axiom/configs/nope.yaml")):
        load_config("nope")
