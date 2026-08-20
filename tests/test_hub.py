"""Hub transport logic, with the network stubbed.

What is worth testing here is the reasoning around the transfer, not the transfer: that a
missing pointer means "start fresh" rather than an error, that a sha mismatch refuses rather than
resumes, and that a missing token fails before any request goes out.

The real round trip against `axiom-runs` is a manual drill in `docs/RUNBOOK.md`, because it needs
credentials this suite deliberately does not have.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from axiom.config.settings import AxiomSettings
from axiom.ops import hub
from axiom.ops.checkpoint import META_FILENAME, STATE_FILENAME, TrainState, save_checkpoint
from axiom.ops.seeding import capture_rng_state, seed_all


def make_settings(token: str | None = "hf_fake") -> AxiomSettings:
    return AxiomSettings(
        hf_token=SecretStr(token) if token else None,
        hf_namespace="test-ns",
        runs_repo="axiom-runs",
    )


def make_state(step: int = 200) -> TrainState:
    seed_all(1337)
    return TrainState(
        step=step,
        acc=1.5,
        rng=capture_rng_state(),
        config_hash="deadbeef1234",
        run_id="run-x",
    )


def stub_downloads(monkeypatch, files: dict[str, Path]) -> list[str]:
    """Serve ``hf_hub_download`` from a dict of repo path -> local file. Records what was asked."""
    asked: list[str] = []

    def fake_download(*, repo_id, filename, repo_type, token):
        asked.append(filename)
        if filename not in files:
            raise FileNotFoundError(filename)
        return str(files[filename])

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    return asked


def test_runs_repo_id_joins_namespace_and_repo():
    assert make_settings().runs_repo_id == "test-ns/axiom-runs"


def test_pushing_without_a_token_fails_before_any_request(tmp_path):
    with pytest.raises(RuntimeError, match="AXIOM_HF_TOKEN"):
        hub.push_checkpoint(tmp_path, "run-x", 200, settings=make_settings(token=None))


def test_a_run_that_never_checkpointed_resumes_from_nothing(tmp_path, monkeypatch):
    """First launch on a fresh kernel. Must be None, not an exception."""
    stub_downloads(monkeypatch, {})

    assert hub.pull_latest("run-x", tmp_path, settings=make_settings()) is None


def test_pull_restores_the_state_the_pointer_names(tmp_path, monkeypatch):
    step_dir = save_checkpoint(make_state(), tmp_path / "remote")
    pointer = tmp_path / "latest.json"
    pointer.write_text(
        json.dumps(
            {
                "step": 200,
                "path_in_repo": "loop-test/run-x/step_00000200",
                "sha256": json.loads((step_dir / META_FILENAME).read_text())["sha256"],
            }
        ),
        encoding="utf-8",
    )
    stub_downloads(
        monkeypatch,
        {
            "loop-test/run-x/latest.json": pointer,
            f"loop-test/run-x/step_00000200/{STATE_FILENAME}": step_dir / STATE_FILENAME,
            f"loop-test/run-x/step_00000200/{META_FILENAME}": step_dir / META_FILENAME,
        },
    )

    state = hub.pull_latest("run-x", tmp_path / "local", settings=make_settings())

    assert state is not None
    assert (state.step, state.acc, state.config_hash) == (200, 1.5, "deadbeef1234")


def test_pull_refuses_a_payload_that_does_not_match_the_pointer_hash(tmp_path, monkeypatch):
    """A truncated upload resuming into garbage is the failure that would not announce itself."""
    step_dir = save_checkpoint(make_state(), tmp_path / "remote")
    pointer = tmp_path / "latest.json"
    pointer.write_text(
        json.dumps(
            {
                "step": 200,
                "path_in_repo": "loop-test/run-x/step_00000200",
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    stub_downloads(
        monkeypatch,
        {
            "loop-test/run-x/latest.json": pointer,
            f"loop-test/run-x/step_00000200/{STATE_FILENAME}": step_dir / STATE_FILENAME,
            f"loop-test/run-x/step_00000200/{META_FILENAME}": step_dir / META_FILENAME,
        },
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        hub.pull_latest("run-x", tmp_path / "local", settings=make_settings())


def test_push_writes_the_pointer_after_the_folder_it_names(tmp_path, monkeypatch):
    """Order matters: a pointer naming a directory that is still uploading is a broken resume."""
    step_dir = save_checkpoint(make_state(), tmp_path)
    calls: list[str] = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def upload_folder(self, **kwargs):
            calls.append(f"folder:{kwargs['path_in_repo']}")
            return "future"

        def upload_file(self, **kwargs):
            calls.append(f"file:{kwargs['path_in_repo']}")

    monkeypatch.setattr(hub, "HfApi", FakeApi)

    future = hub.push_checkpoint(step_dir, "run-x", 200, settings=make_settings())

    assert future == "future"
    assert calls == [
        "folder:loop-test/run-x/step_00000200",
        "file:loop-test/run-x/latest.json",
    ]
