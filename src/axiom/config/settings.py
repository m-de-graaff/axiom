"""Settings (environment) and run configuration (YAML).

Two different things, kept apart on purpose. ``AxiomSettings`` holds what changes with *where*
the code runs: tokens, namespaces. ``LoopConfig`` holds what changes with *what experiment* is
running, and is the thing the config hash is taken over.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AxiomSettings(BaseSettings):
    """Environment-derived settings.

    Reads ``.env`` on the laptop and plain environment variables in cloud kernels, where writing
    a dotfile would be a needless place for a token to land.
    """

    model_config = SettingsConfigDict(
        env_prefix="AXIOM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_token: SecretStr | None = None
    hf_namespace: str = "m-de-graaff"
    runs_repo: str = "axiom-runs"
    raw_repo: str = "axiom-raw"

    @property
    def runs_repo_id(self) -> str:
        return f"{self.hf_namespace}/{self.runs_repo}"

    @property
    def raw_repo_id(self) -> str:
        return f"{self.hf_namespace}/{self.raw_repo}"


class LoopConfig(BaseModel):
    """The v0.0 dummy training run.

    Every field except ``run_id`` and ``backend_tag`` feeds the config hash, so adding a field
    here changes the identity of every run that follows.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    seed: int = 1337
    total_steps: int = Field(gt=0)
    save_every: int = Field(gt=0)
    sleep_s: float = Field(default=0.05, ge=0.0)
    backend_tag: str = "local"
    schema_version: int = 0


#: Fields that identify a particular run rather than the experiment it belongs to. Two runs that
#: differ only in these are the same experiment and must hash identically.
VOLATILE_FIELDS: frozenset[str] = frozenset({"run_id", "backend_tag"})


def resolve_config_path(name_or_path: str | Path) -> Path:
    """Accept either a filesystem path or a bare config name shipped inside the package.

    Cloud kernels install a wheel and have no checkout, so ``axiom loop run --config loop_test``
    has to work with nothing but the installed package. A bare name resolves against
    ``axiom/configs/``; anything that looks like a path is used as given.
    """
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate

    name = candidate.name
    packaged = files("axiom.configs") / (name if name.endswith(".yaml") else f"{name}.yaml")
    if packaged.is_file():
        return Path(str(packaged))

    raise FileNotFoundError(
        f"no config at {name_or_path!r}, and none packaged as axiom/configs/{name}.yaml"
    )


def load_config(name_or_path: str | Path) -> LoopConfig:
    """Load a ``LoopConfig`` from YAML, rejecting unknown keys.

    Unknown keys are an error rather than a warning because a typo in a config filename silently
    training the wrong thing for 12 hours is the failure this guards against.
    """
    path = resolve_config_path(name_or_path)
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return LoopConfig.model_validate(raw)
