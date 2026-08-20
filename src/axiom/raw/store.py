"""Where raw-tier artifacts land: a local directory, or the private `axiom-raw` dataset.

Two implementations because there are genuinely two destinations. The Hub one is what a pull job
uses. The local one is what the tests use, and what a debugging session uses when it wants to
look at a Parquet file without a network round trip -- so it is not a mock, it is the same class
of thing pointed somewhere cheaper.

The store is also the resume mechanism. :meth:`read_sidecar` is the only state a pull job
consults on startup: if the remote sidecar already names the same source checksums, the symbol is
done and the job moves on (ADR-0010, `provenance.is_current`).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Protocol

from axiom.provenance.manifest import SIDECAR_SUFFIX, FileManifest

log = logging.getLogger("axiom.raw")

#: Files per Hub commit. The Hub is happier with a few dozen files per commit than with one
#: commit per file, and a batch that fails is a batch worth retrying whole.
DEFAULT_BATCH = 50


class RawStore(Protocol):
    """Read a sidecar, write an artifact plus its sidecar, flush whatever is pending."""

    def read_sidecar(self, artifact_path: str) -> FileManifest | None: ...

    def put(self, artifact_path: str, data: bytes, manifest: FileManifest) -> None: ...

    def flush(self) -> None: ...


class LocalRawStore:
    """A directory laid out exactly like the dataset repo."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, artifact_path: str) -> Path:
        return self.root / artifact_path

    def read_sidecar(self, artifact_path: str) -> FileManifest | None:
        path = self._path(artifact_path + SIDECAR_SUFFIX)
        if not path.exists():
            return None
        return FileManifest.from_json(path.read_text(encoding="utf-8"))

    def put(self, artifact_path: str, data: bytes, manifest: FileManifest) -> None:
        path = self._path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._path(artifact_path + SIDECAR_SUFFIX).write_text(manifest.to_json(), encoding="utf-8")

    def flush(self) -> None:
        return None


class HubRawStore:
    """The private `axiom-raw` dataset, written in batched commits.

    Artifacts are staged in a container-local directory and committed a batch at a time. Commits
    are synchronous: a full v0.1 pull is roughly a dozen batches of small files, so overlapping
    the uploads with the downloads would save a minute or two of a job that spends most of an
    hour talking to Binance. :meth:`flush` commits whatever is left.

    # ponytail: synchronous commits; overlap uploads with downloads if the corpus grows enough
    # that upload time stops being noise (v0.2, when the file count goes up an order of magnitude)
    """

    def __init__(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        staging: Path | str,
        batch_size: int = DEFAULT_BATCH,
        api: Any = None,
    ) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.staging = Path(staging)
        self.batch_size = batch_size
        self._api = api or HfApi(token=token)
        self._token = token
        self._staged = 0

    # --- reads ------------------------------------------------------------------------

    def read_sidecar(self, artifact_path: str) -> FileManifest | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=artifact_path + SIDECAR_SUFFIX,
                repo_type="dataset",
                token=self._token,
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        except Exception as exc:  # a network blip must not be read as "not pulled yet"
            raise RuntimeError(f"could not read sidecar for {artifact_path}: {exc}") from exc
        return FileManifest.from_json(Path(path).read_text(encoding="utf-8"))

    # --- writes -----------------------------------------------------------------------

    def put(self, artifact_path: str, data: bytes, manifest: FileManifest) -> None:
        staged = self.staging / artifact_path
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(data)
        (self.staging / (artifact_path + SIDECAR_SUFFIX)).write_text(
            manifest.to_json(), encoding="utf-8"
        )
        self._staged += 2
        if self._staged >= self.batch_size:
            self._commit()

    def upload_json(self, path_in_repo: str, text: str) -> None:
        """Write one small file directly, outside the batching. Used for run manifests."""
        self._api.upload_file(
            path_or_fileobj=text.encode("utf-8"),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
        )

    def _commit(self) -> None:
        if not self._staged:
            return
        log.info("committing %d file(s) to %s", self._staged, self.repo_id)
        self._api.upload_folder(
            folder_path=str(self.staging),
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"pull: {self._staged} file(s)",
        )
        shutil.rmtree(self.staging, ignore_errors=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        self._staged = 0

    def flush(self) -> None:
        self._commit()
