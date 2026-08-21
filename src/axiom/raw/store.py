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

#: Hub HTTP timeouts, in seconds, applied when the caller has not set them.
#:
#: `huggingface_hub` defaults to ten seconds, which is fine for a handful of files and not fine
#: for thirteen thousand: a registry build lost 19 sidecars to `ReadTimeout` and then failed
#: outright when every retry round died early on one slow read. The Hub is not slow, it is
#: rate-shaping, and waiting is the correct response to that.
HF_TIMEOUT_ENV: dict[str, str] = {
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    "HF_HUB_ETAG_TIMEOUT": "60",
}


def set_hub_timeouts() -> None:
    """Widen the Hub HTTP timeouts unless the environment already says otherwise.

    Called by the commands that read the whole tier. Explicit rather than an import-time side
    effect, so a caller that wants the library defaults simply does not call it.
    """
    import os

    for name, value in HF_TIMEOUT_ENV.items():
        os.environ.setdefault(name, value)


#: Files per Hub commit.
#:
#: The number that governs this is not throughput, it is **the Hub's limit of 128 commits per
#: hour per repository**. v0.1's 600 series at 50 files a commit was 24 commits and nowhere near
#: it. The v0.2 equities tier is 12 436 series -- 24 872 files -- which at the same batch size is
#: roughly 500 commits, and the real pull died partway through with
#: `429 ... You have exceeded the rate limit for repository commits (128 per hour)`.
#:
#: 2 000 files a commit puts a full equities pull at about 13 commits, which leaves room for the
#: other sources and the registry to run in the same hour. Resume granularity is the cost: a
#: killed run loses up to a batch, so this trades a thousand series against being able to finish
#: at all.
#:
#: ponytail: batched `upload_folder`; switch to `upload_large_folder` if one source ever needs
#: more files than a batch of this size can carry in a single commit (the roadmap names it)
DEFAULT_BATCH = 2_000


def retry(call, *, what: str, attempts: int = 12, base_delay: float = 20.0):
    """Run ``call`` until it stops raising, backing off between tries.

    For work that makes progress even when it fails. A resumable download is exactly that: the
    attempt that hits a rate limit still leaves everything it fetched in the cache, so the next
    one has less to do. Retrying something that is *not* resumable this way would just be a
    slower way to fail twelve times.

    Twelve attempts because `snapshot_download` abandons the whole batch on the first file that
    times out, so a run of thirteen thousand small files needs more rounds than a run of a few
    hundred. Raising :data:`HF_TIMEOUT_ENV` matters more than the count -- it is what stops each
    round dying early.
    """
    import time

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = base_delay * attempt
            log.warning(
                "%s attempt %d/%d failed (%s: %s); retrying in %.0fs -- what it already "
                "fetched is kept",
                what,
                attempt,
                attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


class RawStore(Protocol):
    """Read a sidecar, write an artifact plus its sidecar, flush whatever is pending."""

    def read_sidecar(self, artifact_path: str) -> FileManifest | None: ...

    def get(self, artifact_path: str) -> bytes | None: ...

    def put(self, artifact_path: str, data: bytes, manifest: FileManifest) -> None: ...

    def stage_bytes(self, path_in_repo: str, data: bytes) -> None: ...

    def list_manifests(self) -> list[FileManifest]: ...

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

    def get(self, artifact_path: str) -> bytes | None:
        path = self._path(artifact_path)
        return path.read_bytes() if path.exists() else None

    def put(self, artifact_path: str, data: bytes, manifest: FileManifest) -> None:
        path = self._path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._path(artifact_path + SIDECAR_SUFFIX).write_text(manifest.to_json(), encoding="utf-8")

    def stage_bytes(self, path_in_repo: str, data: bytes) -> None:
        path = self._path(path_in_repo)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def list_manifests(self) -> list[FileManifest]:
        paths = sorted(self.root.rglob(f"*{SIDECAR_SUFFIX}"))
        return [FileManifest.from_json(p.read_text(encoding="utf-8")) for p in paths]

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

    def get(self, artifact_path: str) -> bytes | None:
        """Read an artifact back out of the dataset.

        A source that extends a series rather than rebuilding it needs the rows it already
        landed. Missing is not an error -- it is the first pull.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        staged = self.staging / artifact_path
        if staged.exists():  # written this run and not yet committed
            return staged.read_bytes()
        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=artifact_path,
                repo_type="dataset",
                token=self._token,
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        return Path(path).read_bytes()

    def list_manifests(self) -> list[FileManifest]:
        """Every sidecar in the dataset.

        Fetched as one snapshot rather than file by file. v0.1 had a few hundred and a serial
        loop was fine; v0.2 has 13,580, and 13,580 `hf_hub_download` calls is a HEAD plus a GET
        each, which the Hub answers with 429 and no amount of patience gets past. Same lesson as
        the bar tier, learned twice.

        The snapshot is resumable, so the retry wraps the whole thing: an attempt that is cut off
        leaves what it fetched in the cache and the next one starts further along.
        """
        from huggingface_hub import snapshot_download

        set_hub_timeouts()
        root = Path(
            retry(
                lambda: snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    allow_patterns=[f"raw/**/*{SIDECAR_SUFFIX}"],
                    token=self._token,
                    max_workers=4,
                ),
                what="sidecar snapshot",
            )
        )
        paths = sorted(root.rglob(f"*{SIDECAR_SUFFIX}"))
        log.info("%d sidecar(s) in the snapshot", len(paths))
        return [FileManifest.from_json(p.read_text(encoding="utf-8")) for p in paths]

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

    def stage_bytes(self, path_in_repo: str, data: bytes) -> None:
        """Add one file to the next batched commit.

        For writing many small files that are not bar artifacts -- twelve thousand sidecars, say.
        `upload_bytes` would be one Hub commit each, and the Hub allows 128 an hour.
        """
        staged = self.staging / path_in_repo
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(data)
        self._staged += 1
        if self._staged >= self.batch_size:
            self._commit()

    def upload_json(self, path_in_repo: str, text: str) -> None:
        """Write one small file directly, outside the batching. Used for run manifests."""
        self.upload_bytes(path_in_repo, text.encode("utf-8"))

    def upload_bytes(self, path_in_repo: str, data: bytes) -> None:
        """Write one file directly, outside the batching.

        Used for things that are not bar artifacts and have no sidecar -- run manifests, the
        corpus registry -- so they are not worth staging into a batch that exists to amortize
        commits over thousands of small files.
        """
        self._api.upload_file(
            path_or_fileobj=data,
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
