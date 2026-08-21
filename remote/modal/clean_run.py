"""Modal CPU job: clean the whole corpus into a segment index (v0.3, Phase E).

Fan-out is real here. The registry names roughly fourteen thousand bar artifacts; each one is a
download of a few hundred kilobytes and a handful of vectorized passes, so the cost is almost
entirely latency and `.map()` is what removes it. The driver concatenates what comes back, checks
the corpus-wide invariants, and uploads three files.

Run it with `modal run remote/modal/clean_run.py`, or `just clean-corpus`. Secrets `axiom-gh`
(GH_PAT, image build) and `axiom-hf` (HF_TOKEN, run time) must exist in the workspace.

The map function returns **rows, not tables**: a worker that returned Arrow would serialize a
schema fourteen thousand times, and the driver has to concatenate anyway. `--limit` exists for the
smoke run; a partial clean is recorded as partial in the run manifest rather than looking like a
full one.
"""

import modal

REPO = "m-de-graaff/axiom"
BRANCH = "main"
CONFIG = "clean_v1"

app = modal.App("axiom-clean")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"git+https://x-access-token:$GH_PAT@github.com/{REPO}.git@{BRANCH}[calendars]",
        secrets=[modal.Secret.from_name("axiom-gh")],
    )
)

HF = [modal.Secret.from_name("axiom-hf")]


def _hf_token() -> str:
    import os

    os.environ.setdefault("AXIOM_HF_TOKEN", os.environ["HF_TOKEN"])
    return os.environ["HF_TOKEN"]


@app.function(image=image, secrets=HF, timeout=60 * 20, max_containers=32)
def clean_one(payload: dict) -> dict:
    """Clean one artifact. The unit `.map()` fans out over.

    Failures come back as data rather than as an exception, so one unreadable file cannot take
    the run down with it. The driver counts them and the run manifest lists them.
    """
    from huggingface_hub import hf_hub_download

    from axiom.clean.config import load_clean_config
    from axiom.clean.run import ArtifactRef, clean_artifact

    token = _hf_token()
    ref = ArtifactRef(**payload["ref"])
    try:
        path = hf_hub_download(
            repo_id=payload["repo_id"],
            filename=ref.artifact_path,
            repo_type="dataset",
            token=token,
        )
        with open(path, "rb") as handle:
            result = clean_artifact(handle.read(), ref, load_clean_config(CONFIG))
    except Exception as exc:
        return {
            "artifact_path": ref.artifact_path,
            "error": f"{type(exc).__name__}: {exc}",
            "segments": [],
            "dropstats": [],
            "total_bars": 0,
            "kept_bars": 0,
        }
    return {
        "artifact_path": ref.artifact_path,
        "error": "",
        "segments": result.segments,
        "dropstats": result.dropstats,
        "total_bars": result.total_bars,
        "kept_bars": result.kept_bars,
    }


@app.function(image=image, secrets=HF, timeout=60 * 90)
def drive(limit: int | None = None) -> str:
    """Read the registry, fan out, reduce, upload."""
    import dataclasses
    import io
    import time

    from huggingface_hub import HfApi, hf_hub_download

    from axiom.clean.config import load_clean_config
    from axiom.clean.reports import clean_summary_markdown
    from axiom.clean.run import CleanRun, bar_artifacts, clean_paths, write_outputs
    from axiom.config.settings import AxiomSettings
    from axiom.registry import REGISTRY_PATH, read_registry
    from axiom.registry.build import registry_metadata

    token = _hf_token()
    settings = AxiomSettings()
    repo_id = settings.raw_repo_id
    config = load_clean_config(CONFIG)

    registry_file = hf_hub_download(
        repo_id=repo_id, filename=REGISTRY_PATH, repo_type="dataset", token=token
    )
    with open(registry_file, "rb") as handle:
        registry = read_registry(handle.read())
    refs = bar_artifacts(registry)
    if limit is not None:
        refs = refs[:limit]
    print(f"{len(refs)} bar artifact(s), config hash {config.config_hash}")

    run = CleanRun(
        clean_version=config.clean_version,
        clean_config_hash=config.config_hash,
        registry_hash=registry_metadata(registry).get("axiom_registry_hash", ""),
    )
    started = time.monotonic()
    payloads = [{"repo_id": repo_id, "ref": dataclasses.asdict(ref)} for ref in refs]
    for result in clean_one.map(payloads):
        if result["error"]:
            run.failed += 1
            run.failures.append(
                {"artifact_path": result["artifact_path"], "error": result["error"]}
            )
            continue
        run.ok += 1
        run.segments.extend(result["segments"])
        run.dropstats.extend(result["dropstats"])
        run.total_bars += result["total_bars"]
        run.kept_bars += result["kept_bars"]
    run.wall_seconds = time.monotonic() - started

    outputs = write_outputs(run)
    api = HfApi(token=token)
    for path, data in outputs.items():
        api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="dataset",
        )

    import pyarrow.parquet as pq

    paths = clean_paths(config.clean_version)
    summary = clean_summary_markdown(
        pq.read_table(io.BytesIO(outputs[paths["segments"]])),
        pq.read_table(io.BytesIO(outputs[paths["dropstats"]])),
        clean_config_hash=config.config_hash,
        registry_hash=run.registry_hash,
    )
    api.upload_file(
        path_or_fileobj=summary.encode("utf-8"),
        path_in_repo=f"{paths['root']}/summary.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(summary)
    return run.line()


@app.local_entrypoint()
def main(limit: int | None = None) -> None:
    print(drive.remote(limit))
