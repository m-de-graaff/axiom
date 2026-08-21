"""Modal CPU job: the two v0.4 contract passes, as backend #2.

Same code as `axiom contract fit-constants` and `axiom contract dryrun` — this file only changes
who downloads the bytes and how many workers do it at once. The GitHub runner path is the one that
runs today; Modal is blocked on the account gate (ADR-0009), and keeping both means the version
that unblocks it does not also have to write the job.

Fan-out is over **artifacts**, not segments: an artifact holds every segment of one series, so one
download serves all of them. The map function returns sketches in their base64 transport form, so a
worker sends back 480 KB of counts rather than a feature block.

Run with `modal run remote/modal/contract_run.py --job fit`. Secrets `axiom-gh` (image build) and
`axiom-hf` (run time) must exist in the workspace.
"""

import modal

REPO = "m-de-graaff/axiom"
BRANCH = "main"
SPECS = ("contract_geo_v1", "contract_ret_v1")

app = modal.App("axiom-contract")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"git+https://x-access-token:$GH_PAT@github.com/{REPO}.git@{BRANCH}",
        secrets=[modal.Secret.from_name("axiom-gh")],
    )
)

HF = [modal.Secret.from_name("axiom-hf")]


def _hf_token() -> str:
    import os

    os.environ.setdefault("AXIOM_HF_TOKEN", os.environ["HF_TOKEN"])
    return os.environ["HF_TOKEN"]


def _fetch(repo_id: str, path: str, token: str) -> bytes:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset", token=token)
    with open(local, "rb") as handle:
        return handle.read()


@app.function(image=image, secrets=HF, timeout=60 * 20, max_containers=32)
def fit_one(payload: dict) -> dict:
    """Sketch one artifact's pre-firewall feature distributions.

    Failures come back as data rather than as an exception, so one unreadable file cannot take the
    run down with it.
    """
    from axiom.contract.corpus import SegmentRef, fit_artifact
    from axiom.contract.spec import load_spec

    token = _hf_token()
    refs = [SegmentRef(**ref) for ref in payload["refs"]]
    try:
        data = _fetch(payload["repo_id"], payload["artifact_path"], token)
        sketches, skipped = fit_artifact(
            data, refs, [load_spec(name) for name in SPECS], payload["firewall_ts"]
        )
    except Exception as exc:
        return {"error": f"{payload['artifact_path']}: {type(exc).__name__}: {exc}"}
    return {"error": "", "sketches": sketches.to_dict(), "skipped": len(skipped)}


@app.function(image=image, secrets=HF, timeout=60 * 20, max_containers=32)
def dryrun_one(payload: dict) -> dict:
    """Stream one artifact's segments through both specs, keeping statistics and nothing else."""
    from axiom.contract.corpus import SegmentRef, dryrun_artifact
    from axiom.contract.spec import load_constants, load_spec

    token = _hf_token()
    refs = [SegmentRef(**ref) for ref in payload["refs"]]
    try:
        data = _fetch(payload["repo_id"], payload["artifact_path"], token)
        result, snapshots = dryrun_artifact(
            data,
            refs,
            [load_spec(name) for name in SPECS],
            load_constants(),
            audit_splits=payload["audit_splits"],
        )
    except Exception as exc:
        return {"error": f"{payload['artifact_path']}: {type(exc).__name__}: {exc}"}
    return {"error": "", "result": result.to_dict(), "snapshots": snapshots}


def _payloads(repo_id: str, limit: int | None):
    import dataclasses

    from axiom.clean.run import clean_paths
    from axiom.config.settings import AxiomSettings
    from axiom.contract.corpus import group_by_artifact, read_segment_refs

    token = _hf_token()
    settings = AxiomSettings()
    refs = read_segment_refs(_fetch(settings.raw_repo_id, clean_paths(1)["segments"], token))
    grouped = group_by_artifact(refs)
    paths = sorted(grouped)
    if limit is not None:
        paths = paths[:limit]
    return refs, [
        {
            "repo_id": repo_id,
            "artifact_path": path,
            "refs": [dataclasses.asdict(ref) for ref in grouped[path]],
        }
        for path in paths
    ]


@app.function(image=image, secrets=HF, timeout=60 * 120)
def drive(job: str = "fit", limit: int | None = None) -> str:
    """Read the segment index, fan out, reduce, print. Writes nothing to the Hub.

    v0.4's outputs are committed files, not dataset artifacts (see `docs/REPOS.md`), so the driver
    prints them and a human commits what they have read.
    """
    import datetime as dt

    from axiom.config.settings import AxiomSettings
    from axiom.contract.corpus import (
        DryrunResult,
        constants_tables,
        constants_yaml,
        pick_audit_segments,
        quantile_rows,
        usable_windows,
    )
    from axiom.contract.reports import contract_qa_markdown
    from axiom.contract.spec import (
        SCHEMA_VERSION,
        firewall_sha256,
        load_constants,
        load_firewall,
        load_spec,
    )
    from axiom.contract.stats import SketchSet

    settings = AxiomSettings()
    repo_id = settings.raw_repo_id
    specs = [load_spec(name) for name in SPECS]
    firewall = load_firewall()
    refs, payloads = _payloads(repo_id, limit)
    partial = limit is not None and limit < len(payloads)
    print(f"{len(refs)} segment(s) in {len(payloads)} artifact(s); job {job}")

    if job == "fit":
        for payload in payloads:
            payload["firewall_ts"] = firewall.firewall_ts
        merged, failures, skipped = SketchSet(), [], 0
        for result in fit_one.map(payloads):
            if result["error"]:
                failures.append(result["error"])
                continue
            merged.merge(SketchSet.from_dict(result["sketches"]))
            skipped += result["skipped"]
        respected = merged.max_ts < firewall.firewall_ts
        if not respected:
            return f"FIREWALL BREACH: consumed a bar at {merged.max_ts}"
        manifest = {
            "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": BRANCH,
            "registry_hash": "",
            "clean_config_hash": "",
            "firewall_ts": firewall.firewall_ts,
            "firewall_config_sha256": firewall_sha256(),
            "firewall_respected": respected,
            "segments_consumed": merged.segments,
            "bars_consumed": merged.bars,
            "partial": partial,
        }
        print(constants_yaml(constants_tables(merged, specs), manifest, SCHEMA_VERSION).decode())
        return f"fit: {merged.segments} segments, {merged.bars} bars, {len(failures)} failed"

    splits = pick_audit_segments(refs)
    for payload in payloads:
        payload["audit_splits"] = splits
    total, snapshots, failures = DryrunResult(), {}, []
    for result in dryrun_one.map(payloads):
        if result["error"]:
            failures.append(result["error"])
            continue
        total.merge(DryrunResult.from_dict(result["result"]))
        snapshots.update(result["snapshots"])
    print(
        contract_qa_markdown(
            quantile_rows(total),
            total,
            constants=load_constants(),
            specs=specs,
            usable_windows_512=sum(usable_windows(r.n_bars) for r in refs),
            snapshot_hashes=snapshots,
            failures=failures,
            partial=partial,
        )
    )
    return (
        f"dryrun: {total.rows} feature rows, {total.audits_passed}/{total.audits_run} audits, "
        f"{len(failures)} failed"
    )


@app.local_entrypoint()
def main(job: str = "fit", limit: int | None = None) -> None:
    print(drive.remote(job, limit))
