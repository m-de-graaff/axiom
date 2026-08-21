"""The `axiom` command line.

One entry point that every backend calls identically. A Kaggle kernel and the laptop run the same
argv, which is what makes a cloud result comparable to a local one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from axiom import __version__
from axiom.config.hashing import config_hash
from axiom.config.settings import load_config
from axiom.ops.logx import git_commit, setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
config_app = typer.Typer(no_args_is_help=True, help="Inspect run configuration.")
loop_app = typer.Typer(no_args_is_help=True, help="The v0.0 dispatch/checkpoint/resume loop.")
universe_app = typer.Typer(no_args_is_help=True, help="Build and inspect the pinned universe.")
pull_app = typer.Typer(no_args_is_help=True, help="Land source data in the raw tier.")
raw_app = typer.Typer(no_args_is_help=True, help="Inspect, verify and summarize the raw tier.")
registry_app = typer.Typer(no_args_is_help=True, help="Build and query the corpus registry.")
app.add_typer(config_app, name="config")
app.add_typer(loop_app, name="loop")
app.add_typer(universe_app, name="universe")
app.add_typer(pull_app, name="pull")
app.add_typer(raw_app, name="raw")
app.add_typer(registry_app, name="registry")


def _csv(value: str) -> list[str]:
    """Split a comma-separated option into a clean list."""
    return [item.strip() for item in value.split(",") if item.strip()]


@app.command()
def version() -> None:
    """Print the package version and the commit it was built from."""
    typer.echo(f"axiom {__version__} ({git_commit()})")


@config_app.command("hash")
def config_hash_cmd(
    path: Annotated[Path, typer.Argument(help="Path to a run config YAML.")],
) -> None:
    """Print the config hash, which is the identity of the experiment this config describes."""
    typer.echo(config_hash(load_config(path)))


@config_app.command("show")
def config_show(
    path: Annotated[Path, typer.Argument(help="Path to a run config YAML.")],
) -> None:
    """Print the parsed config, so a rejected key is visible before a 12-hour run starts."""
    cfg = load_config(path)
    for key, value in cfg.model_dump(mode="json").items():
        typer.echo(f"{key}: {value}")
    typer.echo(f"config_hash: {config_hash(cfg)}")


@loop_app.command("run")
def loop_run(
    config: Annotated[Path, typer.Option("--config", "-c", help="Run config YAML.")],
    resume: Annotated[
        bool, typer.Option("--resume", help="Resume from the newest checkpoint.")
    ] = False,
    backend_tag: Annotated[
        str | None, typer.Option("--backend-tag", help="Override the config's backend tag.")
    ] = None,
    total_steps: Annotated[
        int | None, typer.Option("--total-steps", help="Override the config's step count.")
    ] = None,
    save_every: Annotated[
        int | None, typer.Option("--save-every", help="Override the checkpoint interval.")
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Override the run id.")] = None,
    no_push: Annotated[
        bool, typer.Option("--no-push", help="Keep checkpoints local; skip the Hub entirely.")
    ] = False,
) -> None:
    """Run the loop to completion, checkpointing as it goes.

    The overrides exist so one config file serves every backend: Kaggle runs more steps than the
    laptop drill, and both are the same experiment except for the fields the config hash ignores.
    Overriding a hashed field (`--total-steps`, `--save-every`) makes it a different experiment,
    which is why a resume with a mismatched hash refuses to start.
    """
    from axiom.loop.dummy_trainer import run

    setup_logging()
    cfg = load_config(config)
    updates = {
        k: v
        for k, v in {
            "backend_tag": backend_tag,
            "total_steps": total_steps,
            "save_every": save_every,
            "run_id": run_id,
        }.items()
        if v is not None
    }
    if updates:
        cfg = cfg.model_copy(update=updates)

    state = run(cfg, resume=resume, push=not no_push)
    typer.echo(f"final step={state.step} acc={state.acc!r}")


@loop_app.command("verify")
def loop_verify(
    config: Annotated[Path, typer.Option("--config", "-c", help="Run config YAML.")],
    kill_at: Annotated[int, typer.Option("--kill-at", help="Step to die at during run B.")] = 437,
) -> None:
    """The local determinism drill: run clean, run with a kill and a resume, compare.

    Exits non-zero if the two final states differ, so this is usable as a gate in a script as
    well as by eye. Never touches the Hub.
    """
    import shutil

    from axiom.loop.dummy_trainer import KilledAtStep, checkpoint_root, run

    setup_logging()
    base = load_config(config)

    clean_cfg = base.model_copy(update={"run_id": f"{base.run_id}-verify-a", "sleep_s": 0.0})
    shutil.rmtree(checkpoint_root(clean_cfg.run_id), ignore_errors=True)
    os.environ.pop("AXIOM_KILL_AT_STEP", None)
    clean = run(clean_cfg, resume=False, push=False)

    killed_cfg = base.model_copy(update={"run_id": f"{base.run_id}-verify-b", "sleep_s": 0.0})
    shutil.rmtree(checkpoint_root(killed_cfg.run_id), ignore_errors=True)
    os.environ["AXIOM_KILL_AT_STEP"] = str(kill_at)
    try:
        run(killed_cfg, resume=False, push=False)
    except KilledAtStep:
        pass
    else:
        typer.echo(f"fault injection never fired at step {kill_at}", err=True)
        raise typer.Exit(2)
    finally:
        os.environ.pop("AXIOM_KILL_AT_STEP", None)

    resumed = run(killed_cfg, resume=True, push=False)

    typer.echo(f"clean   step={clean.step} acc={clean.acc!r}")
    typer.echo(f"resumed step={resumed.step} acc={resumed.acc!r}")
    if (clean.step, clean.acc) != (resumed.step, resumed.acc):
        typer.echo("MISMATCH: resume is not bit-identical", err=True)
        raise typer.Exit(1)
    typer.echo("OK: resume is bit-identical")


@universe_app.command("build")
def universe_build(
    month: Annotated[
        str, typer.Option("--month", help="Selection month, YYYY-MM. Pinned, never 'now'.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Where to write the universe YAML.")],
    market: Annotated[
        list[str] | None,
        typer.Option("--market", help="Repeatable. 'spot=200' or bare 'spot' for the default."),
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="Cap on simultaneous requests to the bucket.")
    ] = 12,
) -> None:
    """Rank the bucket's symbols by the selection month's volume and emit the pinned universe.

    Runs in the cloud. The laptop invocation exists so the command can be tested against a fake
    client; a real build downloads a couple of thousand small archives, which is cloud work by
    the roadmap's rules even though it is small.
    """
    from axiom.sources.binance_vision import BinanceVision
    from axiom.universe.binance import build_universe

    setup_logging()
    defaults = {"spot": 200, "um": 100}
    top_n: dict[str, int] = {}
    for entry in market or ["spot", "um"]:
        name, _, count = entry.partition("=")
        top_n[name] = int(count) if count else defaults.get(name, 100)

    with BinanceVision(concurrency=concurrency) as client:
        universe = build_universe(client, month=month, top_n=top_n)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(universe.to_yaml(), encoding="utf-8")
    for name, symbols in sorted(universe.symbols.items()):
        typer.echo(f"{name}: {len(symbols)} symbols")
    typer.echo(f"universe_hash: {universe.universe_hash}")
    typer.echo(f"wrote {out}")


@universe_app.command("show")
def universe_show(
    path: Annotated[str, typer.Argument(help="Universe YAML path, or a packaged config name.")],
) -> None:
    """Print a universe's criteria and counts, verifying its hash on the way."""
    from axiom.universe.binance import load_universe

    universe = load_universe(path)
    typer.echo(f"selection_month: {universe.criteria.selection_month}")
    typer.echo(f"min_history_months: {universe.criteria.min_history_months}")
    for name, symbols in sorted(universe.symbols.items()):
        typer.echo(f"{name}: {len(symbols)} symbols, first 5 {symbols[:5]}")
    typer.echo(f"universe_hash: {universe.universe_hash}")


@universe_app.command("build-equities")
def universe_build_equities(
    out: Annotated[Path, typer.Option("--out", help="Where to write the universe YAML.")],
    top_n: Annotated[int, typer.Option("--top-n", help="Tickers to keep.")] = 3000,
    min_history_years: Annotated[int, typer.Option("--min-history-years")] = 5,
    window: Annotated[
        int, typer.Option("--window", help="Trading days the ranking metric is measured over.")
    ] = 252,
    dest: Annotated[
        Path | None, typer.Option("--dest", help="Build over a local raw tier instead.")
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency")] = 16,
) -> None:
    """Build the equities training universe from the pulled corpus (ADR-0016 criteria).

    Two passes on purpose. The history filter is answered from the registry — eighteen thousand
    rows of metadata, no downloads — and only the survivors are fetched to compute the ranking
    metric. Ranking first would pull every artifact in the tier to sort a list that discards
    most of them.

    The pulled corpus stays a superset of the result. This governs sampling from v0.5 onward,
    never what is stored.
    """
    from datetime import UTC, datetime

    from axiom.registry import REGISTRY_PATH, read_registry
    from axiom.registry.build import registry_metadata
    from axiom.universe.equities import build_equity_universe

    setup_logging()
    store = _raw_store(dest)
    if dest is not None:
        data = (Path(dest) / REGISTRY_PATH).read_bytes()
    else:
        data = store.get(REGISTRY_PATH)
    if data is None:
        typer.echo("no registry found; run `axiom registry build` first", err=True)
        raise typer.Exit(2)

    registry = read_registry(data)
    recorded_hash = registry_metadata(registry).get("axiom_registry_hash", "")

    universe = build_equity_universe(
        store,
        registry,
        registry_hash=recorded_hash,
        generated_at=datetime.now(UTC).date().isoformat(),
        min_history_years=min_history_years,
        top_n=top_n,
        window=window,
        concurrency=concurrency,
    )
    if not universe.symbols:
        typer.echo("no ticker cleared the criteria; is the equities tier populated?", err=True)
        raise typer.Exit(2)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(universe.to_yaml(), encoding="utf-8")
    typer.echo(
        f"{len(universe.symbols)} of {universe.candidates_considered} candidates kept; "
        f"universe_hash={universe.universe_hash}, from registry_hash={recorded_hash or 'unknown'}"
    )
    typer.echo(f"wrote {out}")


@pull_app.command("binance")
def pull_binance(
    universe: Annotated[
        str, typer.Option("--universe", help="Universe YAML path, or a packaged config name.")
    ] = "universe_v1",
    frequencies: Annotated[str, typer.Option("--frequencies")] = "1h,1d",
    markets: Annotated[str, typer.Option("--markets")] = "spot,um",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Smoke runs only. Symbols per market.")
    ] = None,
    symbols: Annotated[
        str | None, typer.Option("--symbols", help="Smoke runs only. Comma-separated.")
    ] = None,
    dest: Annotated[
        Path | None,
        typer.Option("--dest", help="Write to this directory instead of the Hub dataset."),
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    backend_tag: Annotated[str, typer.Option("--backend-tag")] = "local",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 12,
    force: Annotated[
        bool, typer.Option("--force", help="Re-pull even when the sidecar says it is current.")
    ] = False,
) -> None:
    """Pull the universe into `axiom-raw`, skipping whatever is already current.

    Safe to kill and rerun. Completed symbols are skipped by comparing the source checksums the
    bucket publishes now against the ones the remote sidecar records, so a resumed run costs one
    listing plus a handful of tiny CHECKSUM fetches per finished symbol.
    """
    from datetime import UTC, datetime

    from axiom.config.settings import AxiomSettings
    from axiom.provenance.manifest import PullRunManifest
    from axiom.raw.store import HubRawStore, LocalRawStore
    from axiom.sources.binance import build_tasks, loader_version, run_pull
    from axiom.sources.binance_vision import BinanceVision
    from axiom.universe.binance import load_universe

    setup_logging()
    config = load_universe(universe)
    wanted_markets = _csv(markets)
    wanted_frequencies = _csv(frequencies)
    symbol_filter = _csv(symbols) if symbols else None

    tasks = build_tasks(
        config.symbols,
        wanted_markets,
        wanted_frequencies,
        symbols=symbol_filter,
        limit=limit,
    )
    if not tasks:
        typer.echo("work list is empty; check --markets, --symbols and the universe", err=True)
        raise typer.Exit(2)

    pull_run_id = run_id or f"pull-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    manifest = PullRunManifest(
        pull_run_id=pull_run_id,
        started_at=datetime.now(UTC).isoformat(),
        loader_version=loader_version(),
        backend_tag=backend_tag,
        universe_hash=config.universe_hash,
        universe_path=str(universe),
        markets=wanted_markets,
        frequencies=wanted_frequencies,
        limit=limit,
        symbols_filter=symbol_filter or [],
    )

    settings = AxiomSettings()
    if dest is not None:
        store = LocalRawStore(dest)
    else:
        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        store = HubRawStore(
            settings.raw_repo_id,
            token=token,
            staging=Path(os.environ.get("AXIOM_STAGING_DIR", "/tmp/axiom-raw-staging")),
        )

    with BinanceVision(concurrency=concurrency) as client:
        run = run_pull(client, store, tasks, manifest, force=force)

    final = run.finish()
    _write_run_manifest(store, dest, f"manifests/pulls/{pull_run_id}.json", final)

    typer.echo(
        f"{pull_run_id}: ok={final.ok} skipped={final.skipped} failed={final.failed} "
        f"rows={final.total_rows} bytes={final.total_bytes}"
        + (" (PARTIAL)" if final.is_partial else "")
    )
    for failure in final.failures:
        typer.echo(f"  FAIL {failure.market}/{failure.frequency}/{failure.symbol}: {failure.error}")
    _exit_on_operational_failures(final)


@pull_app.command("dukascopy")
def pull_dukascopy(
    universe: Annotated[
        str, typer.Option("--universe", help="Universe YAML path, or a packaged config name.")
    ] = "universe_dukascopy_v1",
    frequencies: Annotated[str, typer.Option("--frequencies")] = "1h,1d",
    symbols: Annotated[
        str | None, typer.Option("--symbols", help="Smoke runs only. Comma-separated.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Smoke runs only. Instruments, in file order.")
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="Pin the run's date, YYYY-MM-DD. Defaults to today, UTC."),
    ] = None,
    dest: Annotated[
        Path | None,
        typer.Option("--dest", help="Write to this directory instead of the Hub dataset."),
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    backend_tag: Annotated[str, typer.Option("--backend-tag")] = "local",
    force: Annotated[
        bool, typer.Option("--force", help="Re-pull even when the sidecar says it is current.")
    ] = False,
) -> None:
    """Pull FX and commodities into `axiom-raw`, re-fetching only what can still change.

    Safe to kill and rerun. A finished instrument is skipped for the rest of the day and
    re-extended tomorrow, because a year that has ended cannot gain bars and the current one
    can (ADR-0015). ``--as-of`` pins that judgement for the whole run, so a pull that crosses
    midnight does not seal a year for half its instruments and not the other half.
    """
    from datetime import UTC, date, datetime

    from axiom.provenance.manifest import PullRunManifest
    from axiom.sources.base import loader_version, run_pull
    from axiom.sources.dukascopy import DukascopySource
    from axiom.universe.dukascopy import load_dukascopy_universe

    setup_logging()
    config = load_dukascopy_universe(universe)
    wanted_frequencies = _csv(frequencies)
    symbol_filter = _csv(symbols) if symbols else None
    pinned = date.fromisoformat(as_of) if as_of else datetime.now(UTC).date()

    source = DukascopySource(config, as_of=pinned)
    items = source.work_items(wanted_frequencies, symbols=symbol_filter, limit=limit)
    if not items:
        typer.echo("work list is empty; check --symbols and the universe", err=True)
        raise typer.Exit(2)

    pull_run_id = run_id or f"dukascopy-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    manifest = PullRunManifest(
        pull_run_id=pull_run_id,
        started_at=datetime.now(UTC).isoformat(),
        loader_version=loader_version(),
        backend_tag=backend_tag,
        universe_hash=config.universe_hash,
        universe_path=str(universe),
        markets=sorted({item.market for item in items}),
        frequencies=wanted_frequencies,
        limit=limit,
        symbols_filter=symbol_filter or [],
    )

    store = _raw_store(dest)
    run = run_pull(source, store, items, manifest, force=force)
    final = run.finish()

    _write_run_manifest(store, dest, f"manifests/pulls/{pull_run_id}.json", final)
    typer.echo(
        f"{pull_run_id}: ok={final.ok} skipped={final.skipped} failed={final.failed} "
        f"rows={final.total_rows} bytes={final.total_bytes} as_of={pinned}"
        + (" (PARTIAL)" if final.is_partial else "")
    )
    for failure in final.failures:
        typer.echo(f"  FAIL {failure.market}/{failure.frequency}/{failure.symbol}: {failure.error}")
    _exit_on_operational_failures(final)


@pull_app.command("stooq")
def pull_stooq(
    archive_url: Annotated[
        str | None,
        typer.Option(
            "--archive-url",
            help="Direct archive URL, taken from the browser after solving the CAPTCHA.",
        ),
    ] = None,
    from_staging: Annotated[
        str | None,
        typer.Option(
            "--from-staging",
            help="Path in axiom-raw to read the archive from, e.g. staging/stooq/d_us_txt.zip. "
            "The ADR-0016 fallback: only when the handed-over URL is bound to the IP that "
            "solved the CAPTCHA.",
        ),
    ] = None,
    symbols: Annotated[
        str | None, typer.Option("--symbols", help="Smoke runs only. Comma-separated tickers.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Smoke runs only. Tickers, in archive order.")
    ] = None,
    dest: Annotated[
        Path | None,
        typer.Option("--dest", help="Write to this directory instead of the Hub dataset."),
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    backend_tag: Annotated[str, typer.Option("--backend-tag")] = "local",
    force: Annotated[
        bool, typer.Option("--force", help="Re-pull even when the sidecar says it is current.")
    ] = False,
) -> None:
    """Land the US daily equities dump, cloud-side, from a handed-over URL.

    Stooq gates the bulk archive behind a CAPTCHA, so a human does the one thing only a human
    can: solve it and copy the resulting direct URL. The archive itself is downloaded here, on
    whatever machine this runs on, and the bytes never touch the laptop (ADR-0016).

    `--from-staging` is the sanctioned fallback for a URL that turns out to be bound to the IP
    that solved the CAPTCHA. Using it sets `staging_exception_used` in the run manifest, because
    the zero-bytes-on-the-laptop rule is only a rule if its exceptions are countable.
    """
    from datetime import UTC, datetime

    from axiom.provenance.manifest import PullRunManifest
    from axiom.sources.base import loader_version, run_pull
    from axiom.sources.stooq import StooqArchive, StooqSource, download_archive

    setup_logging()
    if bool(archive_url) == bool(from_staging):
        typer.echo("give exactly one of --archive-url and --from-staging", err=True)
        raise typer.Exit(2)

    store = _raw_store(dest)
    work_dir = Path(os.environ.get("AXIOM_STAGING_DIR", "/tmp/axiom-raw-staging")) / "stooq"
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / "d_us_txt.zip"

    if archive_url:
        typer.echo(f"downloading the archive from {archive_url}")
        download_archive(archive_url, local)
        archive = StooqArchive(url=archive_url, path=local)
    else:
        typer.echo(f"reading the archive from {from_staging}")
        data = store.get(str(from_staging))
        if data is None:
            typer.echo(f"no archive at {from_staging} in the raw tier", err=True)
            raise typer.Exit(2)
        local.write_bytes(data)
        archive = StooqArchive(
            url=f"axiom-raw://{from_staging}", path=local, staging_exception_used=True
        )

    typer.echo(f"archive sha256: {archive.sha256}")
    pull_run_id = run_id or f"stooq-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"

    with StooqSource(archive) as source:
        items = source.work_items(symbols=_csv(symbols) if symbols else None, limit=limit)
        if not items:
            typer.echo("no tickers matched; check --symbols and the archive layout", err=True)
            raise typer.Exit(2)

        manifest = PullRunManifest(
            pull_run_id=pull_run_id,
            started_at=datetime.now(UTC).isoformat(),
            loader_version=loader_version(),
            backend_tag=backend_tag,
            universe_hash="",  # the archive is the universe; there is nothing to pin separately
            universe_path=archive.url,
            markets=["us"],
            frequencies=["1d"],
            limit=limit,
            symbols_filter=_csv(symbols) if symbols else [],
            staging_exception_used=archive.staging_exception_used,
        )
        run = run_pull(source, store, items, manifest, force=force)
        skipped_short = list(source.skipped_short)

    final = run.finish()
    _write_run_manifest(store, dest, f"manifests/pulls/{pull_run_id}.json", final)

    typer.echo(
        f"{pull_run_id}: ok={final.ok} skipped={final.skipped} failed={final.failed} "
        f"short={len(skipped_short)} rows={final.total_rows} bytes={final.total_bytes}"
        + (" (PARTIAL)" if final.is_partial else "")
        + (" (STAGING EXCEPTION USED)" if final.staging_exception_used else "")
    )
    for failure in final.failures[:20]:
        typer.echo(f"  FAIL {failure.symbol}: {failure.error}")
    if final.failures[20:]:
        typer.echo(f"  ... and {len(final.failures) - 20} more")

    # Two kinds of failure, and only one of them is a parse failure.
    #
    # ADR-0016's 0.1% tolerance is about text this loader could not read. A ticker rejected by
    # `validate_bars` parsed perfectly: the vendor published a bar with `high < low`, which is not
    # something a market did, and ADR-0010 refuses it rather than repairing it. Gating on the two
    # together would fail a run over the vendor's data quality and call it a parser problem.
    attempted = final.ok + final.failed
    unreadable = [f for f in final.failures if "MalformedFile" in f.error]
    impossible = [f for f in final.failures if f not in unreadable]

    parse_rate = len(unreadable) / attempted if attempted else 0.0
    typer.echo(f"parse-failure rate: {parse_rate:.3%} of {attempted} files")
    typer.echo(
        f"rejected on invariants: {len(impossible)} file(s) "
        f"({len(impossible) / attempted if attempted else 0.0:.3%}) — the vendor published bars "
        "that cannot be true; they are absent from the tier and listed above"
    )
    if parse_rate > 0.001:
        typer.echo("parse-failure rate is over the 0.1% tolerance", err=True)
        raise typer.Exit(1)


def _exit_on_operational_failures(final) -> None:
    """Exit non-zero only when the *pipeline* failed, not when the vendor did.

    A series refused by `validate_bars` was ingested correctly and found to be impossible — the
    vendor published a bar with `high < low`, and ADR-0010 declines to repair it. That is a fact
    about the data, it is recorded in the run manifest and printed above, and it will recur on
    every future pull because the vendor is not going to fix 2007. Exiting non-zero on it means a
    Dukascopy pull reports failure forever over one bad instrument, which trains a human to stop
    reading the status.

    Anything else — a refused download, a Hub error, a parse that fell over — is operational and
    still fails the run.
    """
    operational = [f for f in final.failures if not f.error.startswith("ValueError: bars:")]
    rejected = len(final.failures) - len(operational)
    if rejected:
        typer.echo(
            f"{rejected} series rejected on invariants: the vendor published bars that cannot be "
            "true. They are absent from the tier and listed above."
        )
    if operational:
        raise typer.Exit(1)


def _write_run_manifest(store, dest: Path | None, path_in_repo: str, final) -> None:
    """Land the run manifest wherever the artifacts went."""
    from axiom.raw.store import LocalRawStore

    if isinstance(store, LocalRawStore):
        target = Path(dest or ".") / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(final.to_json(), encoding="utf-8")
    else:
        store.upload_json(path_in_repo, final.to_json())


def _raw_store(dest: Path | None):
    """A local directory when one is named, otherwise the private `axiom-raw` dataset."""
    from axiom.config.settings import AxiomSettings
    from axiom.raw.store import HubRawStore, LocalRawStore

    if dest is not None:
        return LocalRawStore(dest)
    settings = AxiomSettings()
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    return HubRawStore(
        settings.raw_repo_id,
        token=token,
        staging=Path(os.environ.get("AXIOM_STAGING_DIR", "/tmp/axiom-raw-staging")),
    )


@raw_app.command("inspect")
def raw_inspect(
    symbol: Annotated[str, typer.Argument(help="Symbol, e.g. BTCUSDT.")],
    market: Annotated[str, typer.Option("--market")] = "spot",
    frequency: Annotated[str, typer.Option("--frequency")] = "1h",
    rows: Annotated[int, typer.Option("--rows", help="Offending rows to print per code.")] = 8,
    concurrency: Annotated[int, typer.Option("--concurrency")] = 12,
) -> None:
    """Fetch one series, validate it, and print what is wrong — writing nothing anywhere.

    This is what to reach for when a pull reports a symbol failed on an invariant. It reproduces
    the failure and shows the rows that caused it, which is the difference between "43 rows are
    off the grid" and knowing which 43.
    """
    import numpy as np

    from axiom.schema.bars import count_gaps, grid_step_ms, validate_bars
    from axiom.sources.binance import PullTask, build_table, enumerate_sources
    from axiom.sources.binance_vision import BinanceVision

    setup_logging()
    task = PullTask(market, symbol.upper(), frequency)
    with BinanceVision(concurrency=concurrency) as client:
        urls = enumerate_sources(client, task)
        typer.echo(f"{task}: {len(urls)} source archive(s)")
        table = build_table(client, task, urls, validate=False)

    ts = table["ts"].to_numpy(zero_copy_only=False)
    step = grid_step_ms(frequency)
    typer.echo(f"rows={table.num_rows} first_ts={int(ts[0])} last_ts={int(ts[-1])}")
    typer.echo(f"gaps={count_gaps(ts, frequency)} step_ms={step}")

    report = validate_bars(table, frequency)
    typer.echo(report.summary())
    for code, violation in sorted(report.violations.items()):
        typer.echo(f"\n{code}: {violation}")
        if code == "ts_off_grid":
            offenders = np.flatnonzero(ts % step != 0)[:rows]
        elif code == "ts_not_increasing":
            offenders = (np.flatnonzero(np.diff(ts) <= 0) + 1)[:rows]
        else:
            offenders = np.array([violation.first_row])
        for index in offenders:
            row = {k: v[0] for k, v in table.slice(int(index), 1).to_pydict().items()}
            typer.echo(f"  [{index}] offset={int(ts[index]) % step} {row}")


@raw_app.command("verify")
def raw_verify(
    sample: Annotated[int, typer.Option("--sample", help="Series to re-derive.")] = 10,
    seed: Annotated[
        int, typer.Option("--seed", help="Sampling seed. Same seed, same picks.")
    ] = 1337,
    dest: Annotated[
        Path | None, typer.Option("--dest", help="Verify a local tier instead.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the report here as well.")
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency")] = 12,
) -> None:
    """Re-derive a sample of series from the archives their manifests name and compare the bytes.

    Exits non-zero if any sampled series fails to reproduce. A series that has gained days since
    it was pulled still reproduces its recorded bytes and is reported as drift, not failure.
    """
    from axiom.raw.qa import sample_tasks, stats_markdown, verify_series
    from axiom.sources.binance_vision import BinanceVision

    setup_logging()
    store = _raw_store(dest)
    manifests = store.list_manifests()
    if not manifests:
        typer.echo("the raw tier is empty; nothing to verify", err=True)
        raise typer.Exit(2)

    tasks = sample_tasks(manifests, sample, seed)
    with BinanceVision(concurrency=concurrency) as client:
        results = [verify_series(client, store, task) for task in tasks]

    for result in results:
        typer.echo(result.line())
    identical = sum(1 for r in results if r.byte_identical)
    typer.echo(f"\n{identical}/{len(results)} byte-identical, seed={seed}")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(stats_markdown(manifests, results), encoding="utf-8")
        typer.echo(f"wrote {out}")

    if any(not r.ok for r in results):
        raise typer.Exit(1)


@raw_app.command("stats")
def raw_stats(
    dest: Annotated[Path | None, typer.Option("--dest", help="Read a local tier instead.")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the markdown report here.")
    ] = None,
) -> None:
    """Summarize the raw tier from its sidecars: counts, history, gaps, and the gate."""
    from axiom.raw.qa import stats_markdown

    setup_logging()
    manifests = _raw_store(dest).list_manifests()
    if not manifests:
        typer.echo("the raw tier is empty", err=True)
        raise typer.Exit(2)

    report = stats_markdown(manifests)
    typer.echo(report)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        typer.echo(f"wrote {out}")


@pull_app.command("yahoo-events")
def pull_yahoo_events(
    tickers: Annotated[
        str, typer.Option("--tickers", help="Ticker list YAML, or a packaged config name.")
    ] = "yahoo_events_v1",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Smoke runs only. Tickers, in file order.")
    ] = None,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Pin the run's date, YYYY-MM-DD.")
    ] = None,
    dest: Annotated[
        Path | None, typer.Option("--dest", help="Write to a directory instead of the Hub.")
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Capture split and dividend events for the pinned cross-check population.

    Non-load-bearing by design. Yahoo has no licence and an active habit of refusing datacenter
    IPs, so partial success is success and total failure is a dated line in the audit report
    rather than a blocker (ADR-0016). Paced to at most 300 requests an hour with jitter.
    """
    from datetime import UTC, datetime

    from axiom.sources.yahoo_events import blocked_report, load_tickers, pull_events

    setup_logging()
    pinned = load_tickers(tickers)
    if limit is not None:
        pinned = pinned[:limit]
    stamp = as_of or datetime.now(UTC).date().isoformat()
    pull_run_id = run_id or f"yahoo-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"

    run = pull_events(pinned, _raw_store(dest), pull_run_id=pull_run_id, as_of=stamp, force=force)

    typer.echo(f"{pull_run_id}: {run.line()} of {len(pinned)} ticker(s), as_of={stamp}")
    for failure in run.failures[:10]:
        typer.echo(f"  FAIL {failure.symbol}: {failure.error}")
    if run.ok == 0 and run.failed:
        # Not an error exit. "Yahoo said no" is a documented outcome of this phase, and failing
        # the command would make a green pipeline impossible for a reason nobody can fix.
        typer.echo("")
        typer.echo(blocked_report(run, as_of=stamp))


@raw_app.command("audit-adjustments")
def raw_audit_adjustments(
    out: Annotated[Path, typer.Option("--out", help="Where to write the audit report.")] = Path(
        "docs/reports/v0.2-adjustment-audit.md"
    ),
    sample: Annotated[int, typer.Option("--sample", help="Tickers for the cross-check half.")] = 25,
    years: Annotated[int, typer.Option("--years")] = 2,
    seed: Annotated[int, typer.Option("--seed")] = 1337,
    skip_crosscheck: Annotated[
        bool,
        typer.Option("--skip-crosscheck", help="Split probes only; claim nothing about dividends."),
    ] = False,
    dest: Annotated[Path | None, typer.Option("--dest", help="Read a local tier instead.")] = None,
) -> None:
    """Measure what Stooq already did to these prices, and write the verdict down.

    Two halves of deliberately different fragility. The split probes need the stored bars and a
    calendar, so they answer even when Yahoo is unavailable. The dividend classification needs
    the cross-check, and is reported as unestablished rather than guessed when it cannot run —
    a corpus can know it is split-adjusted while staying honestly unsure about dividends.
    """
    from datetime import UTC, datetime

    from axiom.raw.adjustments import audit_markdown, classify, run_split_probes
    from axiom.raw.crosscheck import crosscheck_equities, crosscheck_markdown

    setup_logging()
    store = _raw_store(dest)
    manifests = store.list_manifests()
    if not any(m.source == "stooq" for m in manifests):
        typer.echo("no Stooq series in the raw tier; nothing to audit", err=True)
        raise typer.Exit(2)

    as_of = datetime.now(UTC).date().isoformat()
    probes = run_split_probes(store, manifests)
    for result in probes:
        typer.echo(result.line())

    comparisons = []
    section = ""
    if not skip_crosscheck:
        from axiom.sources.yahoo_events import live_price_fetcher

        comparisons = crosscheck_equities(
            store, manifests, live_price_fetcher(), sample=sample, years=years, seed=seed
        )
        section = crosscheck_markdown(comparisons, as_of=as_of)

    policy, reasoning = classify(probes, comparisons)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        audit_markdown(
            probes, comparisons, policy, reasoning, as_of=as_of, crosscheck_section=section
        ),
        encoding="utf-8",
    )

    typer.echo("")
    typer.echo(f"verdict: {policy}")
    typer.echo(reasoning)
    typer.echo(f"wrote {out}")


@raw_app.command("crosscheck-equities")
def raw_crosscheck_equities(
    sample: Annotated[int, typer.Option("--sample", help="Tickers to compare.")] = 25,
    seed: Annotated[int, typer.Option("--seed")] = 1337,
    years: Annotated[int, typer.Option("--years", help="Lookback window.")] = 2,
    dest: Annotated[Path | None, typer.Option("--dest", help="Read a local tier instead.")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the markdown section here.")
    ] = None,
) -> None:
    """Compare stored Stooq closes against Yahoo's adjusted closes on a sample of tickers.

    The output is a number and its interpretation, not a pass or fail. Two vendors differing by
    a fraction of a percent over two years is normal; one ticker differing by forty percent means
    a corporate action only one of them applied, and that is the finding.
    """
    from datetime import UTC, datetime

    from axiom.raw.crosscheck import crosscheck_equities, crosscheck_markdown
    from axiom.sources.yahoo_events import live_price_fetcher

    setup_logging()
    store = _raw_store(dest)
    manifests = store.list_manifests()
    if not any(m.source == "stooq" for m in manifests):
        typer.echo("no Stooq series in the raw tier; nothing to cross-check", err=True)
        raise typer.Exit(2)

    results = crosscheck_equities(
        store, manifests, live_price_fetcher(), sample=sample, years=years, seed=seed
    )
    for result in results:
        typer.echo(result.line())

    section = crosscheck_markdown(results, as_of=datetime.now(UTC).date().isoformat())
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(section, encoding="utf-8")
        typer.echo(f"wrote {out}")


@registry_app.command("build")
def registry_build(
    dest: Annotated[
        Path | None,
        typer.Option("--dest", help="Build over a local raw tier and write the result there."),
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="Simultaneous sidecar downloads.")
    ] = 16,
) -> None:
    """Reduce every sidecar in `axiom-raw` to one queryable table, and upload it back.

    Idempotent: rebuilding from an unchanged tier reproduces the same `registry_hash`. The
    sidecars stay the truth — this is a cache with no authority, built so that questions are
    cheap rather than so that facts live in two places.

    A sidecar that cannot be read is reported into `registry/bad_sidecars.json` and the command
    exits non-zero. A registry that silently omits what it could not parse is worse than none,
    because the omission looks exactly like absence.
    """
    from axiom.config.settings import AxiomSettings
    from axiom.raw.store import LocalRawStore
    from axiom.registry import (
        REGISTRY_PATH,
        SUMMARY_PATH,
        build_from_manifests,
        build_registry,
        summary_markdown,
        write_registry_parquet,
    )
    from axiom.registry.build import bad_sidecars_json

    setup_logging()
    if dest is not None:
        store = LocalRawStore(dest)
        build = build_from_manifests(store.list_manifests())
    else:
        from huggingface_hub import HfApi

        settings = AxiomSettings()
        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        build = build_registry(
            HfApi(token=token), settings.raw_repo_id, token=token, concurrency=concurrency
        )

    parquet = write_registry_parquet(build.table, registry_hash_value=build.registry_hash)
    summary = summary_markdown(
        build.table, registry_hash=build.registry_hash, bad_count=len(build.bad)
    )

    if dest is not None:
        for path, payload in (
            (REGISTRY_PATH, parquet),
            (SUMMARY_PATH, summary.encode("utf-8")),
        ):
            target = Path(dest) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    else:
        store = _raw_store(None)
        store.upload_bytes(REGISTRY_PATH, parquet)
        store.upload_json(SUMMARY_PATH, summary)
        if build.bad:
            store.upload_json("registry/bad_sidecars.json", bad_sidecars_json(build.bad))

    typer.echo(summary)
    typer.echo(f"registry_hash: {build.registry_hash} ({build.table.num_rows} artifacts)")
    for entry in build.bad:
        typer.echo(f"  UNREADABLE {entry.path}: {entry.error}", err=True)
    if build.bad:
        raise typer.Exit(1)


@registry_app.command("query")
def registry_query(
    sql: Annotated[
        str,
        typer.Argument(help="SQL over the registry, which is available as the table `registry`."),
    ],
    dest: Annotated[
        Path | None, typer.Option("--dest", help="Query a local registry instead.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Rows to print.")] = 50,
) -> None:
    """Run arbitrary SQL over the registry.

    The canned reports in `axiom registry build` cover what/from-where/pulled-when. This is for
    everything else — needs the `query` extra (`uv sync --extra query`).
    """
    from axiom.registry import REGISTRY_PATH, read_registry

    setup_logging()
    try:
        import duckdb
    except ImportError:
        typer.echo("duckdb is not installed; run `uv sync --extra query`", err=True)
        raise typer.Exit(2) from None

    if dest is not None:
        data = (Path(dest) / REGISTRY_PATH).read_bytes()
    else:
        data = _raw_store(None).get(REGISTRY_PATH)
        if data is None:
            typer.echo("no registry in the raw tier; run `axiom registry build`", err=True)
            raise typer.Exit(2)

    # DuckDB resolves a bare table name against Arrow objects in the caller's scope, so the
    # registry is queryable as `registry` without registering or copying anything.
    registry = read_registry(data)  # noqa: F841
    typer.echo(duckdb.sql(sql).limit(limit))


if (
    __name__ == "__main__"
):  # `python -m axiom.cli`, for kernels where the console script is not on PATH
    app()
