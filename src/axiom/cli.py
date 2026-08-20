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
app.add_typer(config_app, name="config")
app.add_typer(loop_app, name="loop")
app.add_typer(universe_app, name="universe")
app.add_typer(pull_app, name="pull")


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
    typer.echo(f"min_history_days: {universe.criteria.min_history_days}")
    for name, symbols in sorted(universe.symbols.items()):
        typer.echo(f"{name}: {len(symbols)} symbols, first 5 {symbols[:5]}")
    typer.echo(f"universe_hash: {universe.universe_hash}")


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
    path_in_repo = f"manifests/pulls/{pull_run_id}.json"
    if isinstance(store, LocalRawStore):
        target = Path(dest or ".") / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(final.to_json(), encoding="utf-8")
    else:
        store.upload_json(path_in_repo, final.to_json())

    typer.echo(
        f"{pull_run_id}: ok={final.ok} skipped={final.skipped} failed={final.failed} "
        f"rows={final.total_rows} bytes={final.total_bytes}"
        + (" (PARTIAL)" if final.is_partial else "")
    )
    for failure in final.failures:
        typer.echo(f"  FAIL {failure.market}/{failure.frequency}/{failure.symbol}: {failure.error}")
    if final.failed:
        raise typer.Exit(1)


if (
    __name__ == "__main__"
):  # `python -m axiom.cli`, for kernels where the console script is not on PATH
    app()
