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
app.add_typer(config_app, name="config")
app.add_typer(loop_app, name="loop")


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


if (
    __name__ == "__main__"
):  # `python -m axiom.cli`, for kernels where the console script is not on PATH
    app()
