"""`axiom-eval` CLI (P2-10).

    axiom-eval run --config configs/eval/default.yaml
    axiom-eval run --config configs/eval/default.yaml \
        --models persistence ewma --timeframes 1h --max-anchors 4   # laptop smoke

The overrides exist for smoke runs and for the cross-machine check (P2-13). A run
that is meant to be *compared* to another one uses the config unchanged — that is
the whole point of freezing it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axiom_data import store

from .run import DATASETS_DIR, run


def cmd_run(args: argparse.Namespace) -> int:
    run(
        args.config,
        root=Path(args.root),
        out_dir=Path(args.out) if args.out else None,
        datasets_dir=Path(args.datasets),
        device=args.device,
        models=args.models,
        timeframes=args.timeframes,
        symbols=args.symbols,
        max_anchors=args.max_anchors,
        use_wandb=False if args.no_wandb else None,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="axiom-eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="score models + baselines and write a report")
    r.add_argument("--config", default="configs/eval/default.yaml")
    r.add_argument("--root", default=str(store.DEFAULT_ROOT))
    r.add_argument("--datasets", default=str(DATASETS_DIR), help="where the manifest lives")
    r.add_argument("--out", help="report directory (default: the config's report_dir)")
    r.add_argument("--device", help="cpu / cuda:0 (default: cuda when present)")
    r.add_argument("--models", nargs="*", help="subset of models+baselines to score")
    r.add_argument("--timeframes", nargs="*", help="subset of the config's timeframes")
    r.add_argument("--symbols", nargs="*", help="subset of the universe (smoke runs only)")
    r.add_argument("--max-anchors", type=int, help="override anchors per timeframe")
    r.add_argument("--no-wandb", action="store_true")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
