"""`axiom-train` CLI (P3-02).

    axiom-train --config configs/finetune/crypto_v0.yaml               # both stages
    axiom-train --config configs/finetune/crypto_v0.yaml --stage a
    axiom-train --config configs/finetune/crypto_v0.yaml --no-wandb    # smoke only

A run meant to be *compared* to another uses the committed config unchanged —
overrides exist for smoke tests, not experiments (one change per run = one YAML).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axiom_data import store

from .config import DATASETS_DIR
from .finetune import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="axiom-train", description=__doc__)
    p.add_argument("--config", default="configs/finetune/crypto_v0.yaml")
    p.add_argument("--stage", choices=["a", "b", "all"], default="all")
    p.add_argument("--device", help="cpu / cuda:0 (default: cuda when present)")
    p.add_argument("--root", default=str(store.DEFAULT_ROOT))
    p.add_argument("--datasets", default=str(DATASETS_DIR), help="where the manifest lives")
    p.add_argument("--out", help="checkpoint root (default: the config's out_dir)")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)

    run(
        args.config,
        stage=args.stage,
        device=args.device,
        root=Path(args.root),
        datasets_dir=Path(args.datasets),
        out_dir=Path(args.out) if args.out else None,
        use_wandb=False if args.no_wandb else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
