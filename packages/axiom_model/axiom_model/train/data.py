"""Training windows from the segment index — the replacement for upstream's CSV dataset.

Upstream's `CustomKlineDataset` splits one CSV by row ratios and normalizes with
whatever statistics fall in the window. Here windows are offsets into the gap-free,
split-contained segments that `axiom-data build` indexed, and normalization is
`axiom_data.normalization.normalize_window` — context statistics only, the same
call eval and inference make. Nothing in this module may re-derive splits or stats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from axiom_data import datasets, normalization, store
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, Subset

from .config import DATASETS_DIR


class WindowDataset(Dataset):
    """All `context + horizon`-bar windows of one split/timeframe, in memory.

    Samples are `(x, x_stamp)` like upstream: `x` normalized OHLCVA
    `(sample_bars, 6)`, `x_stamp` raw calendar features `(sample_bars, 5)`.
    """

    def __init__(
        self,
        data_cfg: datasets.DataConfig,
        split: str,
        tf: str,
        context_bars: int,
        horizon_bars: int,
        root: Path = store.DEFAULT_ROOT,
        datasets_dir: Path = DATASETS_DIR,
    ):
        self.context_bars = int(context_bars)
        self.sample_bars = int(context_bars) + int(horizon_bars)
        index = pd.read_parquet(Path(datasets_dir) / data_cfg.name / "segments.parquet")
        index = index[(index.split == split) & (index.tf == tf)]
        if index.empty:
            raise ValueError(f"no segments for split={split!r} tf={tf!r} in {data_cfg.name}")

        self.features: list[np.ndarray] = []
        self.stamps: list[np.ndarray] = []
        counts: list[int] = []
        cache: dict[str, pd.DataFrame] = {}
        for row in index.itertuples():
            if row.symbol not in cache:
                cache[row.symbol] = store.read(row.symbol, tf, root=root, venue=data_cfg.venue)
            bars = cache[row.symbol]
            seg = bars[(bars.ts >= row.start_ts) & (bars.ts <= row.end_ts)]
            if len(seg) != row.bars:
                raise ValueError(
                    f"{row.symbol}/{tf} segment at {row.start_ts}: index says {row.bars} bars, "
                    f"corpus has {len(seg)} — rebuild the dataset"
                )
            self.features.append(seg[normalization.FEATURES].to_numpy("float32"))
            self.stamps.append(normalization.time_features(seg.ts).to_numpy("float32"))
            counts.append(max(0, len(seg) - self.sample_bars + 1))
        self._cum = np.cumsum(counts)
        if self._cum[-1] == 0:
            raise ValueError(f"split={split!r} tf={tf!r}: no segment holds {self.sample_bars} bars")

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seg = int(np.searchsorted(self._cum, idx, side="right"))
        off = idx - (0 if seg == 0 else int(self._cum[seg - 1]))
        x = self.features[seg][off : off + self.sample_bars]
        x_norm, _, _ = normalization.normalize_window(x, self.context_bars)
        stamp = self.stamps[seg][off : off + self.sample_bars]
        return torch.from_numpy(np.ascontiguousarray(x_norm)), torch.from_numpy(stamp.copy())


def build_dataset(cfg, data_cfg, split: str, root: Path, datasets_dir: Path) -> Dataset:
    """One dataset over every configured timeframe of `split`."""
    parts = [
        WindowDataset(
            data_cfg, split, tf, cfg.window["context_bars"], cfg.window["horizon_bars"],
            root=root, datasets_dir=datasets_dir,
        )
        for tf in cfg.window["timeframes"]
    ]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def fit_loader(dataset: Dataset, cfg, epoch: int) -> DataLoader:
    """A fresh, deterministically seeded subsample every epoch (the qlib-pipeline
    `n_train_iter` idea: a true epoch over ~10^6 windows is far too long)."""
    n = min(int(cfg.loader.get("train_samples_per_epoch", 64000)), len(dataset))
    generator = torch.Generator().manual_seed(cfg.seed * 1000003 + epoch)
    sampler = RandomSampler(dataset, num_samples=n, generator=generator)
    return DataLoader(
        dataset,
        batch_size=int(cfg.loader.get("batch_size", 32)),
        sampler=sampler,
        num_workers=int(cfg.loader.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def select_loader(dataset: Dataset, cfg) -> DataLoader:
    """A fixed, evenly spaced subset — the same windows every epoch, so the
    early-stopping loss is comparable across epochs and runs."""
    n = min(int(cfg.loader.get("val_samples", 12800)), len(dataset))
    picks = np.unique(np.linspace(0, len(dataset) - 1, n).astype(np.int64))
    return DataLoader(
        Subset(dataset, picks.tolist()),
        batch_size=int(cfg.loader.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg.loader.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


__all__ = ["WindowDataset", "build_dataset", "fit_loader", "select_loader"]
