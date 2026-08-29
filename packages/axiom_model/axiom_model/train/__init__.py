"""Fine-tuning (P3-02): upstream `vendor/kronos/finetune_csv/` ported onto Axiom's
data foundation.

What was kept from upstream: the two-stage recipe (Stage A tokenizer, Stage B
predictor), the losses, the optimizer settings, OneCycleLR, gradient clipping.
What was replaced: the CSV loader and its ratio-based splits are gone — windows
come from the dataset builder's segment index (chronological splits, embargo,
gap-free segments) and are normalized by `axiom_data.normalization`, the single
implementation. DDP was dropped: every training machine in the plan through M1
(XTX, Modal A10G/L4/A100) is a single GPU.
"""

from .config import FinetuneConfig, load_config
from .data import WindowDataset
from .finetune import run

__all__ = ["FinetuneConfig", "WindowDataset", "load_config", "run"]
