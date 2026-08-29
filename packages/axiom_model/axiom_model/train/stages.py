"""The two training loops, ported from `vendor/kronos/finetune_csv/` (upstream 67b630e).

Losses, clipping norms, optimizer settings and the OneCycle schedule are upstream's,
unchanged — first runs must be attributable to data, not to a rewritten recipe.
Differences: no DDP (single GPU everywhere through M1), optional bf16 autocast,
metrics go to the caller's W&B run instead of Comet, prints flush (B-10).
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data import fit_loader, select_loader


def _autocast(device: str, precision: str):
    if precision == "bf16" and device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _scheduler(optimizer, lr: float, steps_per_epoch: int, epochs: int):
    # Upstream's schedule for both stages.
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs,
        pct_start=0.03, div_factor=10,
    )


def _save_best(model, save_dir: Path) -> Path:
    dest = Path(save_dir) / "best_model"
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    return dest


def fit_tokenizer(
    tokenizer,
    fit_ds: Dataset,
    select_ds: Dataset,
    cfg,
    device: str,
    save_dir: Path,
    wandb_run=None,
) -> dict:
    """Stage A: reconstruction + BSQ loss on the tokenizer, best-val checkpointing."""
    stage = cfg.stage_a
    epochs = int(stage["epochs"])
    accumulation = int(stage.get("accumulation_steps", 1))
    tokenizer = tokenizer.to(device)
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(), lr=float(stage["lr"]),
        weight_decay=float(stage.get("weight_decay", 0.1)),
    )
    steps_per_epoch = len(fit_loader(fit_ds, cfg, epoch=0))
    scheduler = _scheduler(optimizer, float(stage["lr"]), steps_per_epoch, epochs)
    val_loader = select_loader(select_ds, cfg)

    best, history, step = float("inf"), [], 0
    for epoch in range(epochs):
        t0 = time.time()
        tokenizer.train()
        train_loss_sum, train_batches = 0.0, 0
        for batch_x, _ in fit_loader(fit_ds, cfg, epoch):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_loss = 0.0
            for j in range(accumulation):
                lo = j * (batch_x.shape[0] // accumulation)
                hi = (j + 1) * (batch_x.shape[0] // accumulation)
                x = batch_x[lo:hi]
                with _autocast(device, cfg.precision):
                    zs, bsq_loss, _, _ = tokenizer(x)
                    z_pre, z = zs
                    recon = F.mse_loss(z_pre, x) + F.mse_loss(z, x)
                    loss = (recon + bsq_loss) / 2
                (loss / accumulation).backward()
                batch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(
                tokenizer.parameters(), max_norm=float(stage.get("grad_clip", 2.0))
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss_sum += batch_loss / accumulation
            train_batches += 1
            step += 1
            _log_step(cfg, wandb_run, "stage_a", step, epoch, optimizer, batch_loss / accumulation)

        tokenizer.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                with _autocast(device, cfg.precision):
                    zs, _, _, _ = tokenizer(batch_x)
                val_sum += F.mse_loss(zs[1].float(), batch_x).item() * batch_x.size(0)
                val_count += batch_x.size(0)
        best, entry = _end_epoch(
            "stage_a", tokenizer, epoch, epochs, train_loss_sum / max(train_batches, 1),
            val_sum / max(val_count, 1), best, save_dir, t0, wandb_run,
        )
        history.append(entry)
    return {"best_val_loss": best, "history": history, "path": str(Path(save_dir) / "best_model")}


def fit_predictor(
    model,
    tokenizer,
    fit_ds: Dataset,
    select_ds: Dataset,
    cfg,
    device: str,
    save_dir: Path,
    wandb_run=None,
) -> dict:
    """Stage B: next-token cross-entropy on the predictor, tokens from a frozen
    tokenizer encoded on the fly (upstream behaviour)."""
    stage = cfg.stage_b
    epochs = int(stage["epochs"])
    model = model.to(device)
    tokenizer = tokenizer.to(device).eval()
    beta1, beta2 = stage.get("betas", [0.9, 0.95])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(stage["lr"]), betas=(float(beta1), float(beta2)),
        weight_decay=float(stage.get("weight_decay", 0.1)),
    )
    steps_per_epoch = len(fit_loader(fit_ds, cfg, epoch=0))
    scheduler = _scheduler(optimizer, float(stage["lr"]), steps_per_epoch, epochs)
    val_loader = select_loader(select_ds, cfg)

    def token_loss(batch_x, batch_stamp):
        with torch.no_grad():
            s1, s2 = tokenizer.encode(batch_x, half=True)
        logits = model(s1[:, :-1], s2[:, :-1], batch_stamp[:, :-1, :])
        loss, _, _ = model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])
        return loss

    best, history, step = float("inf"), [], 0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for batch_x, batch_stamp in fit_loader(fit_ds, cfg, epoch):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_stamp = batch_stamp.to(device, non_blocking=True)
            with _autocast(device, cfg.precision):
                loss = token_loss(batch_x, batch_stamp)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(stage.get("grad_clip", 3.0))
            )
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_batches += 1
            step += 1
            _log_step(cfg, wandb_run, "stage_b", step, epoch, optimizer, loss.item())

        model.eval()
        val_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_stamp = batch_stamp.to(device, non_blocking=True)
                with _autocast(device, cfg.precision):
                    val_sum += token_loss(batch_x, batch_stamp).item()
                val_batches += 1
        best, entry = _end_epoch(
            "stage_b", model, epoch, epochs, train_loss_sum / max(train_batches, 1),
            val_sum / max(val_batches, 1), best, save_dir, t0, wandb_run,
        )
        history.append(entry)
    return {"best_val_loss": best, "history": history, "path": str(Path(save_dir) / "best_model")}


def _log_step(cfg, wandb_run, stage: str, step: int, epoch: int, optimizer, loss: float) -> None:
    if step % cfg.log_interval != 0:
        return
    lr = optimizer.param_groups[0]["lr"]
    print(f"  [{stage} epoch {epoch + 1} step {step}] lr {lr:.2e}  loss {loss:.4f}", flush=True)
    if wandb_run is not None:
        wandb_run.log({f"{stage}/train_loss": loss, f"{stage}/lr": lr}, step=step)


def _end_epoch(
    stage, model, epoch, epochs, train_loss, val_loss, best, save_dir, t0, wandb_run
) -> tuple[float, dict]:
    print(
        f"  [{stage} epoch {epoch + 1}/{epochs}] train {train_loss:.4f}  "
        f"val {val_loss:.4f}  {time.time() - t0:.1f}s",
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.log({f"{stage}/epoch": epoch + 1, f"{stage}/val_loss": val_loss})
    if val_loss < best:
        best = val_loss
        dest = _save_best(model, save_dir)
        print(f"  [{stage}] best so far — saved to {dest}", flush=True)
    return best, {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}


__all__ = ["fit_predictor", "fit_tokenizer"]
