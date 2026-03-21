"""Training pipeline for FinzMomentumModel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import FinzNewsDataset
from src.models.momentum_model import FinzMomentumModel


def get_device(cfg: dict) -> torch.device:
    preference = cfg["training"]["device"]
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def build_scheduler(optimizer, cfg: dict, total_steps: int):
    sched_type = cfg["training"]["scheduler"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    if sched_type == "none":
        return None

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

    if sched_type == "cosine":
        decay = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    else:
        decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps - warmup_steps)

    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])


def train_epoch(
    model: FinzMomentumModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    num_horizons: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numerical = batch["numerical"].to(device)
        labels = batch["labels"].to(device)       # (B, num_horizons)
        returns = batch["returns"].to(device)      # (B, num_horizons)

        outputs = model(input_ids, attention_mask, numerical)

        loss = torch.tensor(0.0, device=device)
        for i in range(num_horizons):
            cls_loss = cls_criterion(outputs[f"logits_{i}"], labels[:, i])
            reg_loss = reg_criterion(outputs[f"return_{i}"], returns[:, i])
            loss = loss + cls_loss + 0.1 * reg_loss  # classification-weighted

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(
    model: FinzMomentumModel,
    dataloader: DataLoader,
    device: torch.device,
    num_horizons: int,
) -> dict[str, float]:
    model.eval()
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    total_loss = 0.0
    correct = [0] * num_horizons
    total = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numerical = batch["numerical"].to(device)
        labels = batch["labels"].to(device)
        returns = batch["returns"].to(device)

        outputs = model(input_ids, attention_mask, numerical)

        loss = torch.tensor(0.0, device=device)
        for i in range(num_horizons):
            cls_loss = cls_criterion(outputs[f"logits_{i}"], labels[:, i])
            reg_loss = reg_criterion(outputs[f"return_{i}"], returns[:, i])
            loss = loss + cls_loss + 0.1 * reg_loss

            preds = outputs[f"logits_{i}"].argmax(dim=-1)
            correct[i] += (preds == labels[:, i]).sum().item()

        total += labels.size(0)
        total_loss += loss.item()
        n_batches += 1

    metrics = {"val_loss": total_loss / max(n_batches, 1)}
    for i in range(num_horizons):
        metrics[f"acc_horizon_{i}"] = correct[i] / max(total, 1)
    return metrics


def train(cfg: dict | None = None, data_path: str = "data/processed/dataset.parquet"):
    if cfg is None:
        cfg = load_config()

    device = get_device(cfg)
    print(f"Device: {device}")

    # ── Datasets & Loaders ───────────────────────────────────────
    print("Loading datasets...")
    train_ds = FinzNewsDataset(data_path, cfg=cfg, split="train")
    val_ds = FinzNewsDataset(data_path, cfg=cfg, split="val", tokenizer=train_ds.tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Numerical features: {train_ds.num_numerical_features}")

    # ── Model ────────────────────────────────────────────────────
    num_horizons = cfg["model"]["num_horizons"]
    model = FinzMomentumModel(
        num_numerical_features=train_ds.num_numerical_features,
        text_model_name=cfg["text_encoder"]["model_name"],
        text_embed_dim=cfg["model"]["text_embed_dim"],
        numerical_hidden=cfg["model"]["numerical_hidden"],
        fusion_hidden=cfg["model"]["fusion_hidden"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
        num_horizons=num_horizons,
        freeze_text_layers=cfg["text_encoder"]["freeze_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Optimizer & Scheduler ────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    # ── Training Loop ────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg["training"]["patience"]

    ckpt_dir = Path("models/checkpoints")
    best_dir = Path("models/best")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\n═══ Epoch {epoch}/{cfg['training']['epochs']} ═══")

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, num_horizons)
        val_metrics = evaluate(model, val_loader, device, num_horizons)

        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
        for i in range(num_horizons):
            horizon = cfg["data"]["horizons"][i]
            print(f"  Acc {horizon}d:   {val_metrics[f'acc_horizon_{i}']:.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["val_loss"],
            "config": cfg,
            "num_numerical_features": train_ds.num_numerical_features,
        }, ckpt_dir / f"epoch_{epoch}.pt")

        # Best model & early stopping
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "config": cfg,
                "num_numerical_features": train_ds.num_numerical_features,
            }, best_dir / "model.pt")
            print(f"  ★ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement.")
                break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return best_dir / "model.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FinzMomentumModel")
    parser.add_argument("--config", type=str, default=None, help="Override config YAML path")
    parser.add_argument("--data", type=str, default="data/processed/dataset.parquet", help="Dataset parquet path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["training"]["seed"])
    train(cfg=cfg, data_path=args.data)
