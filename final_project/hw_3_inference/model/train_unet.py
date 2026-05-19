"""Training script for U-Net enhancer on MIT-Adobe FiveK (Expert C).

Usage:
    python train_unet.py --data-root /path/to/adobe-fivek --epochs 30 --batch-size 4
    python train_unet.py --data-root /path/to/adobe-fivek --size 256 --epochs 20 --batch-size 8
"""

import argparse
import json
import os
import ssl
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FiveKDataset
from unet_model import UNetEnhancer
from utils import CombinedLoss, PerceptualLoss, fix_seed, get_device


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for raw, target in tqdm(loader, desc="train", leave=False):
        raw, target = raw.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(raw)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * raw.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for raw, target in tqdm(loader, desc="val", leave=False):
        raw, target = raw.to(device), target.to(device)
        pred = model(raw)
        total_loss += criterion(pred, target).item() * raw.size(0)
    return total_loss / len(loader.dataset)


def build_criterion(loss_type, device):
    if loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "perceptual":
        return PerceptualLoss(device)
    elif loss_type == "combined":
        return CombinedLoss(device)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net image enhancer"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expert", default="c")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--loss",
        choices=["l1", "perceptual", "combined"],
        default="combined",
    )
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="checkpoints_unet")
    args = parser.parse_args()

    fix_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Fix SSL for downloading pretrained weights on macOS
    ssl._create_default_https_context = ssl._create_unverified_context

    raw_dir = os.path.join(args.data_root, "raw")
    target_dir = os.path.join(args.data_root, args.expert)

    train_ds = FiveKDataset(
        raw_dir, target_dir, args.size, "train", augment=args.augment
    )
    val_ds = FiveKDataset(raw_dir, target_dir, args.size, "val")

    num_workers = 0 if sys.platform == "darwin" else 2
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(
        f"Train: {len(train_ds)}  Val: {len(val_ds)}  "
        f"num_workers={num_workers}"
    )

    model = UNetEnhancer(pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Freeze encoder initially
    for name, param in model.named_parameters():
        if name.startswith("enc"):
            param.requires_grad = False
    n_frozen = n_params - sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Frozen encoder: {n_frozen:,} params. "
        f"Trainable decoder: {n_trainable:,} params"
    )

    criterion = build_criterion(args.loss, device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    os.makedirs(args.save_dir, exist_ok=True)
    hparams = vars(args)
    hparams["n_params"] = n_params
    hparams["device"] = str(device)
    with open(os.path.join(args.save_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    best_val = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        # Unfreeze encoder after freeze period
        if epoch == args.freeze_encoder_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_encoder_epochs
            )
            print(
                f"  Epoch {epoch}: Unfroze encoder, "
                f"lr={args.lr * 0.1:.1e}, "
                f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
            )

        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.1e}  "
            f"time={elapsed:.0f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "best_unet.pth"),
            )
            print(f"  -> saved best (val={val_loss:.4f})")

    # Save training history
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
