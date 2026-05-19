"""Training script for Learnable3DLUT on MIT-Adobe FiveK (Expert C).

Usage:
    python train.py --data-root /path/to/adobe-fivek --epochs 100 --batch-size 8 --loss combined
    python train.py --data-root /path/to/adobe-fivek --epochs 100 --batch-size 8 --loss l1
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FiveKDataset
from lut_model import Learnable3DLUT
from utils import (
    CombinedLoss,
    PerceptualLoss,
    fix_seed,
    get_device_with_fallback,
)


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
        description="Train 3D LUT image enhancer"
    )
    parser.add_argument("--data-root", default="~/.cache/kagglehub/datasets/weipengzhang/adobe-fivek/versions/1")
    parser.add_argument("--expert", default="c", choices=list("abcde"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lut-dim", type=int, default=17)
    parser.add_argument("--n-luts", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--loss",
        choices=["l1", "perceptual", "combined"],
        default="combined",
    )
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()

    fix_seed(args.seed)
    device = get_device_with_fallback()
    print(f"Device: {device} (MPS fallback enabled for 3D LUT compatibility)")

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

    model = Learnable3DLUT(lut_dim=args.lut_dim, n_luts=args.n_luts).to(
        device
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e3:.1f}K)")

    criterion = build_criterion(args.loss, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
            ckpt_name = "best_lut.pth" if args.loss == "l1" else "best_lut_improved.pth"
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, ckpt_name),
            )
            print(f"  -> saved {ckpt_name} (val={val_loss:.4f})")

    # Save training history
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
