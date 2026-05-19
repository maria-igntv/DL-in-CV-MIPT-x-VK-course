"""Training script for Learnable3DLUT on MIT-Adobe FiveK (Expert C).

Usage (Colab / local):
    python train.py --data-root /kaggle/input/adobe-fivek --epochs 50 --batch-size 8
"""

import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lut_model import Learnable3DLUT


class FiveKDataset(Dataset):
    def __init__(self, raw_dir: str, target_dir: str, size: int = 512, split: str = "train"):
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.size = size

        all_files = sorted(glob(os.path.join(raw_dir, "*.*")))
        n = len(all_files)
        if split == "train":
            self.files = all_files[: int(n * 0.8)]
        elif split == "val":
            self.files = all_files[int(n * 0.8) : int(n * 0.9)]
        else:
            self.files = all_files[int(n * 0.9) :]

    def __len__(self):
        return len(self.files)

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        img = cv2.resize(img, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        fname = os.path.basename(self.files[idx])
        raw = self._load(self.files[idx])
        target = self._load(os.path.join(self.target_dir, fname))

        raw = torch.from_numpy(raw).permute(2, 0, 1)
        target = torch.from_numpy(target).permute(2, 0, 1)
        return raw, target


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/kaggle/input/adobe-fivek")
    parser.add_argument("--expert", default="c", choices=list("abcde"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lut-dim", type=int, default=17)
    parser.add_argument("--n-luts", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_root, "raw")
    target_dir = os.path.join(args.data_root, args.expert)

    train_ds = FiveKDataset(raw_dir, target_dir, args.size, "train")
    val_ds = FiveKDataset(raw_dir, target_dir, args.size, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Learnable3DLUT(lut_dim=args.lut_dim, n_luts=args.n_luts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs}  train L1={train_loss:.4f}  val L1={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_lut.pth"))
            print(f"  -> saved best model (val L1={val_loss:.4f})")

    print(f"\nTraining complete. Best val L1: {best_val:.4f}")


if __name__ == "__main__":
    main()
