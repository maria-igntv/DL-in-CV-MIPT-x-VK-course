"""MIT-Adobe FiveK dataset with optional augmentations."""

import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FiveKDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        target_dir: str,
        size: int = 512,
        split: str = "train",
        augment: bool = False,
    ):
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.size = size
        self.augment = augment

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

        if self.augment:
            raw, target = self._augment(raw, target)

        raw = torch.from_numpy(raw).permute(2, 0, 1)
        target = torch.from_numpy(target).permute(2, 0, 1)
        return raw, target

    @staticmethod
    def _augment(raw: np.ndarray, target: np.ndarray):
        # Horizontal flip (50%)
        if random.random() < 0.5:
            raw = raw[:, ::-1, :]
            target = target[:, ::-1, :]

        # Brightness jitter (same factor for both)
        brightness_factor = random.uniform(0.9, 1.1)
        raw = np.clip(raw * brightness_factor, 0, 1)
        target = np.clip(target * brightness_factor, 0, 1)

        # Contrast jitter (per-channel, same transform for both)
        contrast_factor = random.uniform(0.9, 1.1)
        for ch in range(3):
            raw[:, :, ch] = np.clip(
                0.5 + (raw[:, :, ch] - 0.5) * contrast_factor, 0, 1
            )
            target[:, :, ch] = np.clip(
                0.5 + (target[:, :, ch] - 0.5) * contrast_factor, 0, 1
            )

        return raw, target
