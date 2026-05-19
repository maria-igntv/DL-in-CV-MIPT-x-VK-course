"""Shared utilities: device, seeds, metrics, loss functions, classical baselines."""

import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Device & reproducibility
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_with_fallback():
    """MPS не поддерживает grid_sampler_3d_backward (нужен для 3D LUT).
    Включает PYTORCH_ENABLE_MPS_FALLBACK=1 и возвращает cpu для обучения."""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return torch.device("cpu")


def fix_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ssim_psnr(pred_np, gt_np):
    """pred_np, gt_np: (H, W, 3) uint8 arrays."""
    s = ssim(pred_np, gt_np, channel_axis=2)
    p = psnr(pred_np, gt_np)
    return s, p


def setup_lpips(device):
    """LPIPS with AlexNet weights compatibility for newer torchvision."""
    import lpips

    try:
        _real_alexnet = sys.modules["torchvision.models.alexnet"].alexnet
        from torchvision.models import AlexNet_Weights

        def _compat_alexnet(pretrained=False, **kw):
            if pretrained:
                return _real_alexnet(weights=AlexNet_Weights.IMAGENET1K_V1, **kw)
            return _real_alexnet(**kw)

        import torchvision.models as _tv_models

        _tv_models.alexnet = _compat_alexnet
    except Exception:
        pass

    return lpips.LPIPS(net="alex").to(device)


def measure_inference_time(model, device, n_warmup=10, n_runs=100, img_size=512):
    """Measure mean/median/P95 inference time."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    for _ in range(n_warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.time()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)

    arr = np.array(times)
    return {
        "mean_ms": round(arr.mean(), 2),
        "median_ms": round(np.median(arr), 2),
        "p95_ms": round(np.percentile(arr, 95), 2),
        "fps": round(1000 / arr.mean(), 1),
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """LPIPS-based perceptual loss."""

    def __init__(self, device):
        super().__init__()
        self.lpips_fn = setup_lpips(device)

    def forward(self, pred, target):
        return self.lpips_fn(pred * 2 - 1, target * 2 - 1).mean()


class CombinedLoss(nn.Module):
    """L1 + lambda * LPIPS perceptual loss."""

    def __init__(self, device, lambda_lpips: float = 0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss(device)
        self.lambda_lpips = lambda_lpips

    def forward(self, pred, target):
        return self.l1(pred, target) + self.lambda_lpips * self.perceptual(
            pred, target
        )


# ---------------------------------------------------------------------------
# Classical baseline methods
# ---------------------------------------------------------------------------

class ClassicalMethods:
    @staticmethod
    def clahe(img_bgr, clip_limit=2.0):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def white_balance(img_bgr):
        result = img_bgr.astype(float)
        avg = result.mean()
        for ch in range(3):
            result[:, :, ch] *= avg / (result[:, :, ch].mean() + 1e-6)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def auto_gamma(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()
        if mean_val < 1:
            return img_bgr
        gamma = np.log(127 / 255) / np.log(mean_val / 255)
        gamma = np.clip(gamma, 0.3, 3.0)
        inv_gamma = 1.0 / gamma
        table = (
            np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)])
            .astype("uint8")
        )
        return cv2.LUT(img_bgr, table)

    @staticmethod
    def pipeline(img_bgr):
        x = ClassicalMethods.white_balance(img_bgr)
        x = ClassicalMethods.clahe(x, clip_limit=1.5)
        return x
