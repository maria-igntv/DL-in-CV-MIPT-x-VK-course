"""Unified evaluation script for all image enhancement models.

Evaluates: RAW, classical baselines, 3D LUT (original + improved), U-Net.
Outputs: evaluation_results.json

Usage:
    python evaluate.py --data-root /path/to/adobe-fivek --all
    python evaluate.py --data-root /path/to/adobe-fivek --lut-checkpoint checkpoints/best_lut_improved.pth
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

import torch

from dataset import FiveKDataset
from utils import (
    ClassicalMethods,
    compute_ssim_psnr,
    get_device,
    measure_inference_time,
    setup_lpips,
)


def evaluate_pytorch_model(name, model, test_ds, device, lpips_fn):
    """Evaluate a PyTorch model on the test set."""
    ssim_vals, psnr_vals, lpips_vals = [], [], []
    model.eval()
    with torch.no_grad():
        for raw, target in tqdm(test_ds, desc=f"Eval {name}", leave=False):
            raw_b = raw.unsqueeze(0).to(device)
            pred = model(raw_b).squeeze(0).cpu()
            gt = target.cpu()

            pred_u8 = np.clip(
                pred.numpy().transpose(1, 2, 0) * 255, 0, 255
            ).astype(np.uint8)
            gt_u8 = np.clip(
                gt.numpy().transpose(1, 2, 0) * 255, 0, 255
            ).astype(np.uint8)

            s, p = compute_ssim_psnr(pred_u8, gt_u8)
            l = lpips_fn(
                pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1
            ).item()

            ssim_vals.append(s)
            psnr_vals.append(p)
            lpips_vals.append(l)

    timing = measure_inference_time(model, device)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "ssim_mean": round(float(np.mean(ssim_vals)), 4),
        "ssim_std": round(float(np.std(ssim_vals)), 4),
        "psnr_mean": round(float(np.mean(psnr_vals)), 2),
        "psnr_std": round(float(np.std(psnr_vals)), 2),
        "lpips_mean": round(float(np.mean(lpips_vals)), 4),
        "lpips_std": round(float(np.std(lpips_vals)), 4),
        "n_params": n_params,
        **timing,
    }


def evaluate_classical(name, method_fn, raw_dir, target_dir, test_files):
    """Evaluate a classical method on test files."""
    ssim_vals, psnr_vals = [], []

    for raw_path in tqdm(test_files, desc=f"Eval {name}", leave=False):
        fname = os.path.basename(raw_path)
        raw_bgr = cv2.imread(raw_path)
        expert_bgr = cv2.imread(os.path.join(target_dir, fname))
        if raw_bgr is None or expert_bgr is None:
            continue

        raw_resized = cv2.resize(
            raw_bgr, (expert_bgr.shape[1], expert_bgr.shape[0])
        )
        enhanced = method_fn(raw_resized) if name != "raw" else raw_resized

        e_small = cv2.resize(enhanced, (512, 512))
        g_small = cv2.resize(expert_bgr, (512, 512))

        s, p = compute_ssim_psnr(e_small, g_small)
        ssim_vals.append(s)
        psnr_vals.append(p)

    return {
        "ssim_mean": round(float(np.mean(ssim_vals)), 4),
        "ssim_std": round(float(np.std(ssim_vals)), 4),
        "psnr_mean": round(float(np.mean(psnr_vals)), 2),
        "psnr_std": round(float(np.std(psnr_vals)), 2),
        "lpips_mean": None,
        "lpips_std": None,
        "n_params": 0,
        "mean_ms": 0,
        "median_ms": 0,
        "p95_ms": 0,
        "fps": float("inf"),
    }


def measure_classical_time(name, method_fn, size=512):
    """Measure inference time of a classical method."""
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    for _ in range(10):
        _ = method_fn(img)
    times = []
    for _ in range(100):
        t0 = time.time()
        _ = method_fn(img)
        times.append((time.time() - t0) * 1000)
    arr = np.array(times)
    return {
        "mean_ms": round(arr.mean(), 2),
        "median_ms": round(np.median(arr), 2),
        "p95_ms": round(np.percentile(arr, 95), 2),
        "fps": round(1000 / arr.mean(), 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all image enhancement models"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expert", default="c")
    parser.add_argument("--lut-checkpoint", default=None)
    parser.add_argument("--lut-original", default=None)
    parser.add_argument("--unet-checkpoint", default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device: {device} (CPU required for 3D LUT evaluation)")

    raw_dir = os.path.join(args.data_root, "raw")
    target_dir = os.path.join(args.data_root, args.expert)
    test_files = sorted(glob(os.path.join(raw_dir, "*.*")))
    n = len(test_files)
    test_files = test_files[int(n * 0.9) :]
    print(f"Test files: {len(test_files)}")

    test_ds = FiveKDataset(raw_dir, target_dir, args.size, "test")
    lpips_fn = setup_lpips(device)

    results = {}

    # RAW baseline
    print("\n--- RAW (no processing) ---")
    results["raw"] = evaluate_classical(
        "raw", lambda x: x, raw_dir, target_dir, test_files
    )

    # Classical baselines
    classical = {
        "clahe": ClassicalMethods.clahe,
        "auto_gamma": ClassicalMethods.auto_gamma,
        "white_balance": ClassicalMethods.white_balance,
        "pipeline": ClassicalMethods.pipeline,
    }
    for name, fn in classical.items():
        print(f"\n--- {name} ---")
        results[name] = evaluate_classical(name, fn, raw_dir, target_dir, test_files)
        timing = measure_classical_time(name, fn)
        results[name].update(timing)

    # 3D LUT models
    from lut_model import Learnable3DLUT

    lut_checkpoints = []
    if args.all or args.lut_original:
        lut_checkpoints.append(
            ("3d_lut_original", args.lut_original or os.path.join("checkpoints", "best_lut.pth"))
        )
    if args.all or args.lut_checkpoint:
        lut_checkpoints.append(
            ("3d_lut_improved", args.lut_checkpoint or os.path.join("checkpoints", "best_lut_improved.pth"))
        )

    for name, ckpt_path in lut_checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"\n--- {name}: checkpoint not found ({ckpt_path}), skipping ---")
            continue
        print(f"\n--- {name} ({ckpt_path}) ---")
        model = Learnable3DLUT().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        results[name] = evaluate_pytorch_model(name, model, test_ds, device, lpips_fn)

    # U-Net
    if args.all or args.unet_checkpoint:
        unet_path = args.unet_checkpoint or os.path.join("checkpoints_unet", "best_unet.pth")
        if os.path.exists(unet_path):
            print(f"\n--- unet_mobilenetv2 ({unet_path}) ---")
            from unet_model import UNetEnhancer
            model = UNetEnhancer(pretrained=False).to(device)
            model.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
            results["unet_mobilenetv2"] = evaluate_pytorch_model(
                "unet_mobilenetv2", model, test_ds, device, lpips_fn
            )
        else:
            print(f"\n--- unet: checkpoint not found ({unet_path}), skipping ---")

    # Print summary table
    print("\n" + "=" * 100)
    print(
        f"{'Method':<25} {'SSIM':>12} {'PSNR (dB)':>12} {'LPIPS':>12} "
        f"{'Params':>10} {'Latency':>10} {'FPS':>8}"
    )
    print("-" * 100)
    for name, r in results.items():
        ssim_str = f"{r['ssim_mean']:.4f}+/-{r['ssim_std']:.4f}" if r['ssim_mean'] else "N/A"
        psnr_str = f"{r['psnr_mean']:.2f}+/-{r['psnr_std']:.2f}" if r['psnr_mean'] else "N/A"
        lpips_str = f"{r['lpips_mean']:.4f}+/-{r['lpips_std']:.4f}" if r['lpips_mean'] else "N/A"
        params_str = f"{r['n_params']:,}" if r['n_params'] else "0"
        lat_str = f"{r['mean_ms']:.1f}" if r['mean_ms'] else "0"
        fps_str = f"{r['fps']:.1f}" if r['fps'] else "inf"
        print(
            f"{name:<25} {ssim_str:>12} {psnr_str:>12} {lpips_str:>12} "
            f"{params_str:>10} {lat_str:>9}ms {fps_str:>7}"
        )
    print("=" * 100)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
