"""Error analysis: categorize model failures by image content.

Categories:
- Brightness: night / twilight / day / overexposed
- Color temperature: natural / warm_artificial / cool_artificial
- Saturation: low / medium / high

Outputs:
- error_analysis_summary.csv
- error_analysis_per_image.csv
- error_analysis_best_worst/ (images)

Usage:
    python analyze_errors.py --data-root /path/to/adobe-fivek --checkpoint checkpoints/best_lut_improved.pth
    python analyze_errors.py --data-root /path/to/adobe-fivek --checkpoint checkpoints/best_lut_improved.pth --model-type unet
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset import FiveKDataset
from utils import compute_ssim_psnr, get_device, setup_lpips


def classify_brightness(img_rgb):
    gray = cv2.cvtColor(
        cv2.cvtColor(
            (img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        ),
        cv2.COLOR_BGR2GRAY,
    )
    mean_b = gray.mean()
    if mean_b < 60:
        return "night"
    elif mean_b < 90:
        return "twilight"
    elif mean_b < 120:
        return "day"
    else:
        return "overexposed"


def classify_color_temp(img_rgb):
    r_mean = img_rgb[:, :, 0].mean()
    b_mean = img_rgb[:, :, 2].mean()
    diff = r_mean - b_mean
    if abs(diff) < 0.02:
        return "natural"
    elif diff > 0.02:
        return "warm_artificial"
    else:
        return "cool_artificial"


def classify_saturation(img_rgb):
    hsv = cv2.cvtColor(
        (img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV
    )
    sat_mean = hsv[:, :, 1].mean() / 255.0
    if sat_mean < 0.2:
        return "low"
    elif sat_mean < 0.4:
        return "medium"
    else:
        return "high"


CLASSIFIERS = {
    "brightness": classify_brightness,
    "color_temp": classify_color_temp,
    "saturation": classify_saturation,
}


def load_model(model_type, checkpoint, device):
    if model_type == "unet":
        from unet_model import UNetEnhancer
        model = UNetEnhancer(pretrained=False).to(device)
    else:
        from lut_model import Learnable3DLUT
        model = Learnable3DLUT().to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return model


def run_analysis(model, test_ds, device, lpips_fn, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "best_worst")
    os.makedirs(img_dir, exist_ok=True)

    per_category = {
        cat: defaultdict(list) for cat in CLASSIFIERS
    }
    per_image = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for idx, (raw, target) in enumerate(
            tqdm(test_ds, desc="Analyzing errors")
        ):
            raw_b = raw.unsqueeze(0).to(device)
            pred = model(raw_b).squeeze(0).cpu()

            raw_np = raw.numpy().transpose(1, 2, 0)
            gt_np = target.numpy().transpose(1, 2, 0)
            pred_u8 = np.clip(
                pred.numpy().transpose(1, 2, 0) * 255, 0, 255
            ).astype(np.uint8)
            gt_u8 = np.clip(gt_np * 255, 0, 255).astype(np.uint8)

            s, p = compute_ssim_psnr(pred_u8, gt_u8)
            l = lpips_fn(
                pred.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1
            ).item()

            # Classify by raw image characteristics
            categories = {}
            for cat_name, classify_fn in CLASSIFIERS.items():
                cat_val = classify_fn(raw_np)
                categories[cat_name] = cat_val
                per_category[cat_name][cat_val].append(
                    {"ssim": s, "psnr": p, "lpips": l, "idx": idx}
                )

            per_image.append(
                {"idx": idx, "ssim": s, "psnr": p, "lpips": l, **categories}
            )
            all_preds.append((raw_np, gt_np, pred_u8))

    # --- Summary CSV ---
    summary_rows = []
    for cat_name, cat_vals in per_category.items():
        for cat_val, items in sorted(cat_vals.items()):
            ssims = [x["ssim"] for x in items]
            psnrs = [x["psnr"] for x in items]
            lpips_list = [x["lpips"] for x in items]
            summary_rows.append(
                {
                    "category": cat_name,
                    "value": cat_val,
                    "count": len(items),
                    "ssim_mean": round(np.mean(ssims), 4),
                    "ssim_std": round(np.std(ssims), 4),
                    "psnr_mean": round(np.mean(psnrs), 2),
                    "lpips_mean": round(np.mean(lpips_list), 4),
                }
            )

    summary_path = os.path.join(output_dir, "error_analysis_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=summary_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved to {summary_path}")

    # --- Per-image CSV ---
    per_image_path = os.path.join(
        output_dir, "error_analysis_per_image.csv"
    )
    per_image[0].keys()
    fieldnames = ["idx", "ssim", "psnr", "lpips"] + list(CLASSIFIERS.keys())
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image)
    print(f"Per-image saved to {per_image_path}")

    # --- Best/worst examples per category ---
    for cat_name, cat_vals in per_category.items():
        for cat_val, items in sorted(cat_vals.items()):
            sorted_items = sorted(items, key=lambda x: x["ssim"])
            worst_3 = sorted_items[:3]
            best_3 = sorted_items[-3:]

            for label, subset in [("worst", worst_3), ("best", best_3)]:
                for rank, item in enumerate(subset):
                    idx = item["idx"]
                    raw_np, gt_np, pred_u8 = all_preds[idx]
                    comparison = np.concatenate(
                        [raw_np, gt_np, pred_u8], axis=1
                    )
                    fname = f"{cat_name}_{cat_val}_{label}_{rank+1}_ssim={item['ssim']:.3f}.jpg"
                    cv2.imwrite(
                        os.path.join(img_dir, fname),
                        cv2.cvtColor(
                            (comparison * 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR,
                        ),
                    )

    print(f"Best/worst images saved to {img_dir}")

    # --- Print summary table ---
    print("\n--- Error Analysis Summary ---")
    for cat_name in CLASSIFIERS:
        print(f"\n{cat_name}:")
        rows = [r for r in summary_rows if r["category"] == cat_name]
        for r in sorted(rows, key=lambda x: x["ssim_mean"], reverse=True):
            print(
                f"  {r['value']:<20} n={r['count']:>4}  "
                f"SSIM={r['ssim_mean']:.4f}+/-{r['ssim_std']:.4f}  "
                f"PSNR={r['psnr_mean']:.2f}  "
                f"LPIPS={r['lpips_mean']:.4f}"
            )

    return summary_rows, per_image


def main():
    parser = argparse.ArgumentParser(
        description="Error analysis for image enhancement model"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expert", default="c")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", choices=["lut", "unet"], default="lut")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--output-dir", default="error_analysis")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device: {device} (CPU required for 3D LUT evaluation)")

    raw_dir = os.path.join(args.data_root, "raw")
    target_dir = os.path.join(args.data_root, args.expert)

    test_ds = FiveKDataset(raw_dir, target_dir, args.size, "test")
    print(f"Test images: {len(test_ds)}")

    model = load_model(args.model_type, args.checkpoint, device)
    lpips_fn = setup_lpips(device)

    run_analysis(model, test_ds, device, lpips_fn, args.output_dir)


if __name__ == "__main__":
    main()
