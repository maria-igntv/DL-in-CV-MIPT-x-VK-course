"""Export Learnable3DLUT to ONNX for Triton Inference Server."""

import argparse
import os

import torch

from lut_model import Learnable3DLUT


def export(checkpoint: str, out_path: str, lut_dim: int = 17, n_luts: int = 3, img_size: int = 512):
    model = Learnable3DLUT(lut_dim=lut_dim, n_luts=n_luts)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=16,
        input_names=["input_image"],
        output_names=["output_image"],
        dynamic_axes={
            "input_image": {0: "batch", 2: "height", 3: "width"},
            "output_image": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"Exported ONNX model to {out_path}")
    print(f"  opset=16  dynamic_axes=batch,height,width")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_lut.pth")
    parser.add_argument("--out", default="triton/model_repository/image_enhancer/1/model.onnx")
    parser.add_argument("--lut-dim", type=int, default=17)
    parser.add_argument("--n-luts", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()
    export(args.checkpoint, args.out, args.lut_dim, args.n_luts, args.img_size)
