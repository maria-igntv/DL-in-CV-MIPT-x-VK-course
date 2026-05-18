#!/usr/bin/env python3
"""Merge external ONNX weights (*.onnx.data) into a single model.onnx file.

Run from repo root when you have both model.onnx and model.onnx.data next to each other:

  python scripts/inline_onnx_weights.py \\
    --in triton/model_repository/image_enhancer/1/model.onnx
"""

import argparse
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="Path to model.onnx (external tensors must live alongside)",
    )
    args = parser.parse_args()

    inp = Path(args.inp)
    if not inp.is_file():
        raise SystemExit(f"not found: {inp}")

    model = onnx.load(str(inp), load_external_data=True)
    onnx.save(model, str(inp), save_as_external_data=False)
    print(f"OK: weights inlined into {inp}")


if __name__ == "__main__":
    main()
