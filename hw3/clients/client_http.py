"""HTTP client for image enhancement via Triton Inference Server.

Usage:
    python client_http.py --image photo.jpg --output enhanced.jpg
    python client_http.py --image photo.jpg --url http://localhost:8000
"""

import argparse
import cv2
import numpy as np
import requests
import json
import time


def preprocess(image_path: str, size: int = 512) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(output: np.ndarray) -> np.ndarray:
    img = output.squeeze(0)
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def infer_http(image_np: np.ndarray, url: str = "http://localhost:8000",
               model: str = "image_enhancer") -> np.ndarray:
    payload = {
        "inputs": [
            {
                "name": "input_image",
                "shape": list(image_np.shape),
                "datatype": "FP32",
                "data": image_np.flatten().tolist(),
            }
        ]
    }

    t0 = time.time()
    resp = requests.post(f"{url}/v2/models/{model}/infer", json=payload)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"Triton error: {resp.status_code} {resp.text}")

    result = resp.json()
    output_data = np.array(result["outputs"][0]["data"], dtype=np.float32)
    output_shape = result["outputs"][0]["shape"]
    output = output_data.reshape(output_shape)
    print(f"Inference OK  latency={elapsed*1000:.1f}ms  shape={output_shape}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Triton HTTP client for image enhancement")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="enhanced_output.jpg", help="Path to save output")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="image_enhancer", help="Model name in Triton")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    image_np = preprocess(args.image, args.size)
    print(f"Input: {args.image}  shape={image_np.shape}")

    output_np = infer_http(image_np, args.url, args.model)
    result_img = postprocess(output_np)
    cv2.imwrite(args.output, result_img)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
