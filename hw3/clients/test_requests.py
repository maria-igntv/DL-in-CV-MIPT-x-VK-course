"""Batch test script — sends multiple requests and reports statistics.

Usage:
    python test_requests.py --images-dir /path/to/raw --n 20
"""

import argparse
import os
import time
from glob import glob

import cv2
import numpy as np

try:
    import tritonclient.grpc as grpc_client
except ImportError:
    import tritonclient.http as http_client
    grpc_client = None


def preprocess(path: str, size: int = 512) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--model", default="image_enhancer")
    parser.add_argument("--n", type=int, default=20, help="Number of images to test")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.images_dir, "*.*")))[: args.n]
    if not files:
        print("No images found")
        return

    if grpc_client is not None:
        client = grpc_client.InferenceServerClient(url=args.url)
    else:
        client = http_client.InferenceServerClient(url=args.url.replace("8001", "8000"))

    latencies = []
    for f in files:
        img = preprocess(f, args.size)

        if grpc_client is not None:
            inp = grpc_client.InferInput("input_image", img.shape, "FP32")
            inp.set_data_from_numpy(img)
            out_req = grpc_client.InferRequestedOutput("output_image")
            t0 = time.time()
            resp = client.infer(args.model, [inp], [out_req])
        else:
            t0 = time.time()
            resp = client.infer(args.model, img)

        lat = (time.time() - t0) * 1000
        latencies.append(lat)
        print(f"  {os.path.basename(f):30s}  {lat:.1f} ms")

    arr = np.array(latencies)
    print(f"\n--- Summary ({len(arr)} requests) ---")
    print(f"  Mean:   {arr.mean():.1f} ms")
    print(f"  Median: {np.median(arr):.1f} ms")
    print(f"  P95:    {np.percentile(arr, 95):.1f} ms")
    print(f"  P99:    {np.percentile(arr, 99):.1f} ms")
    print(f"  Min:    {arr.min():.1f} ms")
    print(f"  Max:    {arr.max():.1f} ms")


if __name__ == "__main__":
    main()
