"""gRPC client for image enhancement via Triton Inference Server.

Usage:
    python client_grpc.py --image photo.jpg --output enhanced.jpg
    python client_grpc.py --image photo.jpg --url localhost:8001
"""

import argparse
import cv2
import numpy as np
import time

try:
    import tritonclient.grpc as grpc_client
except ImportError:
    print("Install tritonclient: pip install tritonclient[grpc]")
    raise


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


def infer_grpc(image_np: np.ndarray, url: str = "localhost:8001",
               model: str = "image_enhancer") -> np.ndarray:
    client = grpc_client.InferenceServerClient(url=url)

    inputs = [grpc_client.InferInput("input_image", image_np.shape, "FP32")]
    inputs[0].set_data_from_numpy(image_np)

    outputs = [grpc_client.InferRequestedOutput("output_image")]

    t0 = time.time()
    result = client.infer(model_name=model, inputs=inputs, outputs=outputs)
    elapsed = time.time() - t0

    output = result.as_numpy("output_image")
    print(f"Inference OK  latency={elapsed*1000:.1f}ms  shape={output.shape}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Triton gRPC client for image enhancement")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="enhanced_output.jpg", help="Path to save output")
    parser.add_argument("--url", default="localhost:8001", help="gRPC URL")
    parser.add_argument("--model", default="image_enhancer", help="Model name in Triton")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    image_np = preprocess(args.image, args.size)
    print(f"Input: {args.image}  shape={image_np.shape}")

    output_np = infer_grpc(image_np, args.url, args.model)
    result_img = postprocess(output_np)
    cv2.imwrite(args.output, result_img)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
