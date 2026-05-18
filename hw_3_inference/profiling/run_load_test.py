import time
import numpy as np
import concurrent.futures
import tritonclient.grpc as grpc_client


def infer_once():
    client = grpc_client.InferenceServerClient(url="localhost:8001")
    dummy = np.random.rand(1, 3, 512, 512).astype(np.float32)
    inp = grpc_client.InferInput("input_image", dummy.shape, "FP32")
    inp.set_data_from_numpy(dummy)
    out = grpc_client.InferRequestedOutput("output_image")

    t0 = time.time()
    client.infer(
        model_name="image_enhancer",
        inputs=[inp],
        outputs=[out],
        model_version=""
    )
    return (time.time() - t0) * 1000


def test_concurrency(n_workers, n_requests=30):
    latencies = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(infer_once) for _ in range(n_requests)]
        for f in concurrent.futures.as_completed(futures):
            latencies.append(f.result())

    arr = np.array(latencies)
    throughput = n_requests / (arr.sum() / 1000)
    print(f"\n--- Concurrency {n_workers} ---")
    print(f"  Avg latency: {arr.mean():.1f} ms")
    print(f"  P95 latency: {np.percentile(arr, 95):.1f} ms")
    print(f"  Throughput:  {throughput:.1f} inf/s")


if __name__ == "__main__":
    for c in [1, 2, 4, 8]:
        test_concurrency(c)
