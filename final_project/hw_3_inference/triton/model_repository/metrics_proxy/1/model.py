"""Triton Python-backend proxy model with custom metrics.

Delegates inference to `image_enhancer` via BLS and exposes:
  - COUNTER: total processing time (seconds)
  - GAUGE  : number of requests currently being processed

Custom metrics are registered with Triton via the MetricFamily API
(Python backend >= 23.08) and automatically appear at the
Prometheus endpoint http://localhost:8002/metrics.
"""

import threading
import time
import logging

import numpy as np
import triton_python_backend_utils as pb_utils

logger = logging.getLogger("metrics_proxy")


class TritonPythonModel:
    def initialize(self, args):
        self._lock = threading.Lock()
        self.total_processing_time = 0.0
        self.current_requests = 0

        # Try to register Triton custom metric families (available >= 23.08).
        # Fallback to simple instance variables if the API is unavailable.
        self._use_triton_metrics = False
        try:
            self._processing_time_family = pb_utils.MetricFamily(
                name="image_enhancer_processing_time_seconds",
                description="Total processing time for image enhancement requests (COUNTER)",
                kind=pb_utils.MetricFamily.COUNTER,
            )
            self._current_requests_family = pb_utils.MetricFamily(
                name="image_enhancer_current_requests",
                description="Number of image enhancement requests currently being processed (GAUGE)",
                kind=pb_utils.MetricFamily.GAUGE,
            )
            self._use_triton_metrics = True
            logger.info("Triton custom metric families registered")
        except Exception:
            logger.warning(
                "pb_utils.MetricFamily not available — "
                "metrics will be tracked via instance variables only"
            )

    # ------------------------------------------------------------------
    def execute(self, requests):
        with self._lock:
            self.current_requests += 1

        if self._use_triton_metrics:
            gauge = self._current_requests_family.Metric(
                labels={"model": "metrics_proxy"}
            )
            gauge.set(float(self.current_requests))

        t0 = time.time()

        responses = []
        for req in requests:
            inp = pb_utils.get_input_tensor_by_name(req, "input_image")
            inp_np = inp.as_numpy()

            bls_req = pb_utils.InferenceRequest(
                model_name="image_enhancer",
                inputs=[pb_utils.Tensor("input_image", inp_np)],
                requested_output_names=["output_image"],
            )
            bls_resp = bls_req.exec()

            if bls_resp.has_error():
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=bls_resp.error()
                    )
                )
            else:
                out = pb_utils.get_output_tensor_by_name(bls_resp, "output_image")
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("output_image", out.as_numpy())
                        ]
                    )
                )

        elapsed = time.time() - t0

        with self._lock:
            self.current_requests -= 1
            self.total_processing_time += elapsed

        if self._use_triton_metrics:
            counter = self._processing_time_family.Metric(
                labels={"model": "metrics_proxy"}
            )
            counter.increment(elapsed)

            gauge = self._current_requests_family.Metric(
                labels={"model": "metrics_proxy"}
            )
            gauge.set(float(self.current_requests))

        return responses

    # ------------------------------------------------------------------
    def finalize(self):
        logger.info(
            f"Total processing time: {self.total_processing_time:.3f}s "
            f"| Final current_requests: {self.current_requests}"
        )
