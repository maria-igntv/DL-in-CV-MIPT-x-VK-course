# 24.01 ships ONNX Runtime that only supports ONNX IR ≤ 9; many torch exports use IR 10.
# 24.12+ is a better default for macOS/CPU and modern ONNX. On Linux+NVIDIA, add `deploy`/GPU in compose.
FROM nvcr.io/nvidia/tritonserver:24.12-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ /app/model/
COPY triton/ /models/

ENV TRITON_MODEL_REPOSITORY=/models/model_repository

CMD ["tritonserver", "--model-repository=/models/model_repository"]
