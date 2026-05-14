FROM nvcr.io/nvidia/tritonserver:24.01-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ /app/model/
COPY triton/ /models/

ENV TRITON_MODEL_REPOSITORY=/models/model_repository

CMD ["tritonserver", "--model-repository=/models/model_repository"]
