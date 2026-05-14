#!/bin/bash
# Run Performance Analyzer for load testing.
# Must be executed from the SDK container:
#   docker run --net host nvcr.io/nvidia/tritonserver:24.01-py3-sdk bash -c "perf_analyzer -m image_enhancer"

set -e

MODEL_NAME=${MODEL_NAME:-image_enhancer}
URL=${URL:-localhost:8001}
CONCURRENCY_RANGE=${CONCURRENCY_RANGE:-1:8:1}
INPUT_DATA=${INPUT_DATA:-}

echo "=== Performance Analyzer ==="
echo "Model:     $MODEL_NAME"
echo "URL:       $URL"
echo "Concurrency: $CONCURRENCY_RANGE"

CMD="perf_analyzer -m $MODEL_NAME -u $URL --concurrency-range $CONCURRENCY_RANGE"

if [ -n "$INPUT_DATA" ]; then
    CMD="$CMD --input-data $INPUT_DATA"
fi

echo "Running: $CMD"
echo ""
$CMD 2>&1 | tee perf_analyzer_output.txt

echo ""
echo "Results saved to perf_analyzer_output.txt"
