#!/bin/bash
# Run Model Analyzer to find optimal configuration.
# Must be executed from the Triton container (not SDK) since we have local model files.
#
# Usage:
#   bash run_model_analyzer.sh

set -e

CONFIG=${CONFIG:-model_analyzer_config.yaml}
TRITON_URL=${TRITON_URL:-localhost:8001}

echo "=== Model Analyzer ==="
echo "Config: $CONFIG"
echo "Triton: $TRITON_URL"

model-analyzer profile \
    -f $CONFIG \
    --triton-launch-mode=local \
    --triton-http-endpoint $TRITON_URL

echo ""
echo "Generating summary report..."

model-analyzer analyze \
    -f $CONFIG \
    --analysis-models image_enhancer \
    --export

echo ""
echo "Results saved to model_analyzer_output/"
echo "Check model_analyzer_output/plots/ for visual comparison"
