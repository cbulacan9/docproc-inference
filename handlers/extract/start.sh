#!/bin/bash
# handlers/extract/start.sh
#
# Starts vLLM server in background, waits for it to be ready,
# then starts the RunPod handler.

set -e

echo "Starting dots.ocr extraction handler..."

# Start vLLM server in background
echo "Launching vLLM server with dots.ocr..."
vllm serve ${MODEL_NAME:-rednote-hilab/dots.ocr} \
    --host 0.0.0.0 \
    --port ${VLLM_PORT:-8000} \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 24000 \
    --dtype float16 \
    &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
MAX_RETRIES=60
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:${VLLM_PORT:-8000}/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for vLLM... (attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: vLLM server failed to start"
    exit 1
fi

# Start RunPod handler
echo "Starting RunPod handler..."
python -u handler.py

# If handler exits, kill vLLM
kill $VLLM_PID 2>/dev/null || true
