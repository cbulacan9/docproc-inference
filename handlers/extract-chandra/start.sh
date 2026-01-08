#!/bin/bash
set -e

echo "Starting vLLM server with Chandra model..."

# Start vLLM in background
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME:-datalab-to/chandra} \
    --port ${VLLM_PORT:-8080} \
    --trust-remote-code \
    --max-model-len 8192 &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM server to start..."
for i in {1..60}; do
    if curl -s http://localhost:${VLLM_PORT:-8080}/health > /dev/null 2>&1; then
        echo "vLLM server is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "vLLM server failed to start within 120 seconds"
        exit 1
    fi
    sleep 2
done

# Start handler
echo "Starting Chandra handler..."
python handler.py
