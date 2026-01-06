#!/bin/bash
# Starts vLLM server in background, waits for it to be ready, then starts the RunPod handler.

set -e

echo "Starting dots.ocr extraction handler..."
echo "VLLM_PORT=${VLLM_PORT:-8080}"
echo "MODEL_NAME=${MODEL_NAME:-rednote-hilab/dots.ocr}"

# Start vLLM server in background
echo "Launching vLLM server..."
vllm serve "${MODEL_NAME:-rednote-hilab/dots.ocr}" --host 0.0.0.0 --port "${VLLM_PORT:-8080}" --trust-remote-code --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" --max-model-len "${MAX_MODEL_LEN:-24000}" --dtype float16 &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
python3 -c "
import time
import httpx
import os

max_retries = 90
port = os.environ.get('VLLM_PORT', '8080')

for i in range(max_retries):
    try:
        r = httpx.get(f'http://localhost:{port}/health', timeout=5)
        if r.status_code == 200:
            print('vLLM server is ready!')
            exit(0)
    except Exception as e:
        pass
    print(f'Waiting for vLLM... (attempt {i+1}/{max_retries})')
    time.sleep(2)

print('ERROR: vLLM server failed to start after 3 minutes')
exit(1)
"

if [ $? -ne 0 ]; then
    echo "Health check script failed"
    exit 1
fi

# Debug: Show port status
echo "=== Port status ==="
netstat -tlnp 2>/dev/null || ss -tlnp 2>/dev/null || echo "netstat/ss not available"
echo "==================="

# Start RunPod handler
echo "Starting RunPod handler..."
echo "Handler PID will be: $$"
python3 -u handler.py

# If handler exits, kill vLLM
echo "Handler exited, cleaning up..."
kill $VLLM_PID 2>/dev/null || true
