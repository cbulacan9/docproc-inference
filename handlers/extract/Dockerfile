# handlers/extract/Dockerfile
#
# dots.ocr extraction handler using native vLLM 0.11.0+ support
# Model: rednote-hilab/dots.ocr (1.7B parameters, ~8GB VRAM)

FROM vllm/vllm-openai:v0.11.0

WORKDIR /app

# Install handler dependencies
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    requests>=2.31.0 \
    pillow>=10.0.0 \
    httpx>=0.26.0

# Copy handler code
COPY handler.py .
COPY transformers.py .
COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV MODEL_NAME=rednote-hilab/dots.ocr
ENV VLLM_PORT=8000

# Pre-download model during build (reduces cold start significantly)
# Uncomment for production builds:
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('rednote-hilab/dots.ocr')"

# Start vLLM server and handler
CMD ["./start.sh"]
