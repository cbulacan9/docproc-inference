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
    httpx>=0.26.0 \
    nest_asyncio>=1.5.0

# Copy handler code
COPY handler.py .
COPY doc_transformers.py .
COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV MODEL_NAME=rednote-hilab/dots.ocr
ENV VLLM_PORT=8080
ENV MAX_MODEL_LEN=24000
ENV GPU_MEMORY_UTILIZATION=0.95

# =============================================================================
# Model Pre-download (Optional - for production builds)
# =============================================================================
# Uncomment the following line to pre-download the model during image build.
# This significantly reduces cold start time (~30-60s saved) but increases
# image size by ~4GB. Recommended for production deployments.
#
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('rednote-hilab/dots.ocr')"
#
# Note: If using a private model or custom HuggingFace token, set HF_TOKEN:
#   ARG HF_TOKEN
#   RUN HF_TOKEN=${HF_TOKEN} python -c "..."
# =============================================================================

# Override base image entrypoint and start our script
ENTRYPOINT []
CMD ["/bin/bash", "./start.sh"]
