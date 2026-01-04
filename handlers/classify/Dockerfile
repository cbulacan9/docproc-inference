# handlers/classify/Dockerfile

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py .

# Pre-download model during build (optional but reduces cold start)
# Uncomment for production builds:
# RUN python -c "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor; \
#     Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True); \
#     AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

CMD ["python", "-u", "handler.py"]
