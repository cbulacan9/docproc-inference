# handlers/classify/Dockerfile

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install transformers from git (latest with Qwen2.5-VL fixes)
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers

# Clear any cached model files to force fresh download
RUN rm -rf /root/.cache/huggingface /app/hf_cache

# Copy handler code
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

CMD ["python", "-u", "handler.py"]
