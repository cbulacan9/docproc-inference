FROM vllm/vllm-openai:latest

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py .
COPY chandra_parser.py .
COPY doc_transformers.py .
COPY confidence.py .
COPY start.sh .

RUN chmod +x start.sh

# Environment defaults
ENV VLLM_PORT=8080
ENV MODEL_NAME=datalab-to/chandra
ENV HF_HOME=/app/models

EXPOSE 8080

CMD ["./start.sh"]
