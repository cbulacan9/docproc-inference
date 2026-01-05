# Document Processing Inference Service

A serverless document processing pipeline that classifies and extracts structured data from document images using vision language models on RunPod.

## Architecture

```
                                    +---------------------------+
                                    |     RunPod Endpoints      |
+----------+      +----------+      |  +---------------------+  |
|          |      |          |      |  | Classify            |  |
|  Client  +----->+ Gateway  +----->+  | (Qwen2.5-VL-7B)     |  |
|          |      |  :8000   |      |  +---------------------+  |
+----------+      +----+-----+      |  +---------------------+  |
                       |           |  | Extract             |  |
                       |           |  | (dots.ocr + vLLM)   |  |
                  +----v-----+      |  +---------------------+  |
                  |   Auth   |      +---------------------------+
                  | X-API-Key|
                  +----------+
```

**Components:**

| Component | Description |
|-----------|-------------|
| **Gateway** | FastAPI server that authenticates requests and proxies to RunPod |
| **Classify Handler** | RunPod serverless endpoint using Qwen2.5-VL-7B for document classification |
| **Extract Handler** | RunPod serverless endpoint using dots.ocr via vLLM for OCR and data extraction |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- RunPod account with API key
- RunPod endpoints deployed (see [Deployment](#deployment))

### Local Development

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd docproc-inference
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your RunPod credentials and endpoint IDs
   ```

3. **Start the gateway:**

   ```bash
   docker-compose up --build
   ```

4. **Verify it's running:**

   ```bash
   curl http://localhost:8000/live
   # {"status": "live"}

   curl http://localhost:8000/health
   # {"status": "healthy", "gateway": "healthy", "endpoints": {...}}
   ```

## API Reference

### Authentication

All `/v1/*` endpoints require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/v1/classify ...
```

### POST /v1/classify

Classify a document from page images.

**Request:**

```json
{
  "image_urls": ["https://example.com/document-page1.png"],
  "prompt": "optional custom classification prompt"
}
```

**Response:**

```json
{
  "type": "W2",
  "confidence": 0.95,
  "reasoning": "Document contains W-2 form fields including employer information and wage data",
  "latency_ms": 1234
}
```

**Supported Document Types:**

`W2`, `1099-INT`, `1099-DIV`, `1099-MISC`, `1099-NEC`, `1099-R`, `1098`, `bank_statement`, `credit_card_statement`, `invoice`, `receipt`, `check`, `other`

**Example:**

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "image_urls": ["https://your-presigned-url.com/document.png"]
  }'
```

### POST /v1/extract

Extract structured data from document images.

**Request:**

```json
{
  "image_urls": [
    "https://example.com/page1.png",
    "https://example.com/page2.png"
  ],
  "doc_type": "W2",
  "prompt": "optional custom extraction prompt"
}
```

**Response:**

```json
{
  "data": {
    "employer_name": "Acme Corp",
    "employer_ein": "12-3456789",
    "employee_name": "John Doe",
    "employee_ssn": "***-**-1234",
    "wages": 75000.00,
    "federal_tax_withheld": 12000.00
  },
  "confidence": {
    "employer_name": 0.98,
    "wages": 0.95
  },
  "doc_type": "W2",
  "page_count": 1,
  "latency_ms": 5678
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "image_urls": ["https://your-presigned-url.com/w2.png"],
    "doc_type": "W2"
  }'
```

### Health Endpoints

| Endpoint | Auth Required | Description |
|----------|---------------|-------------|
| `GET /health` | No | Full health check with RunPod endpoint status |
| `GET /ready` | No | Kubernetes readiness probe |
| `GET /live` | No | Kubernetes liveness probe |

**Health Response Example:**

```json
{
  "status": "healthy",
  "gateway": "healthy",
  "endpoints": {
    "classify": {
      "status": "idle",
      "workers": {"idle": 0, "running": 0}
    },
    "extract": {
      "status": "idle",
      "workers": {"idle": 0, "running": 0}
    }
  }
}
```

## Deployment

### 1. Build Handler Docker Images

```bash
# Classification handler
cd handlers/classify
docker build -t your-registry/classify-handler:latest .
docker push your-registry/classify-handler:latest

# Extraction handler
cd handlers/extract
docker build -t your-registry/extract-handler:latest .
docker push your-registry/extract-handler:latest
```

### 2. Create RunPod Endpoints

1. Go to [RunPod Console](https://runpod.io/console/serverless)
2. Create a new Serverless endpoint for each handler
3. Configure GPU type (recommend A40/A100 for production)
4. Set container image to your pushed images
5. Note the endpoint IDs for configuration

### 3. Deploy Gateway

**Option A: Docker Compose (recommended for development)**

```bash
cp .env.example .env
# Configure .env with your RunPod credentials
docker-compose up -d
```

**Option B: Cloud Run / Railway / Fly.io**

Deploy the `gateway/` directory with environment variables configured.

### 4. Verify Deployment

```bash
# Check health
curl https://your-gateway-url/health

# Test classification
curl -X POST https://your-gateway-url/v1/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"image_urls": ["https://your-test-image-url"]}'
```

## Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GATEWAY_API_KEY` | Yes | API key for gateway authentication | - |
| `RUNPOD_API_KEY` | Yes | RunPod platform API key | - |
| `RUNPOD_CLASSIFY_ENDPOINT` | Yes | Endpoint ID for classification | - |
| `RUNPOD_EXTRACT_ENDPOINT` | Yes | Endpoint ID for extraction | - |
| `RUNPOD_TIMEOUT` | No | Request timeout in seconds | 120 |
| `RUNPOD_MAX_RETRIES` | No | Maximum retry attempts | 3 |
| `RATE_LIMIT_REQUESTS` | No | Max requests per window | 100 |
| `RATE_LIMIT_WINDOW` | No | Rate limit window (seconds) | 60 |
| `LOG_LEVEL` | No | Logging level | INFO |

## Development

### Running Unit Tests

```bash
# Gateway tests
cd gateway
pytest tests/ -v

# Handler tests
cd handlers/classify
python test_handler.py

cd handlers/extract
python test_handler.py
```

### Running Integration Tests

Integration tests require a running gateway with configured RunPod endpoints:

```bash
export INTEGRATION_TEST_URL=http://localhost:8000
export GATEWAY_API_KEY=your-test-key
pytest tests/test_integration.py -v -m integration
```

### Project Structure

```
docproc-inference/
├── handlers/
│   ├── classify/          # Document classification handler
│   │   ├── Dockerfile
│   │   ├── handler.py
│   │   ├── requirements.txt
│   │   └── test_handler.py
│   └── extract/           # Document extraction handler
│       ├── Dockerfile
│       ├── handler.py
│       ├── transformers.py
│       ├── start.sh
│       ├── requirements.txt
│       └── test_handler.py
├── gateway/               # FastAPI gateway server
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routes/
│   │   ├── services/
│   │   └── middleware/
│   └── tests/
├── tests/                 # Integration tests
│   └── test_integration.py
├── docker-compose.yml
├── .env.example
└── README.md
```

## Troubleshooting

### Common Issues

**Cold Start Latency**

RunPod serverless endpoints have cold start times of 30-60 seconds when no workers are active. For production:
- Enable FlashBoot in RunPod console
- Configure minimum idle workers
- Increase `RUNPOD_TIMEOUT` if needed

**401 Unauthorized**

- Verify `X-API-Key` header is set correctly
- Check `GATEWAY_API_KEY` environment variable matches
- Ensure header name is exactly `X-API-Key` (case-sensitive)

**502 Bad Gateway / Inference Failed**

- Check RunPod endpoint status in console
- Verify `RUNPOD_API_KEY` is correct
- Check endpoint IDs match deployed handlers
- Review gateway logs for detailed error messages

**Timeout Errors**

- Increase `RUNPOD_TIMEOUT` for large documents
- Check RunPod worker availability
- Consider enabling persistent workers for high-traffic use

### Viewing Logs

```bash
# Docker Compose logs
docker-compose logs -f gateway

# Check RunPod logs in console
# https://runpod.io/console/serverless -> Your Endpoint -> Logs
```

### Health Check Debugging

```bash
# Detailed health status
curl -s http://localhost:8000/health | jq .

# Check individual endpoint status
curl -s http://localhost:8000/health | jq '.endpoints.classify'
```

## License

[Your License Here]
