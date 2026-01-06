# Architecture: Document Processing Inference Service

**Version:** 1.0
**Last Updated:** January 2025
**Status:** Production

---

## Overview

This repository provides ML inference infrastructure for document processing, deployed on RunPod Serverless. The system classifies document images and extracts structured data using specialized vision-language models.

### Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Classification Handler** | Identify document type (W2, 1099, bank statement, etc.) | Qwen2.5-VL-7B-Instruct |
| **Extraction Handler** | Extract structured data with layout preservation | dots.ocr (1.7B) via vLLM |
| **Gateway Server** | Authentication, routing, logging, retry logic | FastAPI |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

  Client Application              Gateway (Your Server)              RunPod Serverless
  ──────────────────              ─────────────────────              ─────────────────

  ┌─────────────────┐             ┌─────────────────────┐
  │                 │             │                     │            ┌──────────────────┐
  │   Your App      │────────────►│   /v1/classify      │───────────►│ classify-endpoint│
  │                 │   HTTPS     │                     │   HTTPS    │ (Qwen2.5-VL-7B)  │
  └─────────────────┘             │                     │            └──────────────────┘
                                  │   Gateway Server    │
                                  │   (FastAPI)         │            ┌──────────────────┐
                                  │                     │───────────►│ extract-endpoint │
                                  │   /v1/extract       │   HTTPS    │ (dots.ocr 1.7B)  │
                                  │                     │            └──────────────────┘
                                  │   Features:         │
                                  │   - API Key Auth    │
                                  │   - Request Logging │
                                  │   - Retry Logic     │
                                  │   - Health Checks   │
                                  └─────────────────────┘
                                           │
                                           ▼
                                  ┌─────────────────────┐
                                  │   Document Storage  │
                                  │   (Pre-signed URLs) │◄─────────── RunPod fetches
                                  └─────────────────────┘             images directly
```

---

## Why a Gateway?

While RunPod endpoints can be called directly, the gateway provides:

| Benefit | Description |
|---------|-------------|
| **Single Auth Point** | One API key for clients; RunPod keys stay server-side |
| **Request Logging** | Centralized logging for debugging and analytics |
| **Abstraction** | Swap RunPod for local GPU later without client changes |
| **Retry Logic** | Automatic retry with exponential backoff |
| **Health Aggregation** | Single health endpoint for all inference services |

---

## Model Selection

### Two Specialized Models

The system uses two models optimized for their respective tasks:

| Task | Model | Parameters | VRAM | Rationale |
|------|-------|------------|------|-----------|
| **Classification** | Qwen2.5-VL-7B-Instruct | 7B | ~16GB | General VLM, excellent at document type recognition |
| **Extraction** | dots.ocr | 1.7B | ~8GB | SOTA OCR, beats GPT-4o on OmniDocBench |

### Why dots.ocr for Extraction?

[dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) achieves state-of-the-art document parsing:

| Benchmark | dots.ocr | GPT-4o | Qwen2.5-VL-72B |
|-----------|----------|--------|----------------|
| OmniDocBench (EN) | **0.125** | 0.233 | 0.252 |
| Table TEDS (EN) | **88.6%** | 72.0% | 76.8% |

**Key advantages:**
- Compact (1.7B) = fast inference, low cost
- Layout-aware: preserves reading order, handles tables
- Native vLLM 0.11.0+ support
- Multilingual (100+ languages)

---

## API Reference

### Classification Endpoint

```
POST /v1/classify
X-API-Key: <your-api-key>
```

**Request:**
```json
{
  "image_urls": ["https://storage.example.com/document.png"],
  "prompt": "optional custom prompt"
}
```

**Response:**
```json
{
  "type": "W2",
  "confidence": 0.95,
  "reasoning": "Document contains W-2 form header and wage fields",
  "latency_ms": 1234
}
```

**Supported Document Types:**
- `W2`, `1099-INT`, `1099-DIV`, `1099-MISC`, `1099-NEC`, `1099-R`, `1098`
- `bank_statement`, `credit_card_statement`
- `invoice`, `receipt`, `check`
- `other`

### Extraction Endpoint

```
POST /v1/extract
X-API-Key: <your-api-key>
```

**Request:**
```json
{
  "image_urls": [
    "https://storage.example.com/page1.png",
    "https://storage.example.com/page2.png"
  ],
  "doc_type": "bank_statement",
  "prompt_mode": "layout_all"
}
```

**Response:**
```json
{
  "data": {
    "header": { "bank_name": "...", "account_number": "..." },
    "transactions": [...],
    "summary": { "beginning_balance": "...", "ending_balance": "..." }
  },
  "raw_ocr": {
    "pages": [...],
    "combined_text": "..."
  },
  "confidence": {
    "overall": 0.95,
    "page_count": 2,
    "element_count": 45
  },
  "page_count": 2,
  "latency_ms": 5678
}
```

### Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Full health check with RunPod endpoint status |
| `GET /ready` | Kubernetes readiness probe |
| `GET /live` | Kubernetes liveness probe |

---

## Repository Structure

```
docproc-inference/
├── handlers/
│   ├── classify/                # Classification handler
│   │   ├── Dockerfile           # runpod/pytorch base + transformers
│   │   ├── handler.py           # RunPod serverless handler
│   │   ├── requirements.txt
│   │   ├── conftest.py          # Test fixtures
│   │   └── test_handler.py      # Unit tests
│   │
│   └── extract/                 # Extraction handler
│       ├── Dockerfile           # vLLM base image
│       ├── handler.py           # RunPod serverless handler
│       ├── doc_transformers.py  # Document schema transformers
│       ├── start.sh             # vLLM + handler startup
│       ├── requirements.txt
│       ├── conftest.py          # Test fixtures
│       └── test_handler.py      # Unit tests
│
├── gateway/
│   ├── Dockerfile               # Python 3.11 slim
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py              # FastAPI application factory
│   │   ├── config.py            # Environment configuration
│   │   ├── routes/
│   │   │   ├── classify.py      # POST /v1/classify
│   │   │   ├── extract.py       # POST /v1/extract
│   │   │   └── health.py        # Health check endpoints
│   │   ├── services/
│   │   │   └── runpod_client.py # Async RunPod API client
│   │   └── middleware/
│   │       ├── auth.py          # API key authentication
│   │       └── logging.py       # Request logging
│   └── tests/                   # Gateway unit tests
│
├── tests/
│   └── test_integration.py      # End-to-end integration tests
│
├── prompts/
│   └── roles/                   # Agent role definitions
│
├── docker-compose.yml           # Local gateway development
├── .env.example                 # Environment template
├── README.md                    # User documentation
└── ARCHITECTURE.md              # This file
```

---

## Deployment Model

### Handler Deployment (RunPod Serverless)

Each handler is deployed as a separate RunPod Serverless endpoint:

```
┌─────────────────────────────────────────────────────────┐
│  RunPod Serverless                                      │
│                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐    │
│  │ classify-endpoint   │    │ extract-endpoint    │    │
│  │                     │    │                     │    │
│  │ Image: ghcr.io/.../ │    │ Image: ghcr.io/.../ │    │
│  │   docproc-classify  │    │   docproc-extract   │    │
│  │                     │    │                     │    │
│  │ GPU: RTX 3090/4090  │    │ GPU: RTX 3090/4090  │    │
│  │ VRAM: 24GB          │    │ VRAM: 24GB          │    │
│  │ Scale: 0-N workers  │    │ Scale: 0-N workers  │    │
│  └─────────────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Configuration:**
- **Max Workers:** 2-5 (adjust based on load)
- **Idle Timeout:** 5s (dev) / 60s+ (prod)
- **Flash Boot:** Enabled for faster cold starts

### Gateway Deployment

The gateway runs as a standard container (Docker Compose, Cloud Run, Railway, etc.):

```bash
# Local development
docker-compose up

# Production
docker build -t gateway ./gateway
docker run -p 8000:8000 --env-file .env gateway
```

---

## Key Design Decisions

### 1. Separate Models for Classification vs Extraction

**Decision:** Use Qwen2.5-VL-7B for classification, dots.ocr for extraction.

**Rationale:**
- Classification needs general document understanding → larger VLM
- Extraction needs precise OCR with layout → specialized model
- dots.ocr is 4x smaller but outperforms GPT-4o on document parsing

### 2. Dual-Process Architecture for Extract Handler

**Decision:** Run vLLM as a server + handler as separate processes in same container.

**Rationale:**
- vLLM server provides OpenAI-compatible API
- Allows async HTTP calls from handler
- Simpler than direct Python API integration for dots.ocr

### 3. Gateway as Abstraction Layer

**Decision:** All clients go through gateway, never call RunPod directly.

**Rationale:**
- Single point for authentication and logging
- Can swap backend (RunPod → local GPU) without client changes
- Centralized retry logic and error handling

### 4. Image URLs Instead of Base64

**Decision:** Pass pre-signed URLs, let RunPod fetch images.

**Rationale:**
- Reduces request payload size
- Avoids gateway memory pressure
- RunPod workers fetch directly from storage

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GATEWAY_API_KEY` | Yes | API key for gateway authentication |
| `RUNPOD_API_KEY` | Yes | RunPod platform API key |
| `RUNPOD_CLASSIFY_ENDPOINT` | Yes | Endpoint ID for classification |
| `RUNPOD_EXTRACT_ENDPOINT` | Yes | Endpoint ID for extraction |
| `RUNPOD_TIMEOUT` | No | Request timeout (default: 120s) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

---

## Cost Considerations

| Resource | Cost Model |
|----------|------------|
| RunPod GPU (active) | ~$0.40-0.80/hr per worker |
| RunPod GPU (idle) | $0 with 0 min workers |
| Gateway (always-on) | ~$5-20/month (Cloud Run/Railway) |

**Optimization tips:**
- Set idle timeout to 5s during development
- Use Flash Boot for faster cold starts
- Scale max workers based on actual load
