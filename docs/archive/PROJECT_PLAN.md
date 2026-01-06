# Inference Service Project Plan

**Generated:** January 2025
**Source:** RUNPOD_IMPLEMENTATION.md
**Purpose:** Phased implementation guide for coding agents

---

## 1. Repository Specification

### Target Directory Structure

```
inference/
├── handlers/
│   ├── classify/
│   │   ├── Dockerfile           # RunPod container with vLLM + Qwen2.5-VL-7B
│   │   ├── handler.py           # RunPod serverless handler for classification
│   │   ├── requirements.txt     # Python dependencies (runpod, transformers, torch)
│   │   └── test_handler.py      # Unit tests for JSON parsing and validation
│   │
│   └── extract/
│       ├── Dockerfile           # RunPod container with vLLM 0.11.0 + dots.ocr
│       ├── handler.py           # RunPod serverless handler for extraction
│       ├── transformers.py      # OCR output → document schema converters
│       ├── start.sh             # vLLM server startup + handler launch script
│       ├── requirements.txt     # Python dependencies (runpod, httpx, pillow)
│       └── test_handler.py      # Unit tests for output parsing
│
├── gateway/
│   ├── Dockerfile               # Python 3.11 slim container with FastAPI
│   ├── requirements.txt         # FastAPI, uvicorn, httpx, pydantic
│   ├── app/
│   │   ├── __init__.py          # Package marker
│   │   ├── main.py              # FastAPI app factory and startup
│   │   ├── config.py            # Settings dataclass from environment
│   │   ├── routes/
│   │   │   ├── __init__.py      # Package marker
│   │   │   ├── classify.py      # POST /v1/classify endpoint
│   │   │   ├── extract.py       # POST /v1/extract endpoint
│   │   │   └── health.py        # GET /health, /ready, /live endpoints
│   │   ├── services/
│   │   │   ├── __init__.py      # Package marker
│   │   │   └── runpod_client.py # Async RunPod API client with retries
│   │   └── middleware/
│   │       ├── __init__.py      # Package marker
│   │       ├── auth.py          # X-API-Key header validation
│   │       └── logging.py       # Request logging with timing
│   └── tests/
│       └── test_routes.py       # Route unit tests with mocked RunPod
│
├── tests/
│   └── test_integration.py      # End-to-end tests against live endpoints
│
├── docker-compose.yml           # Local development compose file
├── .env.example                 # Template environment variables
└── README.md                    # Setup and deployment instructions
```

### File Count Summary

| Component | Files | Estimated Lines |
|-----------|-------|-----------------|
| classify handler | 4 | ~350 |
| extract handler | 5 | ~450 |
| gateway | 12 | ~550 |
| root configs | 3 | ~80 |
| **Total** | **24** | **~1,430** |

---

## 2. Component Dependency Graph

```
                    ┌─────────────────────────────────────┐
                    │           PHASE 4                    │
                    │    Integration & Deployment          │
                    │  (docker-compose, .env, README)      │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        ┌───────────▼───────────┐     ┌────────────▼────────────┐
        │       PHASE 3         │     │     PHASE 3 (parallel)  │
        │  Gateway Server       │     │   Integration Tests     │
        │  (depends on handler  │     │   (depends on gateway   │
        │   API contracts)      │     │    + handlers running)  │
        └───────────┬───────────┘     └─────────────────────────┘
                    │
                    │ Depends on knowing handler I/O format
                    │
    ┌───────────────┴───────────────┬───────────────────────────┐
    │                               │                           │
┌───▼───────────────┐    ┌──────────▼──────────┐    (no dependency)
│    PHASE 1        │    │     PHASE 2         │
│ Classify Handler  │    │  Extract Handler    │
│ (independent)     │    │  (independent)      │
└───────────────────┘    └─────────────────────┘
```

### Build Order

1. **Parallel:** Phase 1 (classify) and Phase 2 (extract) can be built simultaneously
2. **Sequential:** Phase 3 (gateway) requires handler API contracts to be finalized
3. **Sequential:** Phase 4 (integration) requires all components

### Parallelization Opportunities

| Phase | Can Parallelize With |
|-------|---------------------|
| Phase 1 (classify) | Phase 2 (extract) |
| Phase 3 (gateway) | None - needs handler contracts |
| Phase 4 (integration) | None - needs all components |

---

## 3. Phased Implementation Plan

### Phase 1: Classification Handler

**Goal:** Deployable RunPod endpoint that classifies document images into types (W2, 1099, bank_statement, etc.)

**Files to Create:**

| File | Acceptance Criteria |
|------|---------------------|
| `handlers/classify/requirements.txt` | Contains: runpod>=1.6.0, torch>=2.1.0, transformers>=4.40.0, accelerate, qwen-vl-utils, requests, pillow |
| `handlers/classify/handler.py` | - Loads Qwen2.5-VL-7B-Instruct on startup<br>- Fetches images from URLs<br>- Returns JSON: `{type, confidence, reasoning, latency_ms}`<br>- Handles errors gracefully |
| `handlers/classify/test_handler.py` | - Tests `parse_json_response()` with various formats<br>- Tests `validate_classification_result()` edge cases<br>- All tests pass locally |
| `handlers/classify/Dockerfile` | - Based on runpod/pytorch:2.1.0-py3.10-cuda11.8.0<br>- Installs dependencies<br>- CMD runs handler.py |

**Test Strategy:**
1. Run `python test_handler.py` locally
2. Build Docker image: `docker build -t test-classify .`
3. Test with RunPod local testing mode (if available)

**Effort:** Medium (M) - ~3-4 hours

---

### Phase 2: Extraction Handler

**Goal:** Deployable RunPod endpoint that extracts structured data from documents using dots.ocr

**Files to Create:**

| File | Acceptance Criteria |
|------|---------------------|
| `handlers/extract/requirements.txt` | Contains: runpod>=1.6.0, httpx>=0.26.0, requests, pillow |
| `handlers/extract/start.sh` | - Starts vLLM server with dots.ocr model<br>- Waits for health check<br>- Launches handler.py |
| `handlers/extract/handler.py` | - Calls local vLLM server via OpenAI API<br>- Processes multiple pages<br>- Returns structured data with confidence |
| `handlers/extract/transformers.py` | - `transform_to_document_schema(doc_type, elements, text)`<br>- Implementations for: bank_statement, W2, generic<br>- Extracts amounts, dates, SSNs with regex |
| `handlers/extract/test_handler.py` | - Tests `parse_dots_ocr_output()`<br>- Tests schema transformers<br>- Tests amount/date extraction |
| `handlers/extract/Dockerfile` | - Based on vllm/vllm-openai:v0.11.0<br>- Installs handler dependencies<br>- CMD runs start.sh |

**Test Strategy:**
1. Run `python test_handler.py` for unit tests
2. Build and run container locally with GPU
3. Test vLLM health endpoint responds

**Effort:** Large (L) - ~4-6 hours (more complex due to vLLM integration)

---

### Phase 3: Gateway Server

**Goal:** FastAPI server that authenticates requests and proxies to RunPod endpoints

**Files to Create:**

| File | Acceptance Criteria |
|------|---------------------|
| `gateway/requirements.txt` | Contains: fastapi>=0.109.0, uvicorn[standard], httpx, pydantic>=2.5.0 |
| `gateway/app/__init__.py` | Empty package marker |
| `gateway/app/config.py` | - `Settings` dataclass with all config fields<br>- `get_settings()` returns cached instance<br>- Reads from environment variables |
| `gateway/app/services/runpod_client.py` | - `RunPodClient` class with async methods<br>- `classify()`, `extract()`, `health_check()`<br>- Retry logic with exponential backoff |
| `gateway/app/middleware/auth.py` | - `create_auth_dependency(api_key)` factory<br>- Constant-time comparison<br>- Returns 401 on invalid key |
| `gateway/app/middleware/logging.py` | - `RequestLoggingMiddleware` class<br>- Logs request/response with timing<br>- Adds X-Request-ID header |
| `gateway/app/routes/health.py` | - GET /health (checks RunPod endpoints)<br>- GET /ready, GET /live (simple probes) |
| `gateway/app/routes/classify.py` | - POST /v1/classify<br>- Validates ClassifyRequest<br>- Returns ClassifyResponse |
| `gateway/app/routes/extract.py` | - POST /v1/extract<br>- Validates ExtractRequest<br>- Returns ExtractResponse |
| `gateway/app/main.py` | - `create_app()` factory<br>- Registers middleware and routes<br>- Auth on /v1/* routes only |
| `gateway/Dockerfile` | - Based on python:3.11-slim<br>- Non-root user<br>- Healthcheck configured |
| `gateway/tests/test_routes.py` | - Tests with mocked RunPod client<br>- Tests auth rejection<br>- Tests request validation |

**Test Strategy:**
1. Run `pytest gateway/tests/` with mocked RunPod
2. Start locally: `uvicorn app.main:app --reload`
3. Test with curl: health, classify, extract

**Effort:** Medium (M) - ~3-4 hours

---

### Phase 4: Integration & Deployment

**Goal:** Complete deployable system with all components wired together

**Files to Create:**

| File | Acceptance Criteria |
|------|---------------------|
| `docker-compose.yml` | - Gateway service with env vars<br>- Port 8000 exposed<br>- Health check configured |
| `.env.example` | - All required variables documented<br>- Placeholder values for secrets |
| `tests/test_integration.py` | - Tests against live gateway<br>- Tests classify and extract endpoints<br>- Tests auth rejection |
| `README.md` | - Quick start instructions<br>- Deployment steps for RunPod<br>- API reference |

**Test Strategy:**
1. Deploy handlers to RunPod (manual via console)
2. Run gateway locally with real RunPod credentials
3. Run `pytest tests/test_integration.py`
4. Verify end-to-end with real document images

**Effort:** Small (S) - ~1-2 hours

---

## 4. Implementation Prompts

### Prompt 1: Classification Handler

```markdown
# Task: Implement Classification Handler for RunPod

## Context
You are implementing a RunPod Serverless handler that classifies document images
using Qwen2.5-VL-7B-Instruct. This handler will receive ShareFile pre-signed URLs
and return document type classifications.

## Source Reference
See RUNPOD_IMPLEMENTATION.md sections:
- Section 4.1: Directory Structure
- Section 4.2: Dockerfile
- Section 4.3: Handler Implementation
- Section 4.4: Requirements
- Section 4.5: Local Testing

## Files to Create
Create these files in `handlers/classify/`:

1. **requirements.txt** - Dependencies:
   - runpod>=1.6.0
   - torch>=2.1.0
   - transformers>=4.40.0
   - accelerate>=0.27.0
   - qwen-vl-utils>=0.0.2
   - requests>=2.31.0
   - pillow>=10.0.0

2. **handler.py** - Main handler with:
   - Global MODEL, PROCESSOR loaded once at cold start
   - `fetch_image(url)` - Downloads image from URL, returns PIL Image
   - `build_classification_prompt()` - Returns the classification prompt
   - `parse_json_response(text)` - Extracts JSON from model output (handles code blocks)
   - `validate_classification_result(result)` - Normalizes output, enforces valid types
   - `classify(image_urls, prompt)` - Main inference function
   - `handler(job)` - RunPod entry point

3. **test_handler.py** - Unit tests for:
   - `parse_json_response()` with markdown blocks, raw JSON, extra text
   - `validate_classification_result()` with invalid types, out-of-range confidence

4. **Dockerfile** - Based on runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

## Acceptance Criteria
- [ ] `python test_handler.py` passes all tests
- [ ] Docker image builds successfully
- [ ] Handler returns: `{"type": "W2", "confidence": 0.95, "reasoning": "...", "latency_ms": 1234}`
- [ ] Valid document types: W2, 1099-INT, 1099-DIV, 1099-MISC, 1099-NEC, 1099-R, 1098,
      bank_statement, credit_card_statement, invoice, receipt, check, other
- [ ] Errors return: `{"error": "message"}`

## Notes
- Use transformers library directly (not vLLM) for classification
- Temperature should be low (0.1) for deterministic output
- Only process first page for classification (image_urls[0])
```

---

### Prompt 2: Extraction Handler

```markdown
# Task: Implement Extraction Handler for RunPod

## Context
You are implementing a RunPod Serverless handler that extracts structured data from
documents using dots.ocr (rednote-hilab/dots.ocr). This uses vLLM 0.11.0+ which has
native dots.ocr support. The handler runs vLLM as a local server and calls it via
the OpenAI-compatible API.

## Source Reference
See RUNPOD_IMPLEMENTATION.md sections:
- Section 5.1: Directory Structure
- Section 5.2: Dockerfile
- Section 5.3: Startup Script
- Section 5.4: Handler Implementation
- Section 5.5: Document Schema Transformers

## Files to Create
Create these files in `handlers/extract/`:

1. **requirements.txt** - Dependencies:
   - runpod>=1.6.0
   - httpx>=0.26.0
   - requests>=2.31.0
   - pillow>=10.0.0

2. **start.sh** - Startup script that:
   - Launches vLLM server in background with dots.ocr model
   - Waits for /health endpoint to respond
   - Starts handler.py
   - Make executable (chmod +x)

3. **handler.py** - Main handler with:
   - `fetch_image(url)` - Downloads and returns PIL Image
   - `image_to_base64(image)` - Converts PIL to base64
   - `call_vllm(image_b64, prompt_mode, max_tokens)` - Async call to local vLLM
   - `parse_dots_ocr_output(raw_output)` - Parse structured output
   - `extract(image_urls, doc_type, prompt_mode)` - Process all pages
   - `handler(job)` - RunPod entry point (uses asyncio.run)

4. **transformers.py** - Schema converters:
   - `extract_amounts(text)` - Find monetary values with regex
   - `extract_dates(text)` - Find date patterns
   - `transform_bank_statement(elements, raw_text)` - Bank statement schema
   - `transform_w2(elements, raw_text)` - W2 form schema
   - `transform_to_document_schema(doc_type, elements, raw_text)` - Dispatcher

5. **test_handler.py** - Unit tests for:
   - `parse_dots_ocr_output()` with various formats
   - `extract_amounts()` and `extract_dates()`
   - Schema transformer output structure

6. **Dockerfile** - Based on vllm/vllm-openai:v0.11.0

## Acceptance Criteria
- [ ] `python test_handler.py` passes all tests
- [ ] start.sh correctly waits for vLLM before starting handler
- [ ] Docker image builds successfully
- [ ] Handler processes multiple pages
- [ ] Returns: `{"data": {...}, "raw_ocr": {...}, "confidence": {...}, "page_count": N, "latency_ms": N}`
- [ ] Errors return: `{"error": "message"}`

## Notes
- vLLM server runs on localhost:8000 inside the container
- Use httpx for async HTTP calls to vLLM
- Temperature should be 0.0 for deterministic OCR
- dots.ocr uses OpenAI-compatible chat/completions API
```

---

### Prompt 3: Gateway Server

```markdown
# Task: Implement Gateway Server

## Context
You are implementing a FastAPI gateway that authenticates requests and proxies them
to RunPod serverless endpoints. The gateway provides a stable API for the document
processing pipeline while hiding RunPod implementation details.

## Source Reference
See RUNPOD_IMPLEMENTATION.md sections:
- Section 6.1: Directory Structure
- Section 6.2: Configuration
- Section 6.3: RunPod Client Service
- Section 6.4-6.5: Middleware (auth, logging)
- Section 6.6-6.8: Routes (classify, extract, health)
- Section 6.9-6.11: Main app, Dockerfile, requirements

## Files to Create
Create these files in `gateway/`:

1. **requirements.txt**:
   - fastapi>=0.109.0
   - uvicorn[standard]>=0.27.0
   - httpx>=0.26.0
   - pydantic>=2.5.0

2. **app/__init__.py** - Empty

3. **app/config.py** - Settings:
   - `@dataclass Settings` with all config fields
   - `from_env()` class method
   - `@lru_cache get_settings()` function

4. **app/services/__init__.py** - Empty

5. **app/services/runpod_client.py**:
   - `@dataclass RunPodResponse` - success, data, error, latency_ms, job_id
   - `class RunPodClient` - async client with retry logic
   - Methods: classify(), extract(), health_check(), close()

6. **app/middleware/__init__.py** - Empty

7. **app/middleware/auth.py**:
   - `create_auth_dependency(valid_api_key)` - returns FastAPI dependency
   - Uses secrets.compare_digest for timing-safe comparison

8. **app/middleware/logging.py**:
   - `class RequestLoggingMiddleware(BaseHTTPMiddleware)`
   - Logs request/response with timing and request ID

9. **app/routes/__init__.py** - Empty

10. **app/routes/health.py**:
    - GET /health - full health check with RunPod status
    - GET /ready - simple readiness probe
    - GET /live - simple liveness probe

11. **app/routes/classify.py**:
    - `ClassifyRequest` and `ClassifyResponse` Pydantic models
    - POST /v1/classify endpoint

12. **app/routes/extract.py**:
    - `ExtractRequest` and `ExtractResponse` Pydantic models
    - POST /v1/extract endpoint

13. **app/main.py**:
    - `create_app()` factory function
    - Register middleware and routes
    - Auth on /v1/* routes, not on health routes

14. **Dockerfile** - Python 3.11 slim with non-root user

15. **tests/test_routes.py** - Tests with mocked RunPod client

## Acceptance Criteria
- [ ] `pytest gateway/tests/` passes
- [ ] `uvicorn app.main:app` starts successfully
- [ ] GET /health returns endpoint status
- [ ] POST /v1/classify returns 401 without X-API-Key
- [ ] POST /v1/classify returns classification with valid key
- [ ] Request logging shows timing and request ID

## Environment Variables Required
- GATEWAY_API_KEY
- RUNPOD_API_KEY
- RUNPOD_CLASSIFY_ENDPOINT
- RUNPOD_EXTRACT_ENDPOINT
```

---

### Prompt 4: Integration & Deployment

```markdown
# Task: Complete Integration and Deployment Configuration

## Context
You are creating the final integration pieces: docker-compose for local development,
environment template, integration tests, and documentation.

## Source Reference
See RUNPOD_IMPLEMENTATION.md sections:
- Section 7.1-7.5: Deployment (build, RunPod console, docker-compose, env vars)
- Section 8.1-8.3: Testing (gateway, RunPod direct, integration tests)
- Section 1.2-1.3: Overview and repository structure

## Files to Create

1. **docker-compose.yml**:
   - Gateway service on port 8000
   - Environment variables from .env
   - Health check
   - restart: unless-stopped

2. **.env.example**:
   - GATEWAY_API_KEY=your-secure-api-key-here
   - RUNPOD_API_KEY=your-runpod-api-key
   - RUNPOD_CLASSIFY_ENDPOINT=abc123xyz
   - RUNPOD_EXTRACT_ENDPOINT=def456uvw
   - RUNPOD_TIMEOUT=120
   - LOG_LEVEL=INFO

3. **tests/test_integration.py**:
   - pytest fixtures for client setup
   - test_health() - health endpoint works
   - test_classify_w2() - classification returns expected type
   - test_extract_w2() - extraction returns structured data
   - test_extract_bank_statement() - bank extraction works
   - test_invalid_api_key() - returns 401

4. **README.md**:
   - Project overview (3 components)
   - Quick start (local development)
   - RunPod deployment steps
   - API reference (endpoints, request/response formats)
   - Environment variables
   - Troubleshooting

## Acceptance Criteria
- [ ] `docker-compose up` starts gateway successfully
- [ ] Integration tests pass against deployed endpoints
- [ ] README provides complete setup instructions
- [ ] .env.example documents all required variables

## Deployment Checklist (for README)
1. Build handler Docker images
2. Push to container registry
3. Create RunPod endpoints via console
4. Note endpoint IDs
5. Deploy gateway (compose/Cloud Run/Railway)
6. Configure pipeline to use gateway URL
```

---

## 5. Integration Checklist

### Prerequisites
- [ ] RunPod account with API key
- [ ] Container registry (Docker Hub, GHCR, etc.)
- [ ] ShareFile access for document image URLs

### Handler Deployment
- [ ] Build classify handler image
- [ ] Push to registry
- [ ] Create RunPod endpoint for classify
- [ ] Record classify endpoint ID
- [ ] Build extract handler image
- [ ] Push to registry
- [ ] Create RunPod endpoint for extract
- [ ] Record extract endpoint ID
- [ ] Test both endpoints directly via RunPod API

### Gateway Deployment
- [ ] Configure environment variables
- [ ] Deploy gateway (compose/cloud)
- [ ] Verify /health returns healthy status
- [ ] Test /v1/classify with real image
- [ ] Test /v1/extract with real image

### Pipeline Integration
- [ ] Add gateway URL to pipeline config
- [ ] Add gateway API key to pipeline secrets
- [ ] Update classify_worker to call gateway /v1/classify
- [ ] Update extract_worker to call gateway /v1/extract
- [ ] Test end-to-end with real documents

### Production Readiness
- [ ] Configure RunPod worker scaling (max_workers, idle_timeout)
- [ ] Enable FlashBoot for faster cold starts
- [ ] Set up monitoring/alerting for gateway
- [ ] Document rollback procedure
- [ ] Create runbook for common issues

---

## Appendix: Critical Path

The minimum path to first end-to-end test:

```
1. Phase 1: Classify handler (build + deploy)     ─┐
                                                    ├─► 2. Phase 3: Gateway
2. Phase 2: Extract handler (build + deploy)      ─┘        │
                                                            ▼
                                                  3. Phase 4: Integration test
```

**Time to first e2e test:** ~8-12 hours of implementation work

**Critical dependencies:**
1. RunPod account setup (manual, ~30 min)
2. At least one handler deployed to RunPod
3. Gateway running with correct endpoint IDs
