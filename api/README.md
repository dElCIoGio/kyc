# Angolan KYC API

This project exposes the installed `angolan-kyc-engine` library through a
session-based FastAPI service. It accepts explicitly labelled front and back
images, runs extraction in a bounded background thread, and retains the
structured result in memory until the session expires or is deleted.

## Installation

From the repository root:

```powershell
python -m pip install -e ".[ocr]"
python -m pip install -e "api[dev]"
```

The OCR model must already exist locally. The service never downloads models.

## Configuration

Required environment variables:

```powershell
$env:KYC_API_KEY = "replace-with-a-long-random-secret"
$env:KYC_OCR_MODEL_MANIFEST = "private-models\paddleocr\ao-id-front-v5-mobile\manifest.json"
```

Optional settings and defaults:

```text
KYC_OCR_DEVICE=cpu
KYC_LOG_LEVEL=INFO
KYC_ENVIRONMENT=development
KYC_MAX_UPLOAD_BYTES=15728640
KYC_SESSION_TTL_SECONDS=1800
KYC_MAX_SESSIONS=100
KYC_JOB_WORKERS=1
KYC_RATE_LIMIT_REQUESTS=120
KYC_RATE_LIMIT_WINDOW_SECONDS=60
KYC_JOB_TIMEOUT_SECONDS=30
KYC_SESSION_CLEANUP_INTERVAL_SECONDS=60
```

Start the service from the repository root:

```powershell
uvicorn kyc_api.main:app --app-dir api/src --host 127.0.0.1 --port 8000 --no-access-log
```

## Docker Compose

The repository-root Compose project packages this API and the installed engine
in one container. It passes root `.env` values into the service and mounts the
ignored `private-models/` directory read-only at `/models`; the mounted model
manifest path is supplied automatically to the container. The mounted files
are the persistent model cache across container restarts. The build may fetch
Python dependency wheels once, but PaddleOCR model weights are never downloaded
by the service: missing or invalid local model directories fail startup.

```powershell
Copy-Item .env.example .env
# Configure KYC_API_KEY and place verified artifacts under private-models/.
docker compose up --build
```

Use `docker compose down` to stop the service. The service binds container port
`8000`; set `KYC_API_PORT` in root `.env` to choose a different host port.
Never place model files, real document images, or API keys in the image build
context or Git.

The application validates the OCR manifest and initializes both side-specific
pipelines during startup. Startup fails when configuration or model checks fail.
`KYC_API_KEY` must be injected by the local environment or deployment secret
store; never commit it. The complete multipart request body is bounded to the
configured image limit plus 64 KiB of multipart overhead.

## Application Logging

Application logs are emitted to stdout as one JSON object per line, suitable for
Docker and Railway collection. Every record includes a timestamp, level, logger,
service, environment, and event; job lifecycle events may also include existing
opaque session/job IDs, sides, status, and duration.

Set `KYC_LOG_LEVEL` to a standard Python logging level (for example `DEBUG` or
`WARNING`) and `KYC_ENVIRONMENT` to the deployment name. Logs never include
request bodies, filenames, image bytes, OCR text, extracted values, QR payloads,
or serialized results. Unexpected failures record the exception type and safe
stack frames internally, but not exception-message text; API responses retain
their existing generic error codes and messages.

## Session Workflow

All `/v1/*` calls require the `X-API-Key` header. `/healthz` is public and never
returns document data.

Create a session:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/sessions `
  -H "X-API-Key: $env:KYC_API_KEY"
```

Upload each available side using the returned session ID:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/sessions/SESSION_ID/images/front `
  -H "X-API-Key: $env:KYC_API_KEY" `
  -F "image=@private-data/ao-id-front/front.jpeg;type=image/jpeg"

curl.exe -X POST http://127.0.0.1:8000/v1/sessions/SESSION_ID/images/back `
  -H "X-API-Key: $env:KYC_API_KEY" `
  -F "image=@private-data/ao-id-back/back.jpeg;type=image/jpeg"
```

Queue extraction and poll status:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/sessions/SESSION_ID/process `
  -H "X-API-Key: $env:KYC_API_KEY"

curl.exe http://127.0.0.1:8000/v1/sessions/SESSION_ID `
  -H "X-API-Key: $env:KYC_API_KEY"
```

Retrieve the structured result after the state becomes `success`, `partial`, or
`failed`:

```powershell
curl.exe http://127.0.0.1:8000/v1/sessions/SESSION_ID/result `
  -H "X-API-Key: $env:KYC_API_KEY"
```

Delete retained session state explicitly:

```powershell
curl.exe -X DELETE http://127.0.0.1:8000/v1/sessions/SESSION_ID `
  -H "X-API-Key: $env:KYC_API_KEY"
```

## Monitoring and Limits

`GET /v1/metrics` requires `X-API-Key` and returns only process-lifetime
aggregate response counts, safe error codes, job outcomes, and latency summaries.
It never includes session IDs, filenames, image data, OCR values, or extracted
fields. Metrics reset when the process restarts.

Authenticated `/v1/*` requests are globally limited to 120 requests per
60-second window by default. A limited request receives `429 RATE_LIMITED` and a
`Retry-After` header. Jobs have a 30-second soft deadline by default: their
session is safely failed and cleared on timeout, while an already-running native
OCR thread is allowed to finish before its worker slot is released.

## Data Handling

Uploads, session state, and extraction results are memory-only. Image bytes are
removed from the session immediately after processing and all retained state is
removed on deletion or expiry. Sessions do not survive process restarts. Deleting
a currently running session removes its tracked state but cannot interrupt native
OCR work that has already started.

The first version is intended for one trusted, single-process deployment. Running
multiple Uvicorn workers creates independent session stores; use exactly one
worker until a shared store and external queue are introduced.

Keep Uvicorn access logs disabled with `--no-access-log`. The application does
not log request bodies, filenames, source paths, image data, OCR output, or
serialized extraction results.

## Tests

```powershell
python -m unittest discover -s api/tests -p "test_*.py" -v
```
