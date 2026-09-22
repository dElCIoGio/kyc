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
KYC_MAX_UPLOAD_BYTES=15728640
KYC_SESSION_TTL_SECONDS=1800
KYC_MAX_SESSIONS=100
KYC_JOB_WORKERS=1
```

Start the service from the repository root:

```powershell
uvicorn kyc_api.main:app --app-dir api/src --host 127.0.0.1 --port 8000 --no-access-log
```

The application validates the OCR manifest and initializes both side-specific
pipelines during startup. Startup fails when configuration or model checks fail.

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

## Data Handling

Uploads, session state, and extraction results are memory-only. Image bytes are
removed from the session immediately after processing and all retained state is
removed on deletion or expiry. Sessions do not survive process restarts. Deleting
a currently running session removes its tracked state but cannot interrupt native
OCR work that has already started.

The first version is intended for one trusted, single-process deployment. Running
multiple Uvicorn workers creates independent session stores; use exactly one
worker until a shared store and external queue are introduced.

## Tests

```powershell
python -m unittest discover -s api/tests -p "test_*.py" -v
```
