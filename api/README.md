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
KYC_CAPTURE_MIN_SHARPNESS=25.0
KYC_CAPTURE_MIN_BRIGHTNESS=35.0
KYC_CAPTURE_MAX_BRIGHTNESS=220.0
KYC_CAPTURE_MIN_CONTRAST=12.0
KYC_SESSION_TTL_SECONDS=1800
KYC_MAX_SESSIONS=100
KYC_JOB_WORKERS=1
KYC_RATE_LIMIT_REQUESTS=120
KYC_RATE_LIMIT_WINDOW_SECONDS=60
KYC_JOB_TIMEOUT_SECONDS=30
KYC_SESSION_CLEANUP_INTERVAL_SECONDS=60
KYC_OTEL_ENABLED=false
KYC_OTEL_ENDPOINT=
KYC_OTEL_HEADERS=
KYC_OTEL_TRACE_SAMPLE_RATIO=1.0
KYC_OTEL_METRIC_EXPORT_INTERVAL_SECONDS=60
KYC_OTEL_EXPORT_TIMEOUT_SECONDS=10
```

Start the service from the repository root:

```powershell
uvicorn kyc_api.main:app --app-dir api/src --host 127.0.0.1 --port 8000 --no-access-log
```

## Docker Compose

The repository-root Compose project packages this API, the installed engine,
and a Go + HTMX operator console. It passes root `.env` values into the API and
mounts the ignored `private-models/` directory read-only at `/models`; the
mounted model manifest path is supplied automatically to the container. The
mounted files are the persistent model cache across container restarts. The
build may fetch Python dependency wheels once, but PaddleOCR model weights are
never downloaded by the service: missing or invalid local model directories
fail startup.

```powershell
Copy-Item .env.example .env
# Configure KYC_API_KEY and place verified artifacts under private-models/.
docker compose up --build
```

Open the operator console at `http://127.0.0.1:8080`; set `KYC_WEB_PORT` in the
root `.env` to choose a different host port. The API's port `8000` is private to
the Compose network, and the console proxies requests with the API key held
server-side. Never place model files, real document images, or API keys in the
image build context or Git.

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

Every HTTP response includes a server-generated `X-Request-ID`; client-provided
values are ignored. `request_id` identifies one HTTP request, `session_id` is
the existing in-memory KYC session, and `job_id` is one extraction job. Background
work correlates using `session_id` and `job_id`, not the original request ID, so
operators can search a failed extraction's logs by job ID. These opaque IDs carry
no KYC data.

Each pipeline side emits `pipeline_stage_started`, `pipeline_stage_completed`,
or `pipeline_stage_failed` for `intake`, `detection`, `profile`,
`normalization`, `variants`, `quality`, `qr` (when configured),
`field_localization`, `ocr`, and `reconciliation`. Filter by `job_id`, then read
the ordered events and `duration_ms` values to identify the slow or failed stage.

Each background extraction is also one OpenTelemetry trace: `kyc.process_job`
contains supplied `kyc.process_side` spans, which contain the same semantic
`kyc.stage` boundaries listed above. Active structured logs include the matching
lowercase hexadecimal `trace_id` and `span_id`, so operators can move between a
job-ID log search and a trace view. Trace attributes contain only job/session
IDs, side, stage, processing status, and safe exception types; they never carry
KYC values. The engine remains usable without an SDK provider or exporter.

## Operational metrics

`/v1/metrics` keeps its existing process-local aggregate response, including
millisecond request/job latency summaries. In parallel, the API records standard
OpenTelemetry metrics: `kyc.jobs`, `kyc.job.duration` (seconds),
`kyc.stage.executions`, `kyc.stage.failures`, `kyc.stage.duration` (seconds),
and `kyc.field.status`. Their only dimensions are bounded status, stage, side
(`front`, `back`, or `unknown` for standalone engine work), and trusted profile
field names. Field metrics contain a schema field name and final status only,
never its value. Correlation IDs, trace IDs, filenames, exception messages, and
all KYC data are prohibited from metric attributes.

## OTLP export

Logs stay as structured JSON on stdout; this application does not export logs
through OpenTelemetry. Traces and metrics can independently be exported using
standard OTLP HTTP/protobuf by setting `KYC_OTEL_ENABLED=true`. The application
treats `KYC_OTEL_ENDPOINT` as a base URL and derives `/v1/traces` and
`/v1/metrics`, so do not provide a signal-specific URL. Only absolute HTTP(S)
base URLs without embedded credentials, query strings, or fragments are accepted.

```ini
KYC_OTEL_ENABLED=true
KYC_OTEL_ENDPOINT=https://collector.example.com
KYC_OTEL_HEADERS=Authorization=Bearer <secret>
KYC_OTEL_TRACE_SAMPLE_RATIO=1.0
KYC_OTEL_METRIC_EXPORT_INTERVAL_SECONDS=60
KYC_OTEL_EXPORT_TIMEOUT_SECONDS=10
```

`KYC_OTEL_HEADERS` is secret configuration: it is passed only to the OTLP
exporters and is never logged, traced, or exposed by the API. Export is disabled
by default and requires no endpoint or network access. When enabled, traces use
a `BatchSpanProcessor` with parent-based ratio sampling; metrics use a periodic
exporting reader. The interval and timeout are seconds, and the timeout bounds
both exporter requests and shutdown flushing. A temporary backend outage cannot
change KYC processing results because export is asynchronous and outside the
pipeline/job path. The same environment variables work in Docker, VMs, or any
container platform; no hosting-provider SDK or deployment-specific behavior is
used. When a trace is not sampled, its spans are non-recording and the existing
JSON formatter omits `trace_id` and `span_id`; operators can still use the
request/session/job IDs in logs.

Set `KYC_LOG_LEVEL` to a standard Python logging level (for example `DEBUG` or
`WARNING`) and `KYC_ENVIRONMENT` to the deployment name. Logs never include
request bodies, filenames, image bytes, OCR text, extracted values, QR payloads,
or serialized results. Unexpected failures record the exception type and safe
stack frames internally, but not exception-message text; API responses retain
their existing generic error codes and messages.

## Verification Workflow

All `/v1/*` calls require the `X-API-Key` header. `/healthz` is public and never
returns document data.

Create a session:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/sessions `
  -H "X-API-Key: $env:KYC_API_KEY"
```

Each session is one verification. It begins with document capture awaiting both
sides, liveness blocked, and face matching blocked. Upload the front and back
images using the returned ID. Every upload is assessed before it is retained;
an ordinary quality rejection returns `200` with `accepted: false` and safe
issue codes, so the caller can retry without changing stored captures.

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/sessions/SESSION_ID/images/front `
  -H "X-API-Key: $env:KYC_API_KEY" `
  -F "image=@private-data/ao-id-front/front.jpeg;type=image/jpeg"

curl.exe -X POST http://127.0.0.1:8000/v1/sessions/SESSION_ID/images/back `
  -H "X-API-Key: $env:KYC_API_KEY" `
  -F "image=@private-data/ao-id-back/back.jpeg;type=image/jpeg"
```

When both captures are accepted, the document job is queued automatically and
liveness becomes `ready`. Poll the verification state:

```powershell
curl.exe http://127.0.0.1:8000/v1/sessions/SESSION_ID `
  -H "X-API-Key: $env:KYC_API_KEY"
```

`POST /v1/sessions/SESSION_ID/process` remains available as a safe retry if a
completed capture is left `ready` because bounded job capacity was unavailable.
It returns the existing queued or processing job rather than creating a second
one.

Retrieve the structured result after `document.status` becomes `passed`,
`partial`, or `failed`:

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
