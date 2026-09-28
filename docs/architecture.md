# KYC Engine Architecture

## Scope

The core processes exactly one `ao_id_card/front/v1` document in memory and returns text fields. It is a modular monolith: stage responsibilities are separate, while orchestration remains one synchronous Python call.

Authenticity, forgery detection, thresholded face matching, final biometric decisions, storage, queues, HTTP APIs, frontends, and multi-card selection are outside this core.

## Processing Stages

| Stage | Owns | Explicit boundary |
| --- | --- | --- |
| Intake | Bounded path reads, JPEG/PNG preflight, EXIF orientation, decode, BGR conversion. | Never retains source paths or arbitrary metadata. |
| Detection | One-card location, document type/side, confidence, semantic corners, model provenance. | Never claims authenticity; zero or multiple cards fail. |
| Normalization | Corner validation, perspective transform, canonical crop, transform matrices. | Never performs OCR or changes profile geometry. |
| Variants | At most ten balanced preprocessing alternatives. | Never ranks variants or mutates the canonical image. |
| Quality | Sharpness, brightness, contrast, range, and clipping measurements. | No universal score; only constant output is gated initially. |
| Fields | Profile box padding, bounds checks, immutable crops. | Never guesses unconfigured fields. |
| OCR | Local recognition-only or detection-plus-recognition inference. | Never logs or reconciles recognized text. |
| Reconciliation | Candidate grouping, deterministic selection, alternatives, issues. | Never combines characters or invents a value. |
| Validation | Conservative normalization and explicit value states. | Never silently corrects ambiguous input. |
| Pipeline | Stage order, failure isolation, timings, structured result assembly. | No persistence or presentation concerns. |

## Data Flow

```text
Path | bytes | NDArray[uint8]
  -> InputImage
  -> DetectionResult
  -> NormalizedDocument
  -> tuple[VariantBatch, ...]
  -> tuple[VariantBatchAssessment, ...]
  -> tuple[FieldCrop, ...]
  -> tuple[OCRCandidate, ...]
  -> mapping[str, ExtractedField]
  -> KycExtractionResult
```

For a two-sided document, `DocumentCoordinator` runs the front pipeline first and
the back pipeline second when those sources are supplied. It preserves each side's
result as-is and assembles them into a nested `DocumentExtractionResult`. The
coordinator does not auto-detect side, compare front/back values, or reconcile QR
data with OCR data.

Image-bearing contracts hold read-only arrays. Intake copies caller arrays, generators produce new arrays, assessors receive copies, and serialized results contain no pixels.

## Package Layout

```text
pyproject.toml
src/kyc_engine/
  contracts.py
  intake.py
  detection.py
  onnx_detection.py
  normalization.py
  variants/
  quality/
  fields.py
  ocr.py
  reconciliation.py
  validation.py
  coordinator.py
  profiles.py
  pipeline.py
  defaults.py
  configs/ao_id_card_front_v1.json
tests/unit/
tests/integration/
tests/fixtures/synthetic/
```

Only genuinely replaceable edges use protocols: `DocumentDetector` and `TextRecognizer`. Internal stages are concrete classes configured by the composition helpers in `defaults.py`.

## Detector Paths

`OpenCVDocumentDetector` is deterministic development infrastructure based on edges, contours, card aspect ratio, area, rectangularity, and corner angles. It rejects zero and multiple plausible cards. Geometry alone cannot reliably resolve a 180-degree card orientation.

`OnnxDocumentDetector` consumes a checksum-pinned box model and semantic four-corner model. Semantic corners remove the 180-degree ambiguity. It is not the default until private held-out evaluation demonstrates that it beats the classical baseline and the required weights are provisioned. There is no silent fallback between detector types.

## Profile Boundary

The versioned profile supplies canonical dimensions, field boxes, padding, required state, OCR mode, comparison rule, normalizer, and validator. Registry loading rejects unsupported strategies, duplicate fields, invalid dimensions, and out-of-bounds boxes.

The current profile is intentionally marked `provisional`. Production extraction acceptance criteria remain blocked until the complete front-side inventory and geometry are confirmed from authoritative sources.

## Library Boundary

`src/kyc_engine` is the sole supported implementation and installable package.
The repository-level `tests/` suite is the compatibility boundary for all public
contracts. Diagnostic scripts under `scripts/` are development tools and are not
part of the library API.

## HTTP Service Boundary

The separate `api/` project depends on the installed library and owns HTTP
authentication, bounded multipart intake, session state, background job
execution, expiry, and HTTP error mapping. It passes in-memory bytes to
`DocumentCoordinator` and projects its internal result into a stable,
PII-minimal public result only at the authenticated result endpoint. It does
not duplicate extraction stages.

The first API deployment is deliberately single-process. Its session store and
job queue are in memory, so multiple server workers would create isolated state.

When explicitly enabled, the API composes the optional MiniFASNet adapter into
the model-independent `LivenessEvaluator`, then into `VerificationManager`.
After both document captures are accepted, document processing and liveness can
advance independently. The API accepts a bounded, configured number of
ephemeral JPEG/PNG liveness frames; it persists only the safe aggregate
`LivenessResult`, never source frames, crops, tensors, or reference images.
When explicitly configured, an internal local recognizer automatically derives
a raw cosine similarity from the two eligibility-gated artifacts after document
and liveness pass. A locked lifecycle advances face matching through `ready`,
`processing`, and `completed` (or `failed`), then releases both biometric
artifacts. The score is ephemeral internal metadata and never appears in public
API or webhook payloads; it does not apply a threshold or make an identity
decision. Public technical completion is derived only after every check required
for that session completes successfully. A deliberately unconfigured comparison
is reported as unavailable and excluded from that aggregation. The passive
policy is an initial uncalibrated development default. See
[the public API contract](api-contract.md) for the external projection.

## API Service Layout

`api/src/kyc_api` follows the existing runtime boundaries. The engine remains
an independent installable library; the API never moves its HTTP or session
concerns into `kyc_engine`.

- `application/sessions/store.py` contains the current in-memory `SessionStore`
  as one cohesive lifecycle/storage implementation. Its locks, mutable records,
  expiry tombstones, claims, and biometric artifact ownership remain together.
- `application/verification`, `application/jobs`, and
  `application/orchestration` own the existing workflow coordination. They do
  not import FastAPI or the MINFIN/Playwright runtime.
- `domain` holds only provider-independent values: verification enums and the
  NIF verifier/result contract. Legacy `kyc_api.models` re-exports the exact
  same enum objects rather than redeclaring them.
- `infrastructure` contains concrete rate limiting, observability, SQLite
  webhook delivery, and the MINFIN adapter. Only that adapter imports the
  external `nif_checker` package, which retains Playwright portal ownership.
- `http` contains FastAPI authentication, middleware, public schemas, and
  response projection. Existing route handlers remain in the application
  factory because they share its established injection and lifespan behavior;
  they access sessions only through public methods.
- `composition` constructs engine/model dependencies. `app.py` retains the
  application factory and lifecycle ownership, while `main.py` preserves the
  deployed `kyc_api.main:app` entrypoint.

The hosted verifier stays in `kyc_api/verify` because its Vite, Docker, wheel
package-data, and static-file paths are deployment contracts. The repository
`web/` Go sandbox and `nif-checker/` package likewise remain separate.

## Persistence Boundary

`application.sessions.SessionStore` is the current process-local lifecycle and
storage implementation. It is not a pure domain repository. A future
PostgreSQL task may replace it or introduce a persistence abstraction at this
application boundary; this repository has no database models, migrations, SQL,
or persistence adapter.
