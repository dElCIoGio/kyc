# KYC Engine Architecture

## Scope

The core processes exactly one `ao_id_card/front/v1` document in memory and returns text fields. It is a modular monolith: stage responsibilities are separate, while orchestration remains one synchronous Python call.

Authenticity, forgery detection, liveness, biometrics, storage, queues, HTTP APIs, frontends, and multi-card selection are outside this core.

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

## Legacy Components

`app/workers/image_processor` and `app/field-extractor` remain as temporary compatibility surfaces. The root package is now canonical for new development. Their test suites continue to run until downstream imports and the old image-processing CLI are retired deliberately.
