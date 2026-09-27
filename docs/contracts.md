# Core Contracts

The contracts in `src/kyc_engine/contracts.py` are frozen dataclasses. Collections are tuples or immutable mappings, and image arrays are read-only.

## Entry Point

```python
coordinator = build_paddle_document_coordinator(
    model_manifest=Path("local-manifest.json"),
    device="cpu",
)
result = coordinator.process(front=front_source, back=back_source)

ImageSource = Path | str | bytes | NDArray[np.uint8]

For a complete two-sided document, `DocumentCoordinator` accepts explicitly labelled
front and back sources and returns a `DocumentExtractionResult`. Its `front` and
`back` members retain the complete side-specific `KycExtractionResult` values;
fields are not flattened or reconciled across sides. Missing or failed sides make
the document result partial when another side remains usable.
```

Invalid caller types and invalid configuration raise exceptions. Expected image-processing failures return `KycExtractionResult(status="failed")` with a machine-readable issue.

`build_paddle_document_coordinator()` is the supported consumer factory. It
accepts optional per-side detector injection and `IntakeLimits`; its default
OpenCV detectors are development-grade. `KycPipeline`, `build_paddle_pipeline()`,
and `build_balanced_pipeline()` remain advanced composition APIs.

## Image and Geometry Contracts

- `InputImage`: copied BGR pixels plus width, height, channels, decoded format, and random processing ID. It contains no path or EXIF.
- `Point`, `Quadrilateral`, `BoundingBox`: finite geometry in documented coordinate systems. Quadrilateral order is TL, TR, BR, BL.
- `DetectionResult`: type, side, confidence, four corners, detector/version, confidence components, and orientation.
- `NormalizedDocument`: canonical image, profile ID, dimensions, and forward/inverse 3x3 transforms.

## Capture Assessment Contracts

`DocumentCaptureAssessor` is the API-facing, deterministic quality gate before a
capture becomes part of a verification session. It uses `ImageIntake` as its only
decode and validation path, then reports immutable `CaptureAssessment` values:

- `CaptureMetrics`: safe numeric width, height, sharpness, brightness, and contrast.
- `CaptureIssue`: an ordered, PII-safe quality code and message.
- `CaptureAssessment`: accepted flag, issues, and metrics. It never contains source
  bytes or decoded pixels.

The initial gate reports `too_blurry`, `too_dark`, `overexposed`, and
`low_contrast`; it is intentionally not a document detector, OCR, authenticity,
or face-quality check. Its thresholds are initial uncalibrated defaults, not
production-calibrated quality policy.

## Liveness Contracts

`AntiSpoofDetector` is the model adapter boundary: it evaluates one existing
decoded `Image` frame and returns an `AntiSpoofResult`. `LivenessEvaluator`
aggregates ordered frame decisions into the safe public `LivenessResult`.
Its three-frame/two-real policy is an initial uncalibrated development default;
it has no dependency on a specific anti-spoof model.

The API uses its internal live-face evaluation path with the separate
`FaceDetector` localization boundary. It considers only passive-real frames,
requires exactly one sufficiently large face and basic face-region quality, then keeps
at most one readonly original selected frame in process memory. Its deterministic selection
score is 40% relative face size, 30% sharpness, 20% exposure, and 10% face
centering; anti-spoof score is intentionally not a component. No suitable crop
does not alter the public passive-liveness decision. The artifact is pending
until atomically claimed by the active session and is released for failure,
late completion, deletion, or expiry. Artifact IDs, pixels, boxes, paths, and
embeddings are never represented in API JSON. The internal
`resolve_live_face_artifact` boundary returns the selected original frame only
when liveness passed.

## Internal Face Comparison

Face comparison is an internal-only, model-agnostic boundary. It resolves only
the matcher-eligible document portrait and the passed liveness frame, requires
exactly one face from each recognition inference, and returns only a raw finite
cosine similarity. It has no threshold, match decision, API endpoint, session
state transition, or embedding persistence. Embeddings exist only during one
comparison call and are excluded from logs, tracing, metrics, artifacts, and
serialized responses.

The optional InsightFace adapter uses its own `FaceAnalysis` detection,
landmarks, alignment, and recognition pipeline, with local CPU ONNX Runtime
models provisioned through `KYC_FACE_RECOGNITION_MODEL_ROOT` and the safe
`KYC_FACE_RECOGNITION_MODEL_ID` when `KYC_FACE_RECOGNITION_ENABLED` is true.
Automatic model downloads are disabled. InsightFace pretrained model packs are
development/test-only for this project unless commercial-use rights for the
exact weights are confirmed separately; model weights must not be committed to
this repository.

## Profile Contracts

- `DocumentProfile`: profile identity, document type/side, canonical dimensions, review status, ordered fields, and an optional QR region.
- `FieldDefinition`: absolute canonical box, padding, required state, value type, OCR mode, comparison, normalizer, and validator.
- `FieldCrop`: one field from one variant with exact box and OCR mode.

The first registry key is `ao_id_card/front/v1`. A provisional profile remains usable for development but adds a `PROFILE_PROVISIONAL` warning.

## Preprocessing Contracts

- `VariantInfo`: unique variant name, read-only image, immutable effective parameters.
- `VariantBatch`: one generator name and its ordered variants.
- `AssessmentResult`: one assessor's metrics and effective parameters.
- `VariantAssessment`: generator, variant, original variant parameters, and ordered assessor results.
- `VariantBatchAssessment`: assessments for one generator batch.

Balanced mode emits at most ten variants and always retains the original before structural quality gating.

## OCR and Output Contracts

- `OCRCandidate`: field, raw response, OCR confidence, generator/variant provenance, canonical box, engine/model version, and optional failure code.
- `ExtractedField`: raw and optional normalized values, field state, confidence, selected candidate, alternatives, and warnings.
- `PipelineIssue`: stage, stable code, warning/error severity, PII-safe message, and optional field name.
- `KycExtractionResult`: schema version, `success | partial | failed`, profile/detection metadata, ordered field mapping, issues, and stage timings.
- `QrCodeResult`: optional back-side QR payload, independently parsed QR record, canonical QR box, decoder provenance, and a QR-specific status. It does not participate in OCR reconciliation.
- `DocumentExtractionResult`: coordinator schema version, overall status, nested front/back results, coordinator issues, and side timings.

`KycExtractionResult.to_dict()` deliberately omits source images, normalized images, crops, paths, EXIF, and stack traces.

## Provenance and Failure Rules

- Every selected non-empty field references a real `OCRCandidate`.
- Raw OCR text is retained; normalized text is added only for unambiguous conversion.
- Validation state, consensus support, OCR confidence, and stable variant order determine selection in that order.
- Empty and failed OCR responses remain candidates and are distinguishable.
- Required missing, invalid, error, or conflicting fields make the result partial; optional failures add warnings.
- Fatal intake, detection, profile, or geometry failures stop downstream processing.
- Detector confidence, OCR confidence, and reconciliation selection are not collapsed into one score.
