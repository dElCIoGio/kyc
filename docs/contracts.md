# Core Contracts

The contracts in `src/kyc_engine/contracts.py` are frozen dataclasses. Collections are tuples or immutable mappings, and image arrays are read-only.

## Entry Point

```python
class KycPipeline:
    def process(self, source: ImageSource) -> KycExtractionResult: ...

ImageSource = Path | str | bytes | NDArray[np.uint8]
```

Invalid caller types and invalid configuration raise exceptions. Expected image-processing failures return `KycExtractionResult(status="failed")` with a machine-readable issue.

## Image and Geometry Contracts

- `InputImage`: copied BGR pixels plus width, height, channels, decoded format, and random processing ID. It contains no path or EXIF.
- `Point`, `Quadrilateral`, `BoundingBox`: finite geometry in documented coordinate systems. Quadrilateral order is TL, TR, BR, BL.
- `DetectionResult`: type, side, confidence, four corners, detector/version, confidence components, and orientation.
- `NormalizedDocument`: canonical image, profile ID, dimensions, and forward/inverse 3x3 transforms.

## Profile Contracts

- `DocumentProfile`: profile identity, document type/side, canonical dimensions, review status, and ordered fields.
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

`KycExtractionResult.to_dict()` deliberately omits source images, normalized images, crops, paths, EXIF, and stack traces.

## Provenance and Failure Rules

- Every selected non-empty field references a real `OCRCandidate`.
- Raw OCR text is retained; normalized text is added only for unambiguous conversion.
- Validation state, consensus support, OCR confidence, and stable variant order determine selection in that order.
- Empty and failed OCR responses remain candidates and are distinguishable.
- Required missing, invalid, error, or conflicting fields make the result partial; optional failures add warnings.
- Fatal intake, detection, profile, or geometry failures stop downstream processing.
- Detector confidence, OCR confidence, and reconciliation selection are not collapsed into one score.
