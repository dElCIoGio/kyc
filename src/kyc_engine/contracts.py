from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray

Image: TypeAlias = NDArray[np.uint8]
ImageSource: TypeAlias = Path | str | bytes | Image


def immutable_mapping(values: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


def readonly_image(image: Image, *, copy: bool = False) -> Image:
    pixels = np.array(image, dtype=np.uint8, copy=copy, order="C")
    pixels.setflags(write=False)
    return pixels


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class IssueSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class FieldStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    MISSING = "missing"
    ERROR = "error"


class QrCodeStatus(str, Enum):
    DECODED = "decoded"
    NOT_DETECTED = "not_detected"
    DECODE_FAILED = "decode_failed"
    PARSE_FAILED = "parse_failed"


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("Point coordinates must be finite")


@dataclass(frozen=True)
class Quadrilateral:
    """Corners ordered top-left, top-right, bottom-right, bottom-left."""

    points: tuple[Point, Point, Point, Point]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if len(points) != 4:
            raise ValueError("Quadrilateral must contain exactly four points")
        object.__setattr__(self, "points", points)

    def as_array(self) -> NDArray[np.float32]:
        return np.asarray([(point.x, point.y) for point in self.points], dtype=np.float32)


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.x, self.y, self.width, self.height)):
            raise TypeError("Bounding box values must be integers")
        if self.x < 0 or self.y < 0:
            raise ValueError("Bounding box origin must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bounding box dimensions must be positive")

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class InputImage:
    image: Image
    width: int
    height: int
    channels: int
    source_format: str
    processing_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", readonly_image(self.image))


@dataclass(frozen=True)
class DetectionResult:
    document_type: str
    side: str
    confidence: float
    corners: Quadrilateral
    detector: str
    detector_version: str
    confidence_components: Mapping[str, float] = field(default_factory=dict)
    orientation_degrees: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")
        if self.orientation_degrees not in (0, 90, 180, 270):
            raise ValueError("Orientation must be 0, 90, 180, or 270 degrees")
        object.__setattr__(self, "confidence_components", immutable_mapping(self.confidence_components))


@dataclass(frozen=True)
class NormalizedDocument:
    image: Image
    profile_id: str
    width: int
    height: int
    forward_transform: tuple[tuple[float, float, float], ...]
    inverse_transform: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", readonly_image(self.image))


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    bounding_box: BoundingBox
    required: bool = True
    value_type: str = "text"
    multiline: bool = False
    padding: int = 0
    ocr_mode: str = "single_line"
    comparison: str = "casefold_whitespace"
    normalizer: str = "text"
    validator: str = "non_empty"


@dataclass(frozen=True)
class QrCodeDefinition:
    bounding_box: BoundingBox
    padding: int = 0


@dataclass(frozen=True)
class DocumentProfile:
    profile_id: str
    document_type: str
    side: str
    canonical_width: int
    canonical_height: int
    review_status: str
    fields: tuple[FieldDefinition, ...]
    qr_code: QrCodeDefinition | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))


@dataclass(frozen=True)
class VariantInfo:
    name: str
    image: Image
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", readonly_image(self.image))
        object.__setattr__(self, "parameters", immutable_mapping(self.parameters))


@dataclass(frozen=True)
class VariantBatch:
    generator: str
    variants: tuple[VariantInfo, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variants", tuple(self.variants))


@dataclass(frozen=True)
class AssessmentResult:
    assessor: str
    metrics: Mapping[str, float]
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", immutable_mapping(self.metrics))
        object.__setattr__(self, "parameters", immutable_mapping(self.parameters))


@dataclass(frozen=True)
class VariantAssessment:
    variant_name: str
    generator: str
    variant_parameters: Mapping[str, object]
    results: tuple[AssessmentResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant_parameters", immutable_mapping(self.variant_parameters))
        object.__setattr__(self, "results", tuple(self.results))


@dataclass(frozen=True)
class VariantBatchAssessment:
    generator: str
    assessments: tuple[VariantAssessment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assessments", tuple(self.assessments))


@dataclass(frozen=True)
class FieldCrop:
    field_name: str
    generator: str
    variant_name: str
    image: Image
    bounding_box: BoundingBox
    ocr_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", readonly_image(self.image))


@dataclass(frozen=True)
class OCRCandidate:
    field_name: str
    raw_value: str
    confidence: float
    generator: str
    variant_name: str
    bounding_box: BoundingBox
    ocr_engine: str
    ocr_model_version: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("OCR confidence must be between 0 and 1")


@dataclass(frozen=True)
class ExtractedField:
    name: str
    status: FieldStatus
    raw_value: str | None
    normalized_value: str | None
    confidence: float
    selected_candidate: OCRCandidate | None
    alternatives: tuple[OCRCandidate, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True)
class QrIdentityData:
    full_name: str
    id_number: str
    birth_province: str
    date_of_birth: str
    sex: str
    marital_status: str
    issue_date: str
    expiry_date: str
    issuing_province: str
    version: str


@dataclass(frozen=True)
class QrCodeResult:
    status: QrCodeStatus
    raw_payload: str | None
    data: QrIdentityData | None
    bounding_box: BoundingBox
    decoder: str
    decoder_version: str
    variant_name: str | None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.status == QrCodeStatus.DECODED and self.data is None:
            raise ValueError("Decoded QR results require structured data")


@dataclass(frozen=True)
class PipelineIssue:
    stage: str
    code: str
    severity: IssueSeverity
    message: str
    field_name: str | None = None


@dataclass(frozen=True)
class KycExtractionResult:
    schema_version: str
    processing_id: str
    status: ProcessingStatus
    document_type: str | None
    side: str | None
    profile_id: str | None
    detection: DetectionResult | None
    fields: Mapping[str, ExtractedField]
    issues: tuple[PipelineIssue, ...]
    timings_ms: Mapping[str, float]
    qr_code: QrCodeResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", immutable_mapping(self.fields))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "timings_ms", immutable_mapping(self.timings_ms))

    def to_dict(self) -> dict[str, Any]:
        detection = None
        if self.detection is not None:
            detection = {
                "document_type": self.detection.document_type,
                "side": self.detection.side,
                "confidence": self.detection.confidence,
                "corners": [
                    {"x": point.x, "y": point.y}
                    for point in self.detection.corners.points
                ],
                "detector": self.detection.detector,
                "detector_version": self.detection.detector_version,
                "orientation_degrees": self.detection.orientation_degrees,
            }

        return {
            "schema_version": self.schema_version,
            "processing_id": self.processing_id,
            "status": self.status.value,
            "document_type": self.document_type,
            "side": self.side,
            "profile_id": self.profile_id,
            "detection": detection,
            "fields": {
                name: _field_to_dict(value) for name, value in self.fields.items()
            },
            "qr_code": _qr_code_to_dict(self.qr_code),
            "issues": [
                {
                    "stage": issue.stage,
                    "code": issue.code,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "field_name": issue.field_name,
                }
                for issue in self.issues
            ],
            "timings_ms": dict(self.timings_ms),
        }


def _candidate_to_dict(candidate: OCRCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "field_name": candidate.field_name,
        "raw_value": candidate.raw_value,
        "confidence": candidate.confidence,
        "generator": candidate.generator,
        "variant_name": candidate.variant_name,
        "bounding_box": {
            "x": candidate.bounding_box.x,
            "y": candidate.bounding_box.y,
            "width": candidate.bounding_box.width,
            "height": candidate.bounding_box.height,
        },
        "ocr_engine": candidate.ocr_engine,
        "ocr_model_version": candidate.ocr_model_version,
        "error_code": candidate.error_code,
    }


def _field_to_dict(value: ExtractedField) -> dict[str, Any]:
    return {
        "status": value.status.value,
        "raw_value": value.raw_value,
        "normalized_value": value.normalized_value,
        "confidence": value.confidence,
        "selected_candidate": _candidate_to_dict(value.selected_candidate),
        "alternatives": [_candidate_to_dict(item) for item in value.alternatives],
        "warnings": list(value.warnings),
    }


def _qr_code_to_dict(value: QrCodeResult | None) -> dict[str, Any] | None:
    if value is None:
        return None
    data = None
    if value.data is not None:
        data = {
            "full_name": value.data.full_name,
            "id_number": value.data.id_number,
            "birth_province": value.data.birth_province,
            "date_of_birth": value.data.date_of_birth,
            "sex": value.data.sex,
            "marital_status": value.data.marital_status,
            "issue_date": value.data.issue_date,
            "expiry_date": value.data.expiry_date,
            "issuing_province": value.data.issuing_province,
            "version": value.data.version,
        }
    return {
        "status": value.status.value,
        "raw_payload": value.raw_payload,
        "data": data,
        "bounding_box": {
            "x": value.bounding_box.x,
            "y": value.bounding_box.y,
            "width": value.bounding_box.width,
            "height": value.bounding_box.height,
        },
        "decoder": value.decoder,
        "decoder_version": value.decoder_version,
        "variant_name": value.variant_name,
        "warnings": list(value.warnings),
    }
