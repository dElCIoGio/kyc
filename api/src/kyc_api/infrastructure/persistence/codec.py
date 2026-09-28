"""Strict JSON codec for durable, non-biometric document extraction results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kyc_engine import DocumentExtractionResult
from kyc_engine.contracts import (
    BoundingBox,
    DetectionResult,
    ExtractedField,
    FaceCandidate,
    FieldStatus,
    IssueSeverity,
    KycExtractionResult,
    OCRCandidate,
    PipelineIssue,
    Point,
    PortraitDefinition,
    PortraitExtractionResult,
    PortraitStatus,
    ProcessingStatus,
    QrCodeResult,
    QrCodeStatus,
    QrIdentityData,
    Quadrilateral,
)


_CODEC_VERSION = 1


class DocumentResultCodecError(ValueError):
    pass


def encode_document_result(result: DocumentExtractionResult) -> dict[str, Any]:
    return {
        "codec_version": _CODEC_VERSION,
        "result": {
            "schema_version": result.schema_version,
            "status": result.status.value,
            "front": _encode_side(result.front),
            "back": _encode_side(result.back),
            "issues": [_encode_issue(issue) for issue in result.issues],
            "timings_ms": dict(result.timings_ms),
        },
    }


def decode_document_result(value: object) -> DocumentExtractionResult:
    try:
        root = _mapping(value)
        if root.get("codec_version") != _CODEC_VERSION:
            raise DocumentResultCodecError("unsupported document result codec version")
        result = _mapping(root.get("result"))
        return DocumentExtractionResult(
            schema_version=_string(result, "schema_version"),
            status=ProcessingStatus(_string(result, "status")),
            front=_decode_side(result.get("front")),
            back=_decode_side(result.get("back")),
            issues=tuple(_decode_issue(item) for item in _list(result, "issues")),
            timings_ms=_float_mapping(result, "timings_ms"),
        )
    except DocumentResultCodecError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise DocumentResultCodecError("malformed document result") from exc


def _encode_side(value: KycExtractionResult | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "schema_version": value.schema_version,
        "processing_id": value.processing_id,
        "status": value.status.value,
        "document_type": value.document_type,
        "side": value.side,
        "profile_id": value.profile_id,
        "detection": _encode_detection(value.detection),
        "fields": {name: _encode_field(field) for name, field in value.fields.items()},
        "issues": [_encode_issue(issue) for issue in value.issues],
        "timings_ms": dict(value.timings_ms),
        "qr_code": _encode_qr(value.qr_code),
        "portrait": _encode_portrait(value.portrait),
    }


def _decode_side(value: object) -> KycExtractionResult | None:
    if value is None:
        return None
    side = _mapping(value)
    fields = _mapping(side.get("fields"))
    return KycExtractionResult(
        schema_version=_string(side, "schema_version"),
        processing_id=_string(side, "processing_id"),
        status=ProcessingStatus(_string(side, "status")),
        document_type=_optional_string(side, "document_type"),
        side=_optional_string(side, "side"),
        profile_id=_optional_string(side, "profile_id"),
        detection=_decode_detection(side.get("detection")),
        fields={str(name): _decode_field(str(name), item) for name, item in fields.items()},
        issues=tuple(_decode_issue(item) for item in _list(side, "issues")),
        timings_ms=_float_mapping(side, "timings_ms"),
        qr_code=_decode_qr(side.get("qr_code")),
        portrait=_decode_portrait(side.get("portrait")),
    )


def _encode_detection(value: DetectionResult | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "document_type": value.document_type,
        "side": value.side,
        "confidence": value.confidence,
        "corners": [{"x": p.x, "y": p.y} for p in value.corners.points],
        "detector": value.detector,
        "detector_version": value.detector_version,
        "confidence_components": dict(value.confidence_components),
        "orientation_degrees": value.orientation_degrees,
    }


def _decode_detection(value: object) -> DetectionResult | None:
    if value is None:
        return None
    item = _mapping(value)
    corners = _list(item, "corners")
    if len(corners) != 4:
        raise DocumentResultCodecError("detection must contain four corners")
    return DetectionResult(
        document_type=_string(item, "document_type"),
        side=_string(item, "side"),
        confidence=_float(item, "confidence"),
        corners=Quadrilateral(tuple(Point(_float(_mapping(p), "x"), _float(_mapping(p), "y")) for p in corners)),  # type: ignore[arg-type]
        detector=_string(item, "detector"),
        detector_version=_string(item, "detector_version"),
        confidence_components=_float_mapping(item, "confidence_components"),
        orientation_degrees=_integer(item, "orientation_degrees"),
    )


def _encode_field(value: ExtractedField) -> dict[str, Any]:
    return {
        "name": value.name,
        "status": value.status.value,
        "raw_value": value.raw_value,
        "normalized_value": value.normalized_value,
        "confidence": value.confidence,
        "selected_candidate": _encode_candidate(value.selected_candidate),
        "alternatives": [_encode_candidate(item) for item in value.alternatives],
        "warnings": list(value.warnings),
    }


def _decode_field(name: str, value: object) -> ExtractedField:
    item = _mapping(value)
    stored_name = _string(item, "name")
    if stored_name != name:
        raise DocumentResultCodecError("field name does not match mapping key")
    return ExtractedField(
        name=stored_name,
        status=FieldStatus(_string(item, "status")),
        raw_value=_optional_string(item, "raw_value"),
        normalized_value=_optional_string(item, "normalized_value"),
        confidence=_float(item, "confidence"),
        selected_candidate=_decode_candidate(item.get("selected_candidate")),
        alternatives=tuple(_decode_candidate(candidate, required=True) for candidate in _list(item, "alternatives")),  # type: ignore[misc]
        warnings=tuple(_strings(item, "warnings")),
    )


def _encode_candidate(value: OCRCandidate | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "field_name": value.field_name,
        "raw_value": value.raw_value,
        "confidence": value.confidence,
        "generator": value.generator,
        "variant_name": value.variant_name,
        "bounding_box": _encode_box(value.bounding_box),
        "ocr_engine": value.ocr_engine,
        "ocr_model_version": value.ocr_model_version,
        "error_code": value.error_code,
    }


def _decode_candidate(value: object, *, required: bool = False) -> OCRCandidate | None:
    if value is None and not required:
        return None
    item = _mapping(value)
    return OCRCandidate(
        field_name=_string(item, "field_name"),
        raw_value=_string(item, "raw_value"),
        confidence=_float(item, "confidence"),
        generator=_string(item, "generator"),
        variant_name=_string(item, "variant_name"),
        bounding_box=_decode_box(item.get("bounding_box")),
        ocr_engine=_string(item, "ocr_engine"),
        ocr_model_version=_string(item, "ocr_model_version"),
        error_code=_optional_string(item, "error_code"),
    )


def _encode_qr(value: QrCodeResult | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "status": value.status.value,
        "raw_payload": value.raw_payload,
        "data": vars(value.data) if value.data is not None else None,
        "bounding_box": _encode_box(value.bounding_box),
        "decoder": value.decoder,
        "decoder_version": value.decoder_version,
        "variant_name": value.variant_name,
        "warnings": list(value.warnings),
    }


def _decode_qr(value: object) -> QrCodeResult | None:
    if value is None:
        return None
    item = _mapping(value)
    data_value = item.get("data")
    data = None
    if data_value is not None:
        raw = _mapping(data_value)
        data = QrIdentityData(**{name: _string(raw, name) for name in QrIdentityData.__dataclass_fields__})
    return QrCodeResult(
        status=QrCodeStatus(_string(item, "status")),
        raw_payload=_optional_string(item, "raw_payload"),
        data=data,
        bounding_box=_decode_box(item.get("bounding_box")),
        decoder=_string(item, "decoder"),
        decoder_version=_string(item, "decoder_version"),
        variant_name=_optional_string(item, "variant_name"),
        warnings=tuple(_strings(item, "warnings")),
    )


def _encode_portrait(value: PortraitExtractionResult | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "status": value.status.value,
        "requested_region": vars(value.requested_region) if value.requested_region is not None else None,
        "clamped_region": _encode_box(value.clamped_region),
        "face_detected": value.face_detected,
        "face_count": value.face_count,
        "face": _encode_face(value.face),
        "eligible_for_face_match": value.eligible_for_face_match,
        "warnings": list(value.warnings),
    }


def _decode_portrait(value: object) -> PortraitExtractionResult | None:
    if value is None:
        return None
    item = _mapping(value)
    requested = item.get("requested_region")
    requested_region = None
    if requested is not None:
        region = _mapping(requested)
        requested_region = PortraitDefinition(*(_integer(region, key) for key in ("x", "y", "width", "height")))
    return PortraitExtractionResult(
        status=PortraitStatus(_string(item, "status")),
        requested_region=requested_region,
        clamped_region=_decode_box(item.get("clamped_region"), optional=True),
        face_detected=_boolean(item, "face_detected"),
        face_count=_integer(item, "face_count"),
        face=_decode_face(item.get("face")),
        eligible_for_face_match=_boolean(item, "eligible_for_face_match"),
        warnings=tuple(_strings(item, "warnings")),
        artifact_id=None,
    )


def _encode_face(value: FaceCandidate | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "bounding_box": _encode_box(value.bounding_box),
        "confidence": value.confidence,
        "landmarks": [{"x": point.x, "y": point.y} for point in value.landmarks],
    }


def _decode_face(value: object) -> FaceCandidate | None:
    if value is None:
        return None
    item = _mapping(value)
    confidence = item.get("confidence")
    if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))):
        raise DocumentResultCodecError("face confidence must be numeric")
    return FaceCandidate(
        bounding_box=_decode_box(item.get("bounding_box")),
        confidence=float(confidence) if confidence is not None else None,
        landmarks=tuple(Point(_float(_mapping(p), "x"), _float(_mapping(p), "y")) for p in _list(item, "landmarks")),
    )


def _encode_issue(value: PipelineIssue) -> dict[str, Any]:
    return {"stage": value.stage, "code": value.code, "severity": value.severity.value, "message": value.message, "field_name": value.field_name}


def _decode_issue(value: object) -> PipelineIssue:
    item = _mapping(value)
    return PipelineIssue(
        stage=_string(item, "stage"),
        code=_string(item, "code"),
        severity=IssueSeverity(_string(item, "severity")),
        message=_string(item, "message"),
        field_name=_optional_string(item, "field_name"),
    )


def _encode_box(value: BoundingBox | None) -> dict[str, int] | None:
    return None if value is None else {"x": value.x, "y": value.y, "width": value.width, "height": value.height}


def _decode_box(value: object, *, optional: bool = False) -> BoundingBox | None:
    if value is None and optional:
        return None
    item = _mapping(value)
    return BoundingBox(*(_integer(item, key) for key in ("x", "y", "width", "height")))


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DocumentResultCodecError("expected an object")
    return value


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise DocumentResultCodecError(f"{key} must be a list")
    return item


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise DocumentResultCodecError(f"{key} must be a string")
    return item


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is not None and not isinstance(item, str):
        raise DocumentResultCodecError(f"{key} must be a string or null")
    return item


def _float(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise DocumentResultCodecError(f"{key} must be numeric")
    return float(item)


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise DocumentResultCodecError(f"{key} must be an integer")
    return item


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise DocumentResultCodecError(f"{key} must be a boolean")
    return item


def _strings(value: Mapping[str, Any], key: str) -> list[str]:
    items = _list(value, key)
    if not all(isinstance(item, str) for item in items):
        raise DocumentResultCodecError(f"{key} must contain strings")
    return items


def _float_mapping(value: Mapping[str, Any], key: str) -> dict[str, float]:
    item = _mapping(value.get(key))
    result: dict[str, float] = {}
    for name, raw in item.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DocumentResultCodecError(f"{key} values must be numeric")
        result[name] = float(raw)
    return result
