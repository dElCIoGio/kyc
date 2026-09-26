from __future__ import annotations

import io
from pathlib import Path
from threading import Event

from PIL import Image, ImageDraw

from kyc_engine import (
    CaptureAssessment,
    CaptureMetrics,
    DocumentExtractionResult,
    KycExtractionResult,
    ProcessingStatus,
)
from kyc_api.models import DocumentSide

from kyc_api.settings import ApiSettings


API_KEY = "test-api-key-123456789"
AUTH_HEADERS = {"X-API-Key": API_KEY}

def _synthetic_png() -> bytes:
    image = Image.new("L", (320, 240), 128)
    draw = ImageDraw.Draw(image)
    for x in range(0, 320, 16):
        draw.line((x, 0, x, 239), fill=30 if (x // 16) % 2 else 225, width=3)
    for y in range(0, 240, 16):
        draw.line((0, y, 319, y), fill=225 if (y // 16) % 2 else 30, width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


PNG_BYTES = _synthetic_png()


def settings(**overrides) -> ApiSettings:
    values = {
        "api_key": API_KEY,
        "ocr_model_manifest": Path("unused-in-tests.json"),
        "max_upload_bytes": 1024,
        "session_ttl_seconds": 60,
        "max_sessions": 8,
        "job_workers": 1,
    }
    values.update(overrides)
    return ApiSettings(**values)


def extraction_result(
    status: ProcessingStatus = ProcessingStatus.SUCCESS,
) -> DocumentExtractionResult:
    front = _side_result("front", status)
    back = _side_result("back", status) if status == ProcessingStatus.SUCCESS else None
    return DocumentExtractionResult(
        schema_version="1.0",
        status=status,
        front=front,
        back=back,
        issues=(),
        timings_ms={"front_total": 1.0},
    )


def _side_result(side: str, status: ProcessingStatus) -> KycExtractionResult:
    return KycExtractionResult(
        schema_version="1.1",
        processing_id=f"safe-{side}-id",
        status=status,
        document_type="ao_id_card",
        side=side,
        profile_id=f"ao_id_card/{side}/v1",
        detection=None,
        fields={},
        issues=(),
        timings_ms={},
    )


class FakeCoordinator:
    def __init__(
        self,
        *,
        status: ProcessingStatus = ProcessingStatus.SUCCESS,
        started: Event | None = None,
        release: Event | None = None,
        fail: bool = False,
    ) -> None:
        self.output = extraction_result(status)
        self.started = started
        self.release = release
        self.fail = fail
        self.calls: list[tuple[bytes | None, bytes | None]] = []

    def process(
        self,
        *,
        front: bytes | None = None,
        back: bytes | None = None,
    ) -> DocumentExtractionResult:
        self.calls.append((front, back))
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=3)
        if self.fail:
            raise RuntimeError("sensitive backend failure")
        return self.output


def accepted_capture_assessment() -> CaptureAssessment:
    return CaptureAssessment(
        accepted=True,
        issues=(),
        metrics=CaptureMetrics(width=320, height=240, sharpness=100.0, brightness=128.0, contrast=50.0),
    )


def accept_document(store, session_id: str, *, front: bytes = PNG_BYTES, back: bytes = PNG_BYTES) -> None:
    assessment = accepted_capture_assessment()
    store.accept_document_capture(session_id, DocumentSide.FRONT, front, assessment)
    store.accept_document_capture(session_id, DocumentSide.BACK, back, assessment)
