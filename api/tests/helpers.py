from __future__ import annotations

from pathlib import Path
from threading import Event

import cv2
import numpy as np

from kyc_engine import (
    CaptureAssessment,
    CaptureMetrics,
    DocumentExtractionResult,
    KycExtractionResult,
    ProcessingStatus,
    LivenessResult,
)
from kyc_api.models import DocumentSide

from kyc_api.settings import ApiSettings


API_KEY = "test-api-key-123456789"
AUTH_HEADERS = {"X-API-Key": API_KEY}

def _synthetic_png() -> bytes:
    image = np.full((240, 320, 3), (48, 70, 88), dtype=np.uint8)
    card = np.asarray(((48, 52), (270, 46), (279, 185), (40, 190)), dtype=np.int32)
    cv2.fillConvexPoly(image, card, (202, 202, 202), cv2.LINE_8)
    cv2.polylines(image, (card,), True, (246, 246, 246), 3, cv2.LINE_8)
    for offset in range(3):
        y = 88 + offset * 24
        cv2.line(image, (85, y), (220, y - 4), (75, 75, 75), 2, cv2.LINE_8)
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not success:
        raise AssertionError("could not encode synthetic document capture")
    return encoded.tobytes()


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

    def release_pending_artifacts(self, _result: DocumentExtractionResult) -> None:
        """Match the coordinator cleanup hook without retaining test artifacts."""
        return None


class FakeLivenessEvaluator:
    def __init__(
        self,
        result: LivenessResult | None = None,
        *,
        frame_count: int = 3,
        error: Exception | None = None,
    ) -> None:
        self.frame_count = frame_count
        self.result = result or LivenessResult(True, 0.9, frame_count, frame_count)
        self.error = error
        self.calls = 0
        self.frames: tuple | None = None

    def evaluate(self, frames) -> LivenessResult:
        self.calls += 1
        self.frames = tuple(frames)
        if self.error is not None:
            raise self.error
        return self.result


def accepted_capture_assessment() -> CaptureAssessment:
    return CaptureAssessment(
        accepted=True,
        issues=(),
        metrics=CaptureMetrics(
            width=320,
            height=240,
            sharpness=100.0,
            brightness=128.0,
            contrast=50.0,
            document_detected=True,
            document_area_ratio=0.4,
            minimum_margin_ratio=0.1,
            perspective_score=0.0,
            glare_ratio=0.0,
        ),
    )


def accept_document(store, session_id: str, *, front: bytes = PNG_BYTES, back: bytes = PNG_BYTES) -> None:
    assessment = accepted_capture_assessment()
    store.accept_document_capture(session_id, DocumentSide.FRONT, front, assessment)
    store.accept_document_capture(session_id, DocumentSide.BACK, back, assessment)
