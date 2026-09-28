from __future__ import annotations

import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from kyc_engine import (
    CaptureAssessmentConfig,
    DocumentCaptureAssessor,
    DocumentCoordinator,
    IntakeLimits,
    LivenessEvaluationConfig,
    LivenessEvaluator,
    FaceRecognizer,
    InsightFaceRecognizer,
    MiniFASNetAntiSpoofDetector,
    OpenCVHaarFaceDetector,
    build_paddle_document_coordinator,
)
from kyc_engine.intake import ImageIntake
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore

from ..settings import ApiSettings


def create_coordinator(
    settings: ApiSettings,
    *,
    portrait_artifacts: InMemoryPortraitArtifactStore | None = None,
) -> DocumentCoordinator:
    with _suppress_backend_output():
        return build_paddle_document_coordinator(
            model_manifest=settings.ocr_model_manifest,
            device=settings.ocr_device,
            intake_limits=IntakeLimits(max_encoded_bytes=settings.max_upload_bytes),
            portrait_artifacts=portrait_artifacts,
        )


def create_capture_assessor(settings: ApiSettings) -> DocumentCaptureAssessor:
    return DocumentCaptureAssessor(
        intake=ImageIntake(IntakeLimits(max_encoded_bytes=settings.max_upload_bytes)),
        config=CaptureAssessmentConfig(
            min_sharpness=settings.capture_min_sharpness,
            min_brightness=settings.capture_min_brightness,
            max_brightness=settings.capture_max_brightness,
            min_contrast=settings.capture_min_contrast,
            min_document_area_ratio=settings.capture_min_document_area_ratio,
            min_edge_margin_ratio=settings.capture_min_edge_margin_ratio,
            max_perspective_distortion=settings.capture_max_perspective_distortion,
            max_glare_ratio=settings.capture_max_glare_ratio,
        ),
    )


def create_liveness_evaluator(settings: ApiSettings) -> LivenessEvaluator | None:
    """Build the optional passive-liveness dependency graph once at startup."""
    if not settings.liveness_enabled:
        return None
    assert settings.liveness_model_root is not None
    detector = MiniFASNetAntiSpoofDetector(settings.liveness_model_root)
    face_detector = OpenCVHaarFaceDetector()
    return LivenessEvaluator(
        detector,
        LivenessEvaluationConfig(
            frame_count=settings.liveness_frame_count,
            minimum_real_ratio=settings.liveness_min_real_ratio,
        ),
        face_detector=face_detector,
    )


def create_liveness_intake(settings: ApiSettings) -> ImageIntake:
    return ImageIntake(IntakeLimits(max_encoded_bytes=settings.max_liveness_frame_bytes))


def create_face_recognizer(settings: ApiSettings) -> FaceRecognizer | None:
    """Build the optional local recognition dependency graph once at startup."""
    if not settings.face_recognition_enabled:
        return None
    assert settings.face_recognition_model_root is not None
    assert settings.face_recognition_model_id is not None
    return InsightFaceRecognizer(
        settings.face_recognition_model_root,
        settings.face_recognition_model_id,
    )


@contextmanager
def _suppress_backend_output():
    """Prevent model loaders from writing local paths or OCR details to logs."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
