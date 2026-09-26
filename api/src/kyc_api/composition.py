from __future__ import annotations

import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from kyc_engine import (
    CaptureAssessmentConfig,
    DocumentCaptureAssessor,
    DocumentCoordinator,
    IntakeLimits,
    build_paddle_document_coordinator,
)
from kyc_engine.intake import ImageIntake

from .settings import ApiSettings


def create_coordinator(settings: ApiSettings) -> DocumentCoordinator:
    with _suppress_backend_output():
        return build_paddle_document_coordinator(
            model_manifest=settings.ocr_model_manifest,
            device=settings.ocr_device,
            intake_limits=IntakeLimits(max_encoded_bytes=settings.max_upload_bytes),
        )


def create_capture_assessor(settings: ApiSettings) -> DocumentCaptureAssessor:
    return DocumentCaptureAssessor(
        intake=ImageIntake(IntakeLimits(max_encoded_bytes=settings.max_upload_bytes)),
        config=CaptureAssessmentConfig(
            min_sharpness=settings.capture_min_sharpness,
            min_brightness=settings.capture_min_brightness,
            max_brightness=settings.capture_max_brightness,
            min_contrast=settings.capture_min_contrast,
        ),
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
