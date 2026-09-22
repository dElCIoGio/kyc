from __future__ import annotations

import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from kyc_engine import DocumentCoordinator
from kyc_engine.defaults import build_paddle_pipeline

from .settings import ApiSettings


def create_coordinator(settings: ApiSettings) -> DocumentCoordinator:
    with _suppress_backend_output():
        front_pipeline = build_paddle_pipeline(
            model_manifest=settings.ocr_model_manifest,
            device=settings.ocr_device,
            side="front",
        )
        back_pipeline = build_paddle_pipeline(
            model_manifest=settings.ocr_model_manifest,
            device=settings.ocr_device,
            side="back",
        )
    return DocumentCoordinator(front_pipeline, back_pipeline)


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
