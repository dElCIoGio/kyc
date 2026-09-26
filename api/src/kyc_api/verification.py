from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from kyc_engine import CaptureAssessment, DocumentCaptureAssessor

from .jobs import JobCapacityExceeded, JobManager
from .models import DocumentSide, DocumentStatus
from .sessions import SessionConflict, SessionSnapshot, SessionStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureSubmission:
    assessment: CaptureAssessment
    snapshot: SessionSnapshot


class VerificationManager:
    """Coordinates capture assessment, verification state, and document work."""

    def __init__(
        self,
        *,
        assessor: DocumentCaptureAssessor,
        store: SessionStore,
        jobs: JobManager,
        snapshot_publisher: Callable[[SessionSnapshot, str], None] | None = None,
    ) -> None:
        self._assessor = assessor
        self._store = store
        self._jobs = jobs
        self._snapshot_publisher = snapshot_publisher

    def submit_document_capture(
        self, session_id: str, side: DocumentSide, content: bytes
    ) -> CaptureSubmission:
        assessment = self._assessor.assess(content)
        if not assessment.accepted:
            logger.info(
                "document capture rejected",
                extra={
                    "event": "document.capture_rejected",
                    "side": side.value,
                    "issue_codes": [issue.code.value for issue in assessment.issues],
                },
            )
            return CaptureSubmission(assessment=assessment, snapshot=self._store.get(session_id))

        transition = self._store.accept_document_capture(session_id, side, content, assessment)
        self._publish(transition.capture_accepted, "document.capture_accepted")
        logger.info(
            "document capture accepted",
            extra={"event": "document.capture_accepted", "side": side.value},
        )
        if transition.capture_completed is not None:
            self._publish(transition.capture_completed, "document.capture_completed")
            logger.info("document capture completed", extra={"event": "document.capture_completed"})
            try:
                snapshot = self._start_document_processing(session_id)
            except JobCapacityExceeded:
                logger.warning(
                    "document processing capacity unavailable after capture completion",
                    extra={"event": "document.processing_deferred", "error_code": "JOB_CAPACITY_EXCEEDED"},
                )
                snapshot = transition.snapshot
        else:
            snapshot = transition.snapshot
        return CaptureSubmission(assessment=assessment, snapshot=snapshot)

    def start_document_processing(self, session_id: str) -> SessionSnapshot:
        snapshot = self._store.get(session_id)
        if snapshot.document.status in {DocumentStatus.QUEUED, DocumentStatus.PROCESSING}:
            return snapshot
        return self._start_document_processing(session_id)

    def _start_document_processing(self, session_id: str) -> SessionSnapshot:
        try:
            return self._jobs.submit_document_processing(session_id)
        except SessionConflict:
            snapshot = self._store.get(session_id)
            if snapshot.document.status in {DocumentStatus.QUEUED, DocumentStatus.PROCESSING}:
                return snapshot
            raise

    def _publish(self, snapshot: SessionSnapshot, transition_reason: str) -> None:
        if self._snapshot_publisher is None:
            return
        try:
            self._snapshot_publisher(snapshot, transition_reason)
        except Exception:
            logger.exception(
                "webhook event enqueue failed",
                extra={"event": "webhook_enqueue_failed"},
            )
