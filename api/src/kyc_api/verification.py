from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Callable

from kyc_engine import (
    CaptureAssessment,
    DocumentCaptureAssessor,
    LivenessEvaluator,
    LivenessFrameError,
    LivenessResult,
)
from kyc_engine.intake import ImageIntake, IntakeError

from .jobs import JobCapacityExceeded, JobManager
from .models import DocumentSide, DocumentStatus, LivenessStatus
from .orchestration import VerificationOrchestrator
from .sessions import SessionConflict, SessionSnapshot, SessionStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureSubmission:
    assessment: CaptureAssessment
    snapshot: SessionSnapshot


@dataclass(frozen=True)
class LivenessSubmission:
    result: LivenessResult
    snapshot: SessionSnapshot


class LivenessSubmissionError(RuntimeError):
    code = "LIVENESS_SUBMISSION_FAILED"
    status_code = 422


class LivenessUnavailable(LivenessSubmissionError):
    code = "LIVENESS_UNAVAILABLE"
    status_code = 503


class LivenessInputError(LivenessSubmissionError):
    code = "INVALID_LIVENESS_FRAME"
    status_code = 422

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class LivenessOperationalError(LivenessSubmissionError):
    code = "LIVENESS_EVALUATION_FAILED"
    status_code = 503


class VerificationManager:
    """Coordinates capture assessment, verification state, and document work."""

    def __init__(
        self,
        *,
        assessor: DocumentCaptureAssessor,
        store: SessionStore,
        jobs: JobManager,
        liveness_evaluator: LivenessEvaluator | None = None,
        liveness_intake: ImageIntake | None = None,
        orchestrator: VerificationOrchestrator | None = None,
        snapshot_publisher: Callable[[SessionSnapshot, str], None] | None = None,
    ) -> None:
        self._assessor = assessor
        self._store = store
        self._jobs = jobs
        self._liveness_evaluator = liveness_evaluator
        self._liveness_intake = liveness_intake
        self._orchestrator = orchestrator
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

    def start_liveness(self, session_id: str) -> SessionSnapshot:
        snapshot = self._store.start_liveness(session_id)
        self._publish(snapshot, "liveness.started")
        logger.info("liveness started", extra={"event": "liveness.started"})
        return snapshot

    @property
    def liveness_frame_count(self) -> int:
        if self._liveness_evaluator is None:
            raise LivenessUnavailable("Passive liveness is not configured")
        return self._liveness_evaluator.frame_count

    def submit_liveness(
        self, session_id: str, encoded_frames: Sequence[bytes]
    ) -> LivenessSubmission:
        """Evaluate an ephemeral, exact-size liveness frame set for one session."""
        evaluator = self._liveness_evaluator
        intake = self._liveness_intake
        if evaluator is None or intake is None:
            raise LivenessUnavailable("Passive liveness is not configured")
        if len(encoded_frames) != evaluator.frame_count:
            raise LivenessInputError("Incorrect number of liveness frames")
        current = self._store.get(session_id)
        if current.liveness_status != LivenessStatus.READY:
            raise SessionConflict("Liveness is not ready to start")

        frames = self._decode_liveness_frames(encoded_frames, intake)
        self.start_liveness(session_id)
        try:
            evaluated = getattr(evaluator, "evaluate_with_live_face", None)
            evaluation = evaluated(frames) if callable(evaluated) else None
            result = evaluation.result if evaluation is not None else evaluator.evaluate(frames)
        except LivenessFrameError as exc:
            self._fail_liveness_or_conflict(session_id, "LIVENESS_CAPTURE_FAILED")
            raise LivenessInputError(
                "A liveness frame did not contain a usable face", code="NO_FACE_DETECTED"
            ) from exc
        except Exception as exc:
            self._fail_liveness_or_conflict(session_id, "LIVENESS_EVALUATION_FAILED")
            logger.exception(
                "liveness evaluation failed",
                extra={"event": "liveness.evaluation_failed", "exception_type": type(exc).__name__},
            )
            raise LivenessOperationalError("Liveness evaluation could not be completed") from exc

        artifact_id = None
        selection = evaluation.live_face if evaluation is not None else None
        if result.passed and selection is not None and selection.selected_frame is not None:
            artifacts = self._store.portrait_artifacts
            if artifacts is not None:
                try:
                    artifact_id = artifacts.put(selection.selected_frame)
                except Exception:
                    logger.warning(
                        "live face artifact retention failed",
                        extra={"event": "liveness.live_face_retention_failed"},
                    )
        completed = self.complete_liveness(
            session_id,
            result,
            artifact_id,
            live_face_eligible=selection is not None and selection.selected_frame is not None,
        )
        if completed is None:
            raise SessionConflict("Liveness completion is no longer applicable")
        if selection is not None:
            self._log_live_face_selection(selection)
        return LivenessSubmission(result=result, snapshot=completed)

    def complete_liveness(
        self,
        session_id: str,
        result: LivenessResult,
        live_face_artifact_id: str | None = None,
        *,
        live_face_eligible: bool = False,
    ) -> SessionSnapshot | None:
        snapshot = self._store.complete_liveness(
            session_id,
            result,
            live_face_artifact_id,
            live_face_eligible,
        )
        if snapshot is None:
            return None
        transition_reason = "liveness.passed" if result.passed else "liveness.failed"
        self._publish(snapshot, transition_reason)
        logger.info("liveness completed", extra={"event": transition_reason})
        if self._orchestrator is not None:
            self._orchestrator.on_liveness_state_changed(snapshot)
        return snapshot

    def _log_live_face_selection(self, selection) -> None:
        """Emit only safe selection metadata; the crop never reaches logging."""
        logger.info(
            "live face selection evaluated",
            extra={
                "event": "liveness.live_face_selection",
                "candidate_frame_count": selection.candidate_frame_count,
                "eligible_frame_count": selection.eligible_frame_count,
                "selected_frame_index": (
                    selection.selected_frame_index
                    if selection.selected_frame_index is not None
                    else -1
                ),
                "face_detected": selection.face_detected,
                "face_count": selection.face_count,
                "quality_score": selection.quality_score or 0.0,
                "selection_outcome": selection.selection_outcome,
            },
        )

    def fail_liveness(self, session_id: str, code: str) -> SessionSnapshot | None:
        snapshot = self._store.fail_liveness(session_id, code)
        if snapshot is not None:
            self._publish(snapshot, "liveness.failed")
            logger.warning(
                "liveness failed",
                extra={"event": "liveness.failed", "error_code": code},
            )
        return snapshot

    def _decode_liveness_frames(
        self, encoded_frames: Sequence[bytes], intake: ImageIntake
    ) -> tuple:
        frames = []
        for content in encoded_frames:
            try:
                frames.append(intake.load(content).image)
            except IntakeError as exc:
                if exc.code in {"INPUT_TOO_LARGE", "IMAGE_TOO_LARGE"}:
                    raise LivenessInputError(
                        "Liveness frame exceeds supported limits",
                        code="LIVENESS_FRAME_TOO_LARGE",
                        status_code=413,
                    ) from exc
                if exc.code in {"UNSUPPORTED_FORMAT", "DECODE_FAILED"}:
                    raise LivenessInputError("Liveness frame is not a supported image") from exc
                raise LivenessInputError("Liveness frame is invalid") from exc
        return tuple(frames)

    def _fail_liveness_or_conflict(self, session_id: str, code: str) -> None:
        if self.fail_liveness(session_id, code) is None:
            raise SessionConflict("Liveness evaluation is no longer applicable")

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
