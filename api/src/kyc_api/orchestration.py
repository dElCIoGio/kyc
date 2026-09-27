"""Automatic, lifecycle-only face-match orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable

from kyc_engine.face_comparison import FaceComparisonError

from .face_comparison import FaceComparisonService
from .models import FaceMatchStatus
from .sessions import SessionConflict, SessionNotFound, SessionSnapshot, SessionStore


logger = logging.getLogger(__name__)


class VerificationOrchestrator:
    """Run a configured face comparison only after an atomic READY claim."""

    def __init__(
        self,
        *,
        store: SessionStore,
        face_comparison_service: FaceComparisonService,
        snapshot_publisher: Callable[[SessionSnapshot, str], None] | None = None,
    ) -> None:
        self._store = store
        self._face_comparison_service = face_comparison_service
        self._snapshot_publisher = snapshot_publisher

    def on_document_state_changed(self, snapshot: SessionSnapshot) -> None:
        self.attempt_face_match(snapshot.session_id)

    def on_liveness_state_changed(self, snapshot: SessionSnapshot) -> None:
        self.attempt_face_match(snapshot.session_id)

    def attempt_face_match(self, session_id: str) -> SessionSnapshot | None:
        """Claim ready work, then compare outside SessionStore's lock."""
        try:
            snapshot = self._store.get(session_id)
        except SessionNotFound:
            return None
        if snapshot.face_match_status != FaceMatchStatus.READY:
            return snapshot
        try:
            started = self._store.start_face_match(session_id)
        except (SessionConflict, SessionNotFound):
            return None
        self._publish(started, "face_match.started")

        try:
            comparison = self._face_comparison_service.compare(session_id)
        except FaceComparisonError as exc:
            return self._fail(session_id, exc.code)
        except Exception as exc:
            logger.error(
                "face comparison failed unexpectedly",
                extra={
                    "event": "face_match.failed",
                    "error_code": "FACE_COMPARISON_FAILED",
                    "exception_type": type(exc).__name__,
                },
            )
            return self._fail(session_id, "FACE_COMPARISON_FAILED")

        try:
            completed = self._store.complete_face_match(session_id, comparison.similarity)
        except (TypeError, ValueError):
            logger.error(
                "face comparison returned an invalid result",
                extra={
                    "event": "face_match.failed",
                    "error_code": "FACE_COMPARISON_FAILED",
                },
            )
            return self._fail(session_id, "FACE_COMPARISON_FAILED")
        if completed is None:
            return None
        self._publish(completed, "face_match.completed")
        self._release_artifacts(session_id)

        # A delete or expiry after the completed transition is a benign race.
        final = self._store.refresh_verification_completion(session_id)
        if final is not None:
            self._publish(final, "verification.completed")
            return final
        return completed

    def _fail(self, session_id: str, code: str) -> SessionSnapshot | None:
        try:
            failed = self._store.fail_face_match(session_id, code)
        except ValueError:
            # An unexpected subclass must not smuggle an arbitrary error message into state.
            failed = self._store.fail_face_match(session_id, "FACE_COMPARISON_FAILED")
        if failed is None:
            return None
        self._publish(failed, "face_match.failed")
        self._release_artifacts(session_id)
        return failed

    def _release_artifacts(self, session_id: str) -> None:
        try:
            self._store.release_face_match_biometric_artifacts(session_id)
        except (SessionNotFound, SessionConflict):
            # Deletion/expiry has already released sensitive artifacts.
            return
        except Exception as exc:
            logger.error(
                "face-match artifact cleanup failed",
                extra={
                    "event": "face_match.cleanup_failed",
                    "exception_type": type(exc).__name__,
                },
            )

    def _publish(self, snapshot: SessionSnapshot, transition_reason: str) -> None:
        if self._snapshot_publisher is None:
            return
        try:
            self._snapshot_publisher(snapshot, transition_reason)
        except Exception as exc:
            logger.error(
                "webhook event enqueue failed",
                extra={
                    "event": "webhook_enqueue_failed",
                    "exception_type": type(exc).__name__,
                },
            )
