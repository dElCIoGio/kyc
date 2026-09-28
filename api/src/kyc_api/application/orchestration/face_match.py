"""Automatic, lifecycle-only face-match orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from threading import BoundedSemaphore, RLock

from kyc_engine.face_comparison import FaceComparisonError

from ..verification.face_comparison import FaceComparisonService
from ...domain.verification import FaceMatchStatus
from ..sessions.store import SessionConflict, SessionNotFound, SessionSnapshot, SessionStore


logger = logging.getLogger("kyc_api.orchestration")


@dataclass
class _ScheduledSession:
    """One coalesced worker task for an active session."""

    attempt: Callable[[str], object]
    follow_up_needed: bool = False


class FaceMatchDispatcher:
    """Bounded, coalescing delivery of in-process face-match attempts.

    ThreadPoolExecutor itself has an unbounded queue.  The registry and slot
    gate below are the actual bound: at most one task per distinct session and
    no more than ``capacity`` distinct tasks can be queued or running.
    """

    def __init__(
        self,
        *,
        workers: int,
        capacity: int,
        executor: Executor | None = None,
    ) -> None:
        if workers <= 0 or capacity <= 0 or workers > capacity:
            raise ValueError("workers must be positive and no greater than capacity")
        if executor is None:
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="kyc-face-match",
            )
            self._owns_executor = True
        else:
            self._owns_executor = False
        self._executor = executor
        self._slots = BoundedSemaphore(capacity)
        self._lock = RLock()
        self._scheduled: dict[str, _ScheduledSession] = {}
        self._closed = False

    def schedule(self, session_id: str, attempt: Callable[[str], object]) -> bool:
        """Coalesce a trigger, or submit one bounded task for its session."""
        with self._lock:
            if self._closed:
                self._log_schedule_failure("FaceMatchDispatcherClosed", "FACE_MATCH_SCHEDULER_CLOSED")
                return False
            active = self._scheduled.get(session_id)
            if active is not None:
                active.follow_up_needed = True
                return True
            if not self._slots.acquire(blocking=False):
                self._log_schedule_failure(
                    "FaceMatchCapacityExceeded", "FACE_MATCH_SCHEDULING_CAPACITY_EXCEEDED"
                )
                return False
            scheduled = _ScheduledSession(attempt=attempt)
            self._scheduled[session_id] = scheduled

        return self._submit(session_id, scheduled)

    def shutdown(self) -> None:
        """Stop new deliveries and drain application-owned worker threads."""
        with self._lock:
            self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def _run(
        self,
        session_id: str,
        scheduled: _ScheduledSession,
    ) -> None:
        while True:
            try:
                scheduled.attempt(session_id)
            except Exception as exc:
                logger.error(
                    "face-match orchestration task failed",
                    extra={
                        "event": "face_match.execution_failed",
                        "error_code": "FACE_MATCH_EXECUTION_FAILED",
                        "exception_type": type(exc).__name__,
                    },
                )
            if not self._consume_follow_up(session_id, scheduled):
                return

    def _consume_follow_up(self, session_id: str, scheduled: _ScheduledSession) -> bool:
        with self._lock:
            if self._scheduled.get(session_id) is not scheduled:
                return False
            if scheduled.follow_up_needed:
                scheduled.follow_up_needed = False
                return True
            return False

    def _submit(self, session_id: str, scheduled: _ScheduledSession) -> bool:
        try:
            future = self._executor.submit(self._run, session_id, scheduled)
        except Exception as exc:
            self._release(session_id, scheduled)
            self._log_schedule_failure(type(exc).__name__, "FACE_MATCH_SCHEDULING_FAILED")
            return False
        future.add_done_callback(
            lambda completed: self._observe_future(session_id, scheduled, completed)
        )
        return True

    def _release(self, session_id: str, scheduled: _ScheduledSession) -> None:
        with self._lock:
            if self._scheduled.get(session_id) is scheduled:
                del self._scheduled[session_id]
                self._slots.release()

    def _observe_future(
        self,
        session_id: str,
        scheduled: _ScheduledSession,
        future: Future[object],
    ) -> None:
        if future.cancelled():
            self._release(session_id, scheduled)
            self._log_schedule_failure("CancelledError", "FACE_MATCH_EXECUTION_CANCELLED")
            return
        try:
            escaped = future.exception()
        except Exception as exc:
            self._release(session_id, scheduled)
            self._log_schedule_failure(type(exc).__name__, "FACE_MATCH_EXECUTION_FAILED")
            return
        if escaped is not None:
            self._release(session_id, scheduled)
            self._log_schedule_failure(type(escaped).__name__, "FACE_MATCH_EXECUTION_FAILED")
            return
        self._resubmit_follow_up_if_needed(session_id, scheduled)

    def _resubmit_follow_up_if_needed(
        self, session_id: str, scheduled: _ScheduledSession
    ) -> None:
        with self._lock:
            if self._scheduled.get(session_id) is not scheduled:
                return
            if self._closed:
                del self._scheduled[session_id]
                self._slots.release()
                return
            if not scheduled.follow_up_needed:
                del self._scheduled[session_id]
                self._slots.release()
                return
            scheduled.follow_up_needed = False
        self._submit(session_id, scheduled)

    @staticmethod
    def _log_schedule_failure(exception_type: str, error_code: str) -> None:
        logger.error(
            "face-match orchestration scheduling failed",
            extra={
                "event": "face_match.scheduling_failed",
                "error_code": error_code,
                "exception_type": exception_type,
            },
        )


class VerificationOrchestrator:
    """Run a configured face comparison only after an atomic READY claim."""

    def __init__(
        self,
        *,
        store: SessionStore,
        face_comparison_service: FaceComparisonService,
        dispatcher: FaceMatchDispatcher,
        snapshot_publisher: Callable[[SessionSnapshot, str], None] | None = None,
    ) -> None:
        self._store = store
        self._face_comparison_service = face_comparison_service
        self._dispatcher = dispatcher
        self._snapshot_publisher = snapshot_publisher

    def on_document_state_changed(self, snapshot: SessionSnapshot) -> None:
        self._dispatcher.schedule(snapshot.session_id, self.attempt_face_match)

    def on_liveness_state_changed(self, snapshot: SessionSnapshot) -> None:
        self._dispatcher.schedule(snapshot.session_id, self.attempt_face_match)

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

        if completed.verification_status.value == "completed":
            self._publish(completed, "verification.completed")
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
        if failed.face_match_required:
            self._publish(failed, "verification.failed")
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
