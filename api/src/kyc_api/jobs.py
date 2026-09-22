from __future__ import annotations

import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from threading import BoundedSemaphore, Timer
from time import perf_counter
from typing import Callable

from kyc_engine import DocumentCoordinator

from .metrics import MetricsRegistry
from .sessions import SessionSnapshot, SessionStore, SessionStoreError


logger = logging.getLogger(__name__)


class JobCapacityExceeded(RuntimeError):
    pass


class JobManager:
    """Submit extraction work without allowing an unbounded executor queue."""

    def __init__(
        self,
        coordinator: DocumentCoordinator,
        store: SessionStore,
        *,
        workers: int,
        capacity: int,
        timeout_seconds: int = 30,
        metrics: MetricsRegistry | None = None,
        executor: Executor | None = None,
        timer_factory: Callable[[float, Callable[[], None]], Timer] = Timer,
    ) -> None:
        if workers <= 0 or capacity <= 0 or timeout_seconds <= 0:
            raise ValueError("workers, capacity, and timeout_seconds must be positive")
        self._coordinator = coordinator
        self._store = store
        self._executor = executor or ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="kyc-extraction",
        )
        self._owns_executor = executor is None
        self._slots = BoundedSemaphore(capacity)
        self._timeout_seconds = timeout_seconds
        self._metrics = metrics or MetricsRegistry()
        self._timer_factory = timer_factory

    def submit(self, session_id: str) -> SessionSnapshot:
        if not self._slots.acquire(blocking=False):
            raise JobCapacityExceeded("Job capacity has been reached")
        queued: SessionSnapshot | None = None
        try:
            queued = self._store.queue(session_id)
            assert queued.job_id is not None
            self._executor.submit(self._run, session_id, queued.job_id)
            return queued
        except SessionStoreError:
            self._slots.release()
            raise
        except Exception as exc:
            logger.exception(
                "processing job submission failed",
                extra={
                    "event": "processing_job_submission_failed",
                    "session_id": session_id,
                    "job_id": queued.job_id if queued is not None else None,
                    "error_code": "JOB_SUBMISSION_FAILED",
                    "exception_type": type(exc).__name__,
                },
            )
            if queued is not None and queued.job_id is not None:
                self._store.fail(session_id, queued.job_id, "JOB_SUBMISSION_FAILED")
            self._slots.release()
            raise

    def shutdown(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def _run(self, session_id: str, job_id: str) -> None:
        started = perf_counter()
        timeout_timer: Timer | None = None
        try:
            front, back = self._store.start(session_id, job_id)
            logger.info(
                "processing job started",
                extra={
                    "event": "processing_job_started",
                    "session_id": session_id,
                    "job_id": job_id,
                    "sides": [
                        side
                        for side, source in (("front", front), ("back", back))
                        if source is not None
                    ],
                },
            )
            timeout_timer = self._timer_factory(
                self._timeout_seconds,
                lambda: self._handle_timeout(session_id, job_id),
            )
            timeout_timer.daemon = True
            timeout_timer.start()
            result = self._coordinator.process(front=front, back=back)
            if self._store.complete(session_id, job_id, result):
                duration_ms = (perf_counter() - started) * 1000.0
                self._metrics.record_job(
                    result.status.value,
                    duration_ms,
                )
                logger.info(
                    "processing job completed",
                    extra={
                        "event": "processing_job_completed",
                        "session_id": session_id,
                        "job_id": job_id,
                        "status": result.status.value,
                        "duration_ms": round(duration_ms, 3),
                    },
                )
        except Exception as exc:
            duration_ms = (perf_counter() - started) * 1000.0
            logger.exception(
                "processing job failed",
                extra={
                    "event": "processing_job_failed",
                    "session_id": session_id,
                    "job_id": job_id,
                    "error_code": "PROCESSING_FAILED",
                    "duration_ms": round(duration_ms, 3),
                    "exception_type": type(exc).__name__,
                },
            )
            if self._store.fail(session_id, job_id, "PROCESSING_FAILED"):
                self._metrics.record_job("failed", duration_ms)
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()
            self._slots.release()

    def _handle_timeout(self, session_id: str, job_id: str) -> None:
        if self._store.timeout(session_id, job_id):
            self._metrics.record_error("JOB_TIMEOUT")
            self._metrics.record_job("timeout", self._timeout_seconds * 1000.0)
            logger.warning(
                "processing job timed out",
                extra={
                    "event": "processing_job_timed_out",
                    "session_id": session_id,
                    "job_id": job_id,
                    "error_code": "JOB_TIMEOUT",
                    "duration_ms": self._timeout_seconds * 1000.0,
                },
            )
