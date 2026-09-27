from __future__ import annotations

import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from threading import BoundedSemaphore, Timer
from time import perf_counter
from typing import Callable

from kyc_engine import DocumentCoordinator, ProcessingStatus
from kyc_engine.instrumentation import pipeline_metrics_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, Status, StatusCode

from .metrics import MetricsRegistry
from .logging import job_logging_context
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
        webhook_publisher: Callable[[SessionSnapshot, str], None] | None = None,
        on_document_state_changed: Callable[[SessionSnapshot], None] | None = None,
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
        self._webhook_publisher = webhook_publisher
        self._on_document_state_changed = on_document_state_changed

    def submit_document_processing(self, session_id: str) -> SessionSnapshot:
        if not self._slots.acquire(blocking=False):
            raise JobCapacityExceeded("Job capacity has been reached")
        queued: SessionSnapshot | None = None
        try:
            queued = self._store.queue_document_processing(session_id)
            assert queued.document.job_id is not None
            self._publish(queued, "document.processing_queued")
            self._executor.submit(self._run, session_id, queued.document.job_id)
            return queued
        except SessionStoreError:
            self._slots.release()
            raise
        except Exception as exc:
            logger.exception(
                "processing job submission failed",
                extra={
                    "event": "document.processing_submission_failed",
                    "session_id": session_id,
                    "job_id": queued.document.job_id if queued is not None else None,
                    "error_code": "JOB_SUBMISSION_FAILED",
                    "exception_type": type(exc).__name__,
                },
            )
            if queued is not None and queued.document.job_id is not None:
                failed = self._store.fail_document_processing(
                    session_id, queued.document.job_id, "JOB_SUBMISSION_FAILED"
                )
                if failed is not None:
                    self._publish(failed, "document.failed")
            self._slots.release()
            raise

    def shutdown(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def _run(self, session_id: str, job_id: str) -> None:
        with job_logging_context(session_id=session_id, job_id=job_id):
            with trace.get_tracer(__name__).start_as_current_span(
                "kyc.process_job",
                context=Context(),
                attributes={
                    "kyc.session_id": session_id,
                    "kyc.job_id": job_id,
                },
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                with pipeline_metrics_context(self._metrics):
                    self._run_with_context(session_id, job_id, span)

    def _run_with_context(self, session_id: str, job_id: str, span: Span) -> None:
        started = perf_counter()
        timeout_timer: Timer | None = None
        result = None
        post_document_callback: SessionSnapshot | None = None
        try:
            try:
                front, back, running = self._store.start_document_processing(session_id, job_id)
            except SessionStoreError:
                span.set_attribute("kyc.job_discarded", True)
                return
            self._publish(running, "document.processing_started")
            logger.info(
                "processing job started",
                extra={
                    "event": "document.processing_started",
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
            span.set_attribute("kyc.processing_status", result.status.value)
            completed = self._store.complete_document_processing(session_id, job_id, result)
            if completed is None:
                self._coordinator.release_pending_artifacts(result)
            elif completed.document.status.value == "failed":
                self._coordinator.release_pending_artifacts(result)
                span.set_attribute("kyc.processing_status", "failed")
                self._publish(completed, "document.failed")
                self._metrics.record_job("failed", (perf_counter() - started) * 1000.0)
            else:
                transition_reason = {
                    ProcessingStatus.SUCCESS: "document.passed",
                    ProcessingStatus.PARTIAL: "document.partial",
                    ProcessingStatus.FAILED: "document.failed",
                }[result.status]
                self._publish(completed, transition_reason)
                duration_ms = (perf_counter() - started) * 1000.0
                self._metrics.record_job(
                    result.status.value,
                    duration_ms,
                )
                self._metrics.record_field_statuses(result)
                logger.info(
                    "processing job completed",
                    extra={
                        "event": transition_reason,
                        "status": result.status.value,
                        "duration_ms": round(duration_ms, 3),
                    },
                )
                if result.status == ProcessingStatus.SUCCESS:
                    post_document_callback = completed
        except Exception as exc:
            if result is not None:
                self._coordinator.release_pending_artifacts(result)
            duration_ms = (perf_counter() - started) * 1000.0
            span.set_attribute("exception.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception(
                "processing job failed",
                extra={
                    "event": "document.failed",
                    "error_code": "PROCESSING_FAILED",
                    "duration_ms": round(duration_ms, 3),
                    "exception_type": type(exc).__name__,
                },
            )
            failed = self._store.fail_document_processing(
                session_id, job_id, "PROCESSING_FAILED"
            )
            if failed is not None:
                self._publish(failed, "document.failed")
                self._metrics.record_job("failed", duration_ms)
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()
            self._slots.release()
        if post_document_callback is not None:
            self._notify_document_state_changed(post_document_callback)

    def _handle_timeout(self, session_id: str, job_id: str) -> None:
        with job_logging_context(session_id=session_id, job_id=job_id):
            timed_out = self._store.timeout_document_processing(session_id, job_id)
            if timed_out is not None:
                self._publish(timed_out, "document.failed")
                self._metrics.record_error("JOB_TIMEOUT")
                self._metrics.record_job("timeout", self._timeout_seconds * 1000.0)
                logger.warning(
                    "processing job timed out",
                    extra={
                        "event": "document.failed",
                        "error_code": "JOB_TIMEOUT",
                        "duration_ms": self._timeout_seconds * 1000.0,
                    },
                )

    def _publish(self, snapshot: SessionSnapshot, transition_reason: str) -> None:
        if self._webhook_publisher is None:
            return
        try:
            self._webhook_publisher(snapshot, transition_reason)
        except Exception as exc:
            logger.error(
                "webhook event enqueue failed",
                extra={"event": "webhook_enqueue_failed", "exception_type": type(exc).__name__},
            )

    def _notify_document_state_changed(self, snapshot: SessionSnapshot) -> None:
        if self._on_document_state_changed is None:
            return
        try:
            self._on_document_state_changed(snapshot)
        except Exception as exc:
            logger.error(
                "document-state callback failed",
                extra={
                    "event": "document.state_callback_failed",
                    "exception_type": type(exc).__name__,
                },
            )
