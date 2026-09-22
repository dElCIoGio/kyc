from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from threading import BoundedSemaphore, Timer
from time import perf_counter
from typing import Callable

from kyc_engine import DocumentCoordinator

from .metrics import MetricsRegistry
from .sessions import SessionSnapshot, SessionStore


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
        except Exception:
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
            timeout_timer = self._timer_factory(
                self._timeout_seconds,
                lambda: self._handle_timeout(session_id, job_id),
            )
            timeout_timer.daemon = True
            timeout_timer.start()
            result = self._coordinator.process(front=front, back=back)
            if self._store.complete(session_id, job_id, result):
                self._metrics.record_job(
                    result.status.value,
                    (perf_counter() - started) * 1000.0,
                )
        except Exception:
            if self._store.fail(session_id, job_id, "PROCESSING_FAILED"):
                self._metrics.record_job("failed", (perf_counter() - started) * 1000.0)
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()
            self._slots.release()

    def _handle_timeout(self, session_id: str, job_id: str) -> None:
        if self._store.timeout(session_id, job_id):
            self._metrics.record_error("JOB_TIMEOUT")
            self._metrics.record_job("timeout", self._timeout_seconds * 1000.0)
