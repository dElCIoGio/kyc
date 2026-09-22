from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from threading import BoundedSemaphore

from kyc_engine import DocumentCoordinator

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
        executor: Executor | None = None,
    ) -> None:
        if workers <= 0 or capacity <= 0:
            raise ValueError("workers and capacity must be positive")
        self._coordinator = coordinator
        self._store = store
        self._executor = executor or ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="kyc-extraction",
        )
        self._owns_executor = executor is None
        self._slots = BoundedSemaphore(capacity)

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
        try:
            front, back = self._store.start(session_id, job_id)
            result = self._coordinator.process(front=front, back=back)
            self._store.complete(session_id, job_id, result)
        except Exception:
            self._store.fail(session_id, job_id, "PROCESSING_FAILED")
        finally:
            self._slots.release()
