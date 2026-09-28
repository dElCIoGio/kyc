"""Serialized NIF work and session lifecycle orchestration."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from time import perf_counter
from ...domain.nif import NifVerificationResult, NifVerifier, unavailable_nif_result
from ...infrastructure.observability.metrics import MetricsRegistry
from ...domain.verification import NifVerificationStatus
from ..sessions.store import SessionSnapshot, SessionStore


logger = logging.getLogger("kyc_api.nif")


class NifVerifierCapacityExceeded(RuntimeError):
    code = "NIF_VERIFIER_CAPACITY_EXCEEDED"


@dataclass
class _Task:
    nif: str
    claimed_name: str | None
    future: Future[NifVerificationResult]
    operation: str


class NifVerificationDispatcher:
    """One Playwright owner with bounded priority queues.

    Session tasks always win over standalone work between requests.  The
    browser-affine verifier is created and used solely by this worker thread.
    """

    def __init__(
        self,
        *,
        verifier_factory: Callable[[], NifVerifier],
        session_capacity: int,
        standalone_capacity: int,
        timeout_seconds: float = 20.0,
    ) -> None:
        if session_capacity <= 0 or standalone_capacity <= 0 or timeout_seconds <= 0:
            raise ValueError("NIF dispatcher capacities must be positive")
        self._factory = verifier_factory
        self._sessions: queue.Queue[_Task | None] = queue.Queue(session_capacity)
        self._standalone: queue.Queue[_Task | None] = queue.Queue(standalone_capacity)
        self._stop = threading.Event()
        self._timeout_seconds = timeout_seconds
        self._thread = threading.Thread(target=self._run, name="kyc-nif", daemon=True)
        self._thread.start()

    def submit_session(self, nif: str, claimed_name: str | None) -> Future[NifVerificationResult]:
        return self._submit(self._sessions, nif, claimed_name, "session")

    def submit_standalone(self, nif: str, claimed_name: str | None) -> Future[NifVerificationResult]:
        return self._submit(self._standalone, nif, claimed_name, "standalone")

    def shutdown(self) -> None:
        self._stop.set()
        # A sentinel wakes an idle worker. Queue fullness is harmless: it is
        # already guaranteed to wake after bounded queued work drains.
        try:
            self._sessions.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=self._timeout_seconds + 5)

    def _submit(
        self, target: queue.Queue[_Task | None], nif: str, claimed_name: str | None, operation: str
    ) -> Future[NifVerificationResult]:
        if self._stop.is_set():
            raise NifVerifierCapacityExceeded()
        future: Future[NifVerificationResult] = Future()
        try:
            target.put_nowait(_Task(nif, claimed_name, future, operation))
        except queue.Full as exc:
            raise NifVerifierCapacityExceeded() from exc
        return future

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        verifier = self._factory()
        try:
            while not self._stop.is_set() or not self._sessions.empty() or not self._standalone.empty():
                task = self._next_task()
                if task is None:
                    await asyncio.sleep(0.01)
                    continue
                if task.future.cancelled():
                    continue
                started = perf_counter()
                try:
                    result = await asyncio.wait_for(
                        verifier.verify(task.nif, task.claimed_name),
                        timeout=self._timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    result = unavailable_nif_result("NIF_TIMEOUT")
                except Exception as exc:
                    logger.error(
                        "NIF verification worker failed",
                        extra={
                            "event": "nif.verification_failed",
                            "error_code": "NIF_VERIFICATION_FAILED",
                            "exception_type": type(exc).__name__,
                        },
                    )
                    result = NifVerificationResult(
                        NifVerificationStatus.FAILED,
                        "minfin",
                        error_code="NIF_VERIFICATION_FAILED",
                    )
                # The dispatcher owns the elapsed measurement because it owns
                # the sole browser-affine verifier call.  Retain it only for
                # internal metrics, never in public/session models.
                duration_seconds = max(perf_counter() - started, 1e-9)
                result = replace(result, duration_seconds=duration_seconds)
                if not task.future.cancelled():
                    task.future.set_result(result)
                reset = getattr(verifier, "reset_after_failure", None)
                if result.status == NifVerificationStatus.UNAVAILABLE and callable(reset):
                    try:
                        await reset()
                    except Exception:
                        logger.warning("NIF verifier reset failed", extra={"event": "nif.reset_failed"})
                logger.info(
                    "NIF verification completed",
                    extra={
                        "event": f"nif.verification_{result.status.value}",
                        "provider": result.source,
                        "status": result.status.value,
                        "duration_ms": round(duration_seconds * 1000, 3),
                    },
                )
        finally:
            try:
                close = getattr(verifier, "close", None)
                if callable(close):
                    await close()
            except Exception:
                logger.warning("NIF verifier shutdown failed", extra={"event": "nif.shutdown_failed"})

    def _next_task(self) -> _Task | None:
        try:
            return self._sessions.get_nowait()
        except queue.Empty:
            try:
                return self._standalone.get_nowait()
            except queue.Empty:
                return None


class NifVerificationOrchestrator:
    """Connect completed document state to the shared NIF dispatcher."""

    def __init__(
        self,
        *,
        store: SessionStore,
        dispatcher: NifVerificationDispatcher,
        metrics: MetricsRegistry | None = None,
        snapshot_publisher: Callable[[SessionSnapshot, str], None] | None = None,
    ) -> None:
        self._store = store
        self._dispatcher = dispatcher
        self._metrics = metrics
        self._publisher = snapshot_publisher

    def on_document_state_changed(self, snapshot: SessionSnapshot) -> None:
        claimed = self._store.claim_nif_verification(snapshot.session_id)
        if claimed is None:
            return
        nif, claimed_name, state = claimed
        if not nif:
            # Skipped work never creates a NIF webhook. It may, however, be the
            # final aggregation decision for a no-liveness session.
            self._publish_terminal(state)
            return
        logger.info(
            "NIF verification started",
            extra={"event": "nif.verification_started", "provider": "minfin", "status": "processing"},
        )
        try:
            future = self._dispatcher.submit_session(nif, claimed_name)
        except NifVerifierCapacityExceeded:
            if self._metrics is not None:
                self._metrics.record_nif_capacity_rejected(operation="session")
            settled = self._store.fail_nif_dispatch_capacity(snapshot.session_id)
            if settled is not None:
                self._publish_terminal(settled)
            return
        future.add_done_callback(lambda completed: self._complete(snapshot.session_id, completed))

    def _complete(self, session_id: str, future: Future[NifVerificationResult]) -> None:
        try:
            result = future.result()
        except Exception:
            result = NifVerificationResult(
                NifVerificationStatus.FAILED, "minfin", error_code="NIF_VERIFICATION_FAILED"
            )
        settled = self._store.complete_nif_verification(
            session_id,
            status=result.status,
            source=result.source,
            name_match=result.name_match,
            error_code=result.error_code,
        )
        if settled is None:
            return
        if self._metrics is not None:
            self._metrics.record_nif_verification(
                outcome=result.status.value,
                operation="session",
                duration_seconds=result.duration_seconds,
            )
        self._publish(settled, "nif.completed")
        self._publish_terminal(settled)

    def _publish_terminal(self, snapshot: SessionSnapshot) -> None:
        if snapshot.verification_status.value == "completed":
            self._publish(snapshot, "verification.completed")
        elif snapshot.verification_status.value == "failed":
            self._publish(snapshot, "verification.failed")

    def _publish(self, snapshot: SessionSnapshot, reason: str) -> None:
        if self._publisher is None:
            return
        try:
            self._publisher(snapshot, reason)
        except Exception as exc:
            logger.error(
                "webhook event enqueue failed",
                extra={"event": "webhook_enqueue_failed", "exception_type": type(exc).__name__},
            )
