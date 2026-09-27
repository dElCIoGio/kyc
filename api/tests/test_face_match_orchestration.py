from __future__ import annotations

import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from fastapi.testclient import TestClient

from kyc_engine import (
    DocumentExtractionResult,
    FaceComparisonResult,
    KycExtractionResult,
    LivenessResult,
    PortraitExtractionResult,
    PortraitStatus,
    ProcessingStatus,
)
from kyc_engine.face_comparison import FaceComparisonError
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore
from kyc_api.main import create_app
from kyc_api.models import FaceMatchStatus, VerificationStatus
from kyc_api.orchestration import FaceMatchDispatcher, VerificationOrchestrator
from kyc_api.sessions import SessionConflict, SessionStore
from kyc_api.verification import VerificationManager
from kyc_api.jobs import JobManager
from kyc_api.webhooks import WebhookEvent

from helpers import AUTH_HEADERS, FakeCoordinator, accept_document, settings


def _document_result(artifact_id: str) -> DocumentExtractionResult:
    portrait = PortraitExtractionResult(
        status=PortraitStatus.AVAILABLE,
        requested_region=None,
        clamped_region=None,
        face_detected=True,
        face_count=1,
        face=None,
        eligible_for_face_match=True,
        warnings=("PORTRAIT_OK",),
        artifact_id=artifact_id,
    )
    front = KycExtractionResult(
        schema_version="1.2",
        processing_id="safe-front",
        status=ProcessingStatus.SUCCESS,
        document_type="ao_id_card",
        side="front",
        profile_id="ao_id_card/front/v1",
        detection=None,
        fields={},
        issues=(),
        timings_ms={},
        portrait=portrait,
    )
    return DocumentExtractionResult(
        schema_version="1.1",
        status=ProcessingStatus.SUCCESS,
        front=front,
        back=None,
        issues=(),
        timings_ms={},
    )


class _Comparison:
    def __init__(self, *, error: FaceComparisonError | None = None) -> None:
        self.calls = 0
        self.error = error
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def compare(self, _session_id: str) -> FaceComparisonResult:
        self.calls += 1
        self.started.set()
        if self.block:
            self.release.wait(timeout=2)
        if self.error is not None:
            raise self.error
        return FaceComparisonResult(similarity=0.6487)


class _InlineExecutor:
    """A deterministic executor for lifecycle tests that do not test dispatch."""

    def submit(self, function, *args, **kwargs) -> Future:
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class _QueuedExecutor:
    def __init__(self) -> None:
        self.tasks: list[tuple[Future, object, tuple, dict]] = []
        self.submissions = 0

    def submit(self, function, *args, **kwargs) -> Future:
        future = Future()
        self.tasks.append((future, function, args, kwargs))
        self.submissions += 1
        return future

    def run_next(self, *, before_completion=None) -> None:
        future, function, args, kwargs = self.tasks.pop(0)
        try:
            result = function(*args, **kwargs)
            if before_completion is not None:
                before_completion()
            future.set_result(result)
        except BaseException as exc:
            future.set_exception(exc)


class _RejectingExecutor:
    def submit(self, *_args, **_kwargs) -> Future:
        raise RuntimeError("private executor failure")


class FaceMatchOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = InMemoryPortraitArtifactStore()
        self.store = SessionStore(
            ttl_seconds=60,
            max_sessions=4,
            portrait_artifacts=self.artifacts,
            face_match_enabled=True,
        )
        self.dispatcher = FaceMatchDispatcher(
            workers=1, capacity=4, executor=_InlineExecutor()
        )

    def _document_passed(
        self, session_id: str, *, captures_already_accepted: bool = False
    ) -> tuple[object, str]:
        if not captures_already_accepted:
            accept_document(self.store, session_id)
        job_id = self.store.queue_document_processing(session_id).document.job_id
        assert job_id is not None
        self.store.start_document_processing(session_id, job_id)
        artifact_id = self.artifacts.put(np.zeros((32, 32, 3), dtype=np.uint8))
        completed = self.store.complete_document_processing(
            session_id, job_id, _document_result(artifact_id)
        )
        assert completed is not None
        return completed, artifact_id

    def _liveness_passed(self, session_id: str) -> tuple[object, str]:
        self.store.start_liveness(session_id)
        artifact_id = self.artifacts.put(np.ones((32, 32, 3), dtype=np.uint8))
        completed = self.store.complete_liveness(
            session_id, LivenessResult(True, 0.9, 3, 3), artifact_id, True
        )
        assert completed is not None
        return completed, artifact_id

    def test_readiness_is_monotonic_and_requires_both_completion_orders(self) -> None:
        first = self.store.create().session_id
        document, _ = self._document_passed(first)
        self.assertEqual(FaceMatchStatus.BLOCKED, document.face_match_status)
        liveness, _ = self._liveness_passed(first)
        self.assertEqual(FaceMatchStatus.READY, liveness.face_match_status)

        second = self.store.create().session_id
        accept_document(self.store, second)
        liveness, _ = self._liveness_passed(second)
        self.assertEqual(FaceMatchStatus.BLOCKED, liveness.face_match_status)
        document, _ = self._document_passed(second, captures_already_accepted=True)
        self.assertEqual(FaceMatchStatus.READY, document.face_match_status)
        self.store.start_face_match(second)
        self.assertEqual(FaceMatchStatus.PROCESSING, self.store.get(second).face_match_status)

    def test_success_orders_events_completes_verification_and_releases_artifacts(self) -> None:
        session_id = self.store.create().session_id
        events: list[tuple[str, int]] = []
        comparison = _Comparison()
        orchestrator = VerificationOrchestrator(
            store=self.store,
            face_comparison_service=comparison,  # type: ignore[arg-type]
            dispatcher=self.dispatcher,
            snapshot_publisher=lambda snapshot, reason: events.append((reason, snapshot.event_sequence)),
        )
        document, document_artifact = self._document_passed(session_id)
        orchestrator.on_document_state_changed(document)
        liveness, live_artifact = self._liveness_passed(session_id)
        orchestrator.on_liveness_state_changed(liveness)

        snapshot = self.store.get(session_id)
        self.assertEqual(1, comparison.calls)
        self.assertEqual(VerificationStatus.COMPLETED, snapshot.verification_status)
        self.assertEqual(FaceMatchStatus.COMPLETED, snapshot.face_match_status)
        self.assertEqual(0.6487, self.store.face_match_result(session_id).similarity)
        self.assertEqual(
            ["face_match.started", "face_match.completed", "verification.completed"],
            [event[0] for event in events],
        )
        self.assertEqual(2, len({event[1] for event in events}))
        self.assertEqual(events[-2][1], events[-1][1])
        self.assertFalse(self.artifacts.exists(document_artifact))
        self.assertFalse(self.artifacts.exists(live_artifact))
        portrait = self.store.result(session_id).front.portrait
        assert portrait is not None
        self.assertIsNone(portrait.artifact_id)
        self.assertTrue(portrait.eligible_for_face_match)
        self.assertEqual(("PORTRAIT_OK",), portrait.warnings)
        self.assertEqual(FaceMatchStatus.COMPLETED, self.store.get(session_id).face_match_status)
        self.assertIsNone(self.store.refresh_verification_completion(session_id))
        with self.assertRaises(SessionConflict):
            self.store.start_liveness(session_id)

    def test_face_match_transition_validation_rejects_nonfinite_scores(self) -> None:
        session_id = self.store.create().session_id
        _, _ = self._document_passed(session_id)
        _, _ = self._liveness_passed(session_id)
        self.store.start_face_match(session_id)
        with self.assertRaises(ValueError):
            self.store.complete_face_match(session_id, float("nan"))
        self.assertEqual(FaceMatchStatus.PROCESSING, self.store.get(session_id).face_match_status)
        failed = self.store.fail_face_match(session_id, "FACE_COMPARISON_FAILED")
        assert failed is not None
        self.assertEqual(FaceMatchStatus.FAILED, failed.face_match_status)
        self.assertIsNone(self.store.complete_face_match(session_id, 0.1))

    def test_known_failure_is_terminal_safe_and_releases_artifacts(self) -> None:
        session_id = self.store.create().session_id
        comparison = _Comparison(error=FaceComparisonError("unavailable"))
        orchestrator = VerificationOrchestrator(
            store=self.store,
            face_comparison_service=comparison,  # type: ignore[arg-type]
            dispatcher=self.dispatcher,
        )
        document, document_artifact = self._document_passed(session_id)
        orchestrator.on_document_state_changed(document)
        liveness, live_artifact = self._liveness_passed(session_id)
        orchestrator.on_liveness_state_changed(liveness)

        record = self.store._records[session_id]
        self.assertEqual(FaceMatchStatus.FAILED, record.face_match.status)
        self.assertEqual("FACE_COMPARISON_FAILED", record.face_match.error_code)
        self.assertIsNone(record.face_match.similarity)
        self.assertEqual(VerificationStatus.FAILED, record.status)
        self.assertFalse(self.artifacts.exists(document_artifact))
        self.assertFalse(self.artifacts.exists(live_artifact))
        with self.assertRaises(ValueError):
            self.store.fail_face_match(session_id, "external exception text")

    def test_concurrent_attempts_claim_and_compare_once(self) -> None:
        session_id = self.store.create().session_id
        _, _ = self._document_passed(session_id)
        liveness, _ = self._liveness_passed(session_id)
        comparison = _Comparison()
        comparison.block = True
        orchestrator = VerificationOrchestrator(
            store=self.store,
            face_comparison_service=comparison,  # type: ignore[arg-type]
            dispatcher=self.dispatcher,
        )
        first = threading.Thread(
            target=orchestrator.on_liveness_state_changed, args=(liveness,)
        )
        first.start()
        self.assertTrue(comparison.started.wait(timeout=1))
        orchestrator.attempt_face_match(session_id)
        comparison.release.set()
        first.join(timeout=1)
        self.assertEqual(1, comparison.calls)
        self.assertEqual(FaceMatchStatus.COMPLETED, self.store.get(session_id).face_match_status)

    def test_document_callback_schedules_without_running_comparison_inline(self) -> None:
        session_id = self.store.create().session_id
        document, _ = self._document_passed(session_id)
        self._liveness_passed(session_id)
        comparison = _Comparison()
        executor = _QueuedExecutor()
        dispatcher = FaceMatchDispatcher(workers=1, capacity=4, executor=executor)
        orchestrator = VerificationOrchestrator(
            store=self.store,
            face_comparison_service=comparison,  # type: ignore[arg-type]
            dispatcher=dispatcher,
        )

        orchestrator.on_document_state_changed(document)

        self.assertEqual(0, comparison.calls)
        self.assertEqual(1, len(executor.tasks))
        executor.run_next()
        self.assertEqual(1, comparison.calls)

    def test_liveness_first_document_trigger_automatically_matches(self) -> None:
        session_id = self.store.create().session_id
        accept_document(self.store, session_id)
        liveness, _ = self._liveness_passed(session_id)
        comparison = _Comparison()
        orchestrator = VerificationOrchestrator(
            store=self.store,
            face_comparison_service=comparison,  # type: ignore[arg-type]
            dispatcher=self.dispatcher,
        )
        orchestrator.on_liveness_state_changed(liveness)
        document, _ = self._document_passed(session_id, captures_already_accepted=True)
        orchestrator.on_document_state_changed(document)

        self.assertEqual(1, comparison.calls)
        self.assertEqual(FaceMatchStatus.COMPLETED, self.store.get(session_id).face_match_status)
        self.assertEqual(VerificationStatus.COMPLETED, self.store.get(session_id).verification_status)

    def test_distinct_scheduled_sessions_are_bounded_by_capacity(self) -> None:
        executor = _QueuedExecutor()
        dispatcher = FaceMatchDispatcher(workers=1, capacity=1, executor=executor)

        self.assertTrue(dispatcher.schedule("first", lambda _session_id: None))
        self.assertFalse(dispatcher.schedule("second", lambda _session_id: None))
        self.assertEqual(1, len(executor.tasks))
        executor.run_next()
        self.assertTrue(dispatcher.schedule("second", lambda _session_id: None))

    def test_shutdown_rejects_active_session_trigger_and_drains_original_task(self) -> None:
        started = threading.Event()
        release = threading.Event()
        calls = 0
        with ThreadPoolExecutor(max_workers=1) as executor:
            dispatcher = FaceMatchDispatcher(workers=1, capacity=1, executor=executor)

            def blocked_attempt(_session_id: str) -> None:
                nonlocal calls
                calls += 1
                started.set()
                self.assertTrue(release.wait(timeout=1))

            self.assertTrue(dispatcher.schedule("session", blocked_attempt))
            self.assertTrue(started.wait(timeout=1))
            dispatcher.shutdown()
            self.assertFalse(dispatcher.schedule("session", blocked_attempt))
            self.assertFalse(dispatcher._scheduled["session"].follow_up_needed)
            release.set()
            executor.submit(lambda: None).result(timeout=1)

            self.assertEqual(1, calls)
            self.assertNotIn("session", dispatcher._scheduled)
            self.assertFalse(dispatcher.schedule("new-session", blocked_attempt))

    def test_closed_dispatcher_drops_deferred_follow_up_without_resubmitting(self) -> None:
        executor = _QueuedExecutor()
        dispatcher = FaceMatchDispatcher(workers=1, capacity=1, executor=executor)
        calls = 0

        def attempt(_session_id: str) -> None:
            nonlocal calls
            calls += 1

        self.assertTrue(dispatcher.schedule("session", attempt))

        def close_with_deferred_follow_up() -> None:
            self.assertTrue(dispatcher.schedule("session", attempt))
            dispatcher.shutdown()

        executor.run_next(before_completion=close_with_deferred_follow_up)

        self.assertEqual(1, calls)
        self.assertEqual(1, executor.submissions)
        self.assertEqual([], executor.tasks)
        self.assertNotIn("session", dispatcher._scheduled)
        self.assertFalse(dispatcher.schedule("session", attempt))

    def test_ready_transition_during_active_attempt_runs_one_coalesced_follow_up(self) -> None:
        session_id = self.store.create().session_id
        document, _ = self._document_passed(session_id)
        comparison = _Comparison()
        first_attempt_finished_claim_check = threading.Event()
        allow_first_attempt_to_exit = threading.Event()
        verification_completed = threading.Event()
        attempt_count = 0
        with ThreadPoolExecutor(max_workers=1) as executor:
            dispatcher = FaceMatchDispatcher(workers=1, capacity=4, executor=executor)
            orchestrator = VerificationOrchestrator(
                store=self.store,
                face_comparison_service=comparison,  # type: ignore[arg-type]
                dispatcher=dispatcher,
                snapshot_publisher=lambda _snapshot, reason: (
                    verification_completed.set()
                    if reason == "verification.completed"
                    else None
                ),
            )
            original_attempt = orchestrator.attempt_face_match

            def gated_attempt(current_session_id: str):
                nonlocal attempt_count
                attempt_count += 1
                result = original_attempt(current_session_id)
                if attempt_count == 1:
                    first_attempt_finished_claim_check.set()
                    self.assertEqual(FaceMatchStatus.BLOCKED, self.store.get(session_id).face_match_status)
                    self.assertTrue(allow_first_attempt_to_exit.wait(timeout=1))
                return result

            orchestrator.attempt_face_match = gated_attempt  # type: ignore[method-assign]
            orchestrator.on_document_state_changed(document)
            self.assertTrue(first_attempt_finished_claim_check.wait(timeout=1))
            liveness, _ = self._liveness_passed(session_id)
            orchestrator.on_liveness_state_changed(liveness)
            allow_first_attempt_to_exit.set()
            self.assertTrue(verification_completed.wait(timeout=1))
            dispatcher.shutdown()

        self.assertEqual(2, attempt_count)
        self.assertEqual(1, comparison.calls)
        self.assertEqual(FaceMatchStatus.COMPLETED, self.store.get(session_id).face_match_status)
        self.assertEqual(VerificationStatus.COMPLETED, self.store.get(session_id).verification_status)

    def test_scheduling_failure_leaves_ready_face_match_recoverable(self) -> None:
        session_id = self.store.create().session_id
        document, _ = self._document_passed(session_id)
        self._liveness_passed(session_id)
        comparison = _Comparison()
        orchestrator = VerificationOrchestrator(
            store=self.store,
            face_comparison_service=comparison,  # type: ignore[arg-type]
            dispatcher=FaceMatchDispatcher(workers=1, capacity=4, executor=_RejectingExecutor()),
        )

        orchestrator.on_document_state_changed(document)

        self.assertEqual(0, comparison.calls)
        self.assertEqual(FaceMatchStatus.READY, self.store.get(session_id).face_match_status)

    def test_unexpected_task_failure_releases_session_for_later_trigger(self) -> None:
        executor = _QueuedExecutor()
        dispatcher = FaceMatchDispatcher(workers=1, capacity=1, executor=executor)
        calls = 0

        def flaky_attempt(_session_id: str) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("private failure")

        self.assertTrue(dispatcher.schedule("session", flaky_attempt))
        executor.run_next()
        self.assertTrue(dispatcher.schedule("session", flaky_attempt))
        executor.run_next()
        self.assertEqual(2, calls)

    def test_liveness_completion_returns_while_comparison_is_blocked(self) -> None:
        session_id = self.store.create().session_id
        self._document_passed(session_id)
        comparison = _Comparison()
        comparison.block = True
        with ThreadPoolExecutor(max_workers=1) as executor:
            dispatcher = FaceMatchDispatcher(workers=1, capacity=4, executor=executor)
            orchestrator = VerificationOrchestrator(
                store=self.store,
                face_comparison_service=comparison,  # type: ignore[arg-type]
                dispatcher=dispatcher,
            )
            jobs = JobManager(FakeCoordinator(), self.store, workers=1, capacity=1)
            manager = VerificationManager(
                assessor=None,  # type: ignore[arg-type]
                store=self.store,
                jobs=jobs,
                orchestrator=orchestrator,
            )
            try:
                manager.start_liveness(session_id)
                live_artifact = self.artifacts.put(np.ones((32, 32, 3), dtype=np.uint8))
                completed = manager.complete_liveness(
                    session_id,
                    LivenessResult(True, 0.9, 3, 3),
                    live_artifact,
                    live_face_eligible=True,
                )
                self.assertIsNotNone(completed)
                self.assertTrue(comparison.started.wait(timeout=1))
                self.assertEqual(FaceMatchStatus.PROCESSING, self.store.get(session_id).face_match_status)
            finally:
                comparison.release.set()
                dispatcher.shutdown()
                jobs.shutdown()

    def test_document_slot_and_job_metrics_finish_before_blocked_comparison(self) -> None:
        first = self.store.create().session_id
        second = self.store.create().session_id
        accept_document(self.store, first)
        accept_document(self.store, second)
        document_artifact = self.artifacts.put(np.zeros((32, 32, 3), dtype=np.uint8))
        live_artifact = self.artifacts.put(np.ones((32, 32, 3), dtype=np.uint8))
        coordinator_started = threading.Event()
        coordinator_release = threading.Event()
        comparison = _Comparison()
        comparison.block = True
        metrics_recorded = threading.Event()

        class _Metrics:
            def record_job(self, *_args) -> None:
                metrics_recorded.set()

            def record_field_statuses(self, _result) -> None:
                return None

            def record_error(self, _code) -> None:
                return None

        coordinator = FakeCoordinator(started=coordinator_started, release=coordinator_release)
        coordinator.output = _document_result(document_artifact)
        with ThreadPoolExecutor(max_workers=1) as document_executor:
            with ThreadPoolExecutor(max_workers=1) as face_executor:
                dispatcher = FaceMatchDispatcher(
                    workers=1, capacity=4, executor=face_executor
                )
                orchestrator = VerificationOrchestrator(
                    store=self.store,
                    face_comparison_service=comparison,  # type: ignore[arg-type]
                    dispatcher=dispatcher,
                )
                jobs = JobManager(
                    coordinator,
                    self.store,
                    workers=1,
                    capacity=1,
                    executor=document_executor,
                    metrics=_Metrics(),  # type: ignore[arg-type]
                    on_document_state_changed=orchestrator.on_document_state_changed,
                )
                try:
                    jobs.submit_document_processing(first)
                    self.assertTrue(coordinator_started.wait(timeout=1))
                    self.store.start_liveness(first)
                    self.store.complete_liveness(
                        first, LivenessResult(True, 0.9, 3, 3), live_artifact, True
                    )
                    coordinator_release.set()
                    self.assertTrue(comparison.started.wait(timeout=1))
                    self.assertTrue(metrics_recorded.is_set())

                    # The blocked comparison cannot retain the single document slot.
                    second_job = jobs.submit_document_processing(second)
                    self.assertEqual("queued", second_job.document.status.value)
                finally:
                    comparison.release.set()
                    jobs.shutdown()
                    dispatcher.shutdown()

    def test_disabled_matching_is_not_required_for_completion(self) -> None:
        artifacts = InMemoryPortraitArtifactStore()
        store = SessionStore(
            ttl_seconds=60, max_sessions=1, portrait_artifacts=artifacts, face_match_enabled=False
        )
        session_id = store.create().session_id
        accept_document(store, session_id)
        job_id = store.queue_document_processing(session_id).document.job_id
        assert job_id is not None
        store.start_document_processing(session_id, job_id)
        portrait = artifacts.put(np.zeros((4, 4, 3), dtype=np.uint8))
        store.complete_document_processing(session_id, job_id, _document_result(portrait))
        store.start_liveness(session_id)
        live = artifacts.put(np.zeros((4, 4, 3), dtype=np.uint8))
        store.complete_liveness(session_id, LivenessResult(True, 0.9, 3, 3), live, True)
        snapshot = store.get(session_id)
        self.assertEqual(FaceMatchStatus.BLOCKED, snapshot.face_match_status)
        self.assertEqual(VerificationStatus.COMPLETED, snapshot.verification_status)

    def test_cleanup_failure_preserves_successful_match_and_final_completion(self) -> None:
        session_id = self.store.create().session_id
        document, _ = self._document_passed(session_id)
        liveness, _ = self._liveness_passed(session_id)
        comparison = _Comparison()
        orchestrator = VerificationOrchestrator(
            store=self.store,
            face_comparison_service=comparison,  # type: ignore[arg-type]
            dispatcher=self.dispatcher,
        )
        original_cleanup = self.store.release_face_match_biometric_artifacts
        cleanup_states: list[FaceMatchStatus] = []

        def cleanup_failure(_session_id: str) -> None:
            cleanup_states.append(self.store.get(session_id).face_match_status)
            raise RuntimeError("artifact backend unavailable")

        self.store.release_face_match_biometric_artifacts = cleanup_failure  # type: ignore[method-assign]
        try:
            orchestrator.on_document_state_changed(document)
            orchestrator.on_liveness_state_changed(liveness)
        finally:
            self.store.release_face_match_biometric_artifacts = original_cleanup  # type: ignore[method-assign]
        self.assertEqual(FaceMatchStatus.COMPLETED, self.store.get(session_id).face_match_status)
        self.assertEqual(0.6487, self.store.face_match_result(session_id).similarity)
        self.assertEqual(VerificationStatus.COMPLETED, self.store.get(session_id).verification_status)
        self.assertEqual([FaceMatchStatus.COMPLETED], cleanup_states)

    def test_result_and_webhook_exclude_face_match_similarity(self) -> None:
        session_id = self.store.create().session_id
        _, _ = self._document_passed(session_id)
        _, _ = self._liveness_passed(session_id)
        self.store.start_face_match(session_id)
        completed = self.store.complete_face_match(session_id, 0.6487)
        assert completed is not None
        self.store.refresh_verification_completion(session_id)
        event = WebhookEvent.from_snapshot(completed, transition_reason="verification.completed")
        assert event is not None
        event_data = event.payload()
        self.assertNotIn(b"0.6487", event_data)
        with TestClient(
            create_app(
                settings=settings(), coordinator=FakeCoordinator(), session_store=self.store
            )
        ) as client:
            response = client.get(f"/v1/sessions/{session_id}/result", headers=AUTH_HEADERS)
        self.assertEqual(200, response.status_code)
        self.assertNotIn("similarity", response.text)
        self.assertEqual("completed", response.json()["face_comparison"]["status"])


if __name__ == "__main__":
    unittest.main()
