from __future__ import annotations

import threading
import unittest

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
from kyc_api.orchestration import VerificationOrchestrator
from kyc_api.sessions import SessionConflict, SessionStore
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


class FaceMatchOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = InMemoryPortraitArtifactStore()
        self.store = SessionStore(
            ttl_seconds=60,
            max_sessions=4,
            portrait_artifacts=self.artifacts,
            face_match_enabled=True,
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
        self.assertEqual(3, len({event[1] for event in events}))
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
            store=self.store, face_comparison_service=comparison  # type: ignore[arg-type]
        )
        document, document_artifact = self._document_passed(session_id)
        orchestrator.on_document_state_changed(document)
        liveness, live_artifact = self._liveness_passed(session_id)
        orchestrator.on_liveness_state_changed(liveness)

        record = self.store._records[session_id]
        self.assertEqual(FaceMatchStatus.FAILED, record.face_match.status)
        self.assertEqual("FACE_COMPARISON_FAILED", record.face_match.error_code)
        self.assertIsNone(record.face_match.similarity)
        self.assertEqual(VerificationStatus.IN_PROGRESS, record.status)
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
            store=self.store, face_comparison_service=comparison  # type: ignore[arg-type]
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

    def test_disabled_matching_never_completes_verification(self) -> None:
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
        self.assertEqual(VerificationStatus.IN_PROGRESS, snapshot.verification_status)

    def test_cleanup_failure_preserves_successful_match_and_final_completion(self) -> None:
        session_id = self.store.create().session_id
        document, _ = self._document_passed(session_id)
        liveness, _ = self._liveness_passed(session_id)
        comparison = _Comparison()
        orchestrator = VerificationOrchestrator(
            store=self.store, face_comparison_service=comparison  # type: ignore[arg-type]
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

    def test_result_appends_safe_face_match_metadata_without_webhook_score(self) -> None:
        session_id = self.store.create().session_id
        _, _ = self._document_passed(session_id)
        _, _ = self._liveness_passed(session_id)
        self.store.start_face_match(session_id)
        completed = self.store.complete_face_match(session_id, 0.6487)
        assert completed is not None
        self.store.refresh_verification_completion(session_id)
        event_data = WebhookEvent.from_snapshot(completed, transition_reason="face_match.completed").payload()
        self.assertNotIn(b"0.6487", event_data)
        with TestClient(
            create_app(
                settings=settings(), coordinator=FakeCoordinator(), session_store=self.store
            )
        ) as client:
            response = client.get(f"/v1/sessions/{session_id}/result", headers=AUTH_HEADERS)
        self.assertEqual(200, response.status_code)
        self.assertEqual("1.1", response.json()["schema_version"])
        self.assertEqual(
            {"status": "completed", "similarity": 0.6487}, response.json()["face_match"]
        )


if __name__ == "__main__":
    unittest.main()
