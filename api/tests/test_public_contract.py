from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import unittest

from fastapi.testclient import TestClient

from kyc_engine import LivenessResult
from kyc_api.main import create_app
from kyc_api.models import CaptureStatus, DocumentSide, DocumentStatus, FaceMatchStatus, LivenessStatus
from kyc_api.projection import result_response, session_response
from kyc_api.sessions import SessionStore, _now

from helpers import (
    AUTH_HEADERS,
    PNG_BYTES,
    FakeCoordinator,
    accept_document,
    accepted_capture_assessment,
    extraction_result,
    settings,
)


class PublicSessionContractTests(unittest.TestCase):
    def test_next_action_and_completion_follow_required_check_order(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id

        new = session_response(store.get(session_id))
        self.assertEqual("in_progress", new.status)
        self.assertEqual("submit_document_front", new.next_action)

        front = store.accept_document_capture(
            session_id,
            DocumentSide.FRONT,
            PNG_BYTES,
            accepted_capture_assessment(),
        ).snapshot
        self.assertEqual("submit_document_back", session_response(front).next_action)

        accept_document(store, session_id)
        waiting_for_document = session_response(store.queue_document_processing(session_id))
        self.assertEqual("wait", waiting_for_document.next_action)

        job_id = store.get(session_id).document.job_id
        assert job_id is not None
        store.start_document_processing(session_id, job_id)
        document_complete = store.complete_document_processing(
            session_id, job_id, extraction_result()
        )
        assert document_complete is not None
        self.assertEqual("submit_liveness", session_response(document_complete).next_action)

        store.start_liveness(session_id)
        liveness_complete = store.complete_liveness(
            session_id, LivenessResult(True, 0.9, frames_evaluated=3, real_frames=3)
        )
        assert liveness_complete is not None
        completed = session_response(liveness_complete)
        self.assertEqual("completed", completed.status)
        self.assertIsNone(completed.next_action)

    def test_required_face_comparison_must_finish_or_fail(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=1, face_match_enabled=True)
        session_id = store.create().session_id
        snapshot = store.get(session_id)
        finished_inputs = replace(
            snapshot,
            document=replace(
                snapshot.document,
                status=DocumentStatus.PASSED,
                front_capture=CaptureStatus.ACCEPTED,
                back_capture=CaptureStatus.ACCEPTED,
            ),
            liveness_status=LivenessStatus.PASSED,
            face_match_status=FaceMatchStatus.PROCESSING,
        )
        waiting = session_response(finished_inputs)
        self.assertEqual("in_progress", waiting.status)
        self.assertEqual("wait", waiting.next_action)

        finished = session_response(
            replace(finished_inputs, face_match_status=FaceMatchStatus.COMPLETED)
        )
        self.assertEqual("completed", finished.status)
        self.assertIsNone(finished.next_action)

        failed = session_response(
            replace(finished_inputs, face_match_status=FaceMatchStatus.FAILED)
        )
        self.assertEqual("failed", failed.status)
        self.assertIsNone(failed.next_action)

    def test_result_projection_excludes_engine_and_biometric_fields(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=1, liveness_required=False)
        snapshot = store.create()
        payload = result_response(snapshot, extraction_result()).model_dump()
        encoded = str(payload)
        for forbidden in (
            "schema_version",
            "processing_id",
            "timings_ms",
            "qr_code",
            "portrait",
            "similarity",
            "artifact_id",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_expired_session_returns_410_without_session_body(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=1)
        with TestClient(
            create_app(settings=settings(), coordinator=FakeCoordinator(), session_store=store)
        ) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            store._records[session_id].expires_at = _now() - timedelta(seconds=1)
            response = client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS)

        self.assertEqual(410, response.status_code)
        self.assertEqual("SESSION_EXPIRED", response.json()["error"]["code"])
        self.assertNotIn("session_id", response.json())

    def test_delete_is_idempotent(self) -> None:
        with TestClient(create_app(settings=settings(), coordinator=FakeCoordinator())) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            self.assertEqual(
                204,
                client.delete(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS).status_code,
            )
            self.assertEqual(
                204,
                client.delete(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS).status_code,
            )


if __name__ == "__main__":
    unittest.main()
