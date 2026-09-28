from __future__ import annotations

import io
import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

from kyc_engine import DocumentCaptureAssessor, LivenessResult
from kyc_engine.intake import ImageIntake
from kyc_api.jobs import JobCapacityExceeded, JobManager
from kyc_api.main import create_app
from kyc_api.models import DocumentSide, VerificationStatus
from kyc_api.sessions import SessionConflict, SessionExpired, SessionStore, _now
from kyc_api.verification import VerificationManager
from kyc_api.webhooks import WebhookEvent

from helpers import (
    AUTH_HEADERS,
    PNG_BYTES,
    FakeCoordinator,
    accept_document,
    accepted_capture_assessment,
    extraction_result,
    settings,
)


def _flat_png(value: int = 128) -> bytes:
    output = io.BytesIO()
    Image.new("L", (320, 240), value).save(output, format="PNG")
    return output.getvalue()


class ApiSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = FakeCoordinator()
        self.client_context = TestClient(
            create_app(settings=settings(), coordinator=self.coordinator)
        )
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def create_session(self) -> str:
        response = self.client.post("/v1/sessions", headers=AUTH_HEADERS)
        self.assertEqual(201, response.status_code)
        return response.json()["session_id"]

    def upload(self, session_id: str, side: str, content: bytes = PNG_BYTES):
        return self.client.post(
            f"/v1/sessions/{session_id}/images/{side}",
            headers=AUTH_HEADERS,
            files={"image": (f"{side}.png", content, "image/png")},
        )

    def wait_for_document_terminal(self, session_id: str) -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            payload = self.client.get(
                f"/v1/sessions/{session_id}", headers=AUTH_HEADERS
            ).json()
            if payload["document"]["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.01)
        self.fail("document did not reach a terminal state")

    def test_new_verification_exposes_independent_initial_states(self) -> None:
        session_id = self.create_session()
        payload = self.client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS).json()
        self.assertEqual("in_progress", payload["status"])
        self.assertEqual("submit_document_front", payload["next_action"])
        self.assertEqual("awaiting_capture", payload["document"]["status"])
        self.assertEqual("missing", payload["document"]["front_capture"])
        self.assertEqual("missing", payload["document"]["back_capture"])
        self.assertEqual("not_available", payload["liveness"]["status"])
        self.assertEqual("not_available", payload["face_comparison"]["status"])
        self.assertNotIn("result", payload)

    def test_rejected_capture_remains_missing_and_can_be_retried(self) -> None:
        session_id = self.create_session()
        rejected = self.upload(session_id, "front", _flat_png())
        self.assertEqual(200, rejected.status_code)
        self.assertFalse(rejected.json()["accepted"])
        self.assertIn("low_contrast", {item["code"] for item in rejected.json()["issues"]})
        self.assertEqual("missing", rejected.json()["session"]["document"]["front_capture"])
        self.assertNotIn("metrics", rejected.json())
        self.assertNotIn("document_area_ratio", rejected.text)
        self.assertNotIn("perspective_score", rejected.text)
        self.assertNotIn("glare_ratio", rejected.text)
        self.assertFalse(self.client_context.app.state.sessions.contains_images(session_id))
        self.assertEqual([], self.coordinator.calls)

        accepted = self.upload(session_id, "front")
        self.assertTrue(accepted.json()["accepted"])
        self.assertEqual("accepted", accepted.json()["session"]["document"]["front_capture"])

    def test_capture_threshold_settings_reject_nonfinite_and_incoherent_values(self) -> None:
        with self.assertRaises(ValueError):
            settings(capture_max_glare_ratio=float("nan"))
        with self.assertRaises(ValueError):
            settings(capture_min_document_area_ratio=1.0)
        with self.assertRaises(ValueError):
            settings(capture_min_brightness=220, capture_max_brightness=220)

    def test_rejected_capture_does_not_publish_a_transition_or_submit_a_job(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id
        jobs = Mock()
        transitions: list[str] = []
        verification = VerificationManager(
            assessor=DocumentCaptureAssessor(ImageIntake()),
            store=store,
            jobs=jobs,
            snapshot_publisher=lambda _snapshot, reason: transitions.append(reason),
        )

        submission = verification.submit_document_capture(
            session_id,
            DocumentSide.FRONT,
            _flat_png(),
        )

        self.assertFalse(submission.assessment.accepted)
        self.assertFalse(store.contains_images(session_id))
        self.assertEqual([], transitions)
        jobs.submit_document_processing.assert_not_called()

    def test_one_accepted_side_does_not_start_processing(self) -> None:
        session_id = self.create_session()
        response = self.upload(session_id, "back")
        payload = response.json()["session"]
        self.assertTrue(response.json()["accepted"])
        self.assertEqual("awaiting_capture", payload["document"]["status"])
        self.assertEqual("not_available", payload["liveness"]["status"])
        self.assertEqual([], self.coordinator.calls)

    def test_front_accepted_while_back_remains_missing(self) -> None:
        session_id = self.create_session()
        response = self.upload(session_id, "front")
        payload = response.json()["session"]
        self.assertTrue(response.json()["accepted"])
        self.assertEqual("accepted", payload["document"]["front_capture"])
        self.assertEqual("missing", payload["document"]["back_capture"])
        self.assertEqual("awaiting_capture", payload["document"]["status"])

    def test_capture_cannot_change_after_atomic_capture_completion(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id
        accept_document(store, session_id)
        with self.assertRaises(SessionConflict):
            store.accept_document_capture(
                session_id,
                DocumentSide.FRONT,
                PNG_BYTES,
                assessment=accepted_capture_assessment(),
            )

    def test_webhook_state_is_safe_and_identifies_the_transition(self) -> None:
        snapshot = SessionStore(ttl_seconds=60, max_sessions=2).create()
        event = WebhookEvent.from_snapshot(snapshot, transition_reason="verification.session.created")
        assert event is not None
        payload = json.loads(event.payload())
        data = payload["data"]
        self.assertEqual("verification.session.created", payload["type"])
        self.assertEqual("in_progress", data["status"])
        self.assertEqual("submit_document_front", data["next_action"])
        self.assertNotIn("metrics", data)
        self.assertNotIn("image", data)
        self.assertNotIn("job_id", data)

    def test_second_required_capture_unlocks_liveness_and_starts_one_job(self) -> None:
        started, release = Event(), Event()
        coordinator = FakeCoordinator(started=started, release=release)
        with TestClient(create_app(settings=settings(), coordinator=coordinator)) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            client.post(
                f"/v1/sessions/{session_id}/images/front",
                headers=AUTH_HEADERS,
                files={"image": ("front.png", PNG_BYTES, "image/png")},
            )
            response = client.post(
                f"/v1/sessions/{session_id}/images/back",
                headers=AUTH_HEADERS,
                files={"image": ("back.png", PNG_BYTES, "image/png")},
            )
            self.assertTrue(response.json()["accepted"])
            self.assertTrue(started.wait(timeout=1))
            current = client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS).json()
            self.assertEqual("not_available", current["liveness"]["status"])
            self.assertEqual("not_available", current["face_comparison"]["status"])
            self.assertEqual("processing", current["document"]["status"])
            duplicate = client.post(f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS)
            self.assertEqual(202, duplicate.status_code)
            self.assertEqual(1, len(coordinator.calls))
            release.set()

    def test_result_and_sensitive_bytes_are_available_only_after_document_processing(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=4)
        with TestClient(
            create_app(settings=settings(), coordinator=FakeCoordinator(), session_store=store)
        ) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            for side in ("front", "back"):
                client.post(
                    f"/v1/sessions/{session_id}/images/{side}",
                    headers=AUTH_HEADERS,
                    files={"image": (f"{side}.png", PNG_BYTES, "image/png")},
                )
            terminal = self._wait_with_client(client, session_id)
            self.assertEqual("completed", terminal["document"]["status"])
            self.assertEqual("completed", terminal["status"])
            self.assertTrue(terminal["document"]["result_available"])
            self.assertFalse(store.contains_images(session_id))
            result = client.get(f"/v1/sessions/{session_id}/result", headers=AUTH_HEADERS)
            self.assertEqual(200, result.status_code)
            self.assertEqual(session_id, result.json()["session_id"])
            self.assertNotIn("schema_version", result.json())

    def test_process_requires_both_accepted_sides(self) -> None:
        session_id = self.create_session()
        self.upload(session_id, "front")
        response = self.client.post(f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS)
        self.assertEqual(409, response.status_code)

    def test_invalid_metadata_and_capture_input_are_safe(self) -> None:
        session_id = self.create_session()
        bad_side = self.upload(session_id, "inside")
        bad_type = self.client.post(
            f"/v1/sessions/{session_id}/images/front",
            headers=AUTH_HEADERS,
            files={"image": ("front.txt", b"text", "text/plain")},
        )
        bad_signature = self.client.post(
            f"/v1/sessions/{session_id}/images/front",
            headers=AUTH_HEADERS,
            files={"image": ("front.png", b"not-a-png", "image/png")},
        )
        self.assertEqual(422, bad_side.status_code)
        self.assertEqual(415, bad_type.status_code)
        self.assertEqual(415, bad_signature.status_code)

    def test_expiration_and_delete_clear_sensitive_bytes(self) -> None:
        store = SessionStore(ttl_seconds=1, max_sessions=2)
        session_id = store.create().session_id
        accept_document(store, session_id)
        self.assertTrue(store.contains_images(session_id))
        store._records[session_id].expires_at = _now() - timedelta(seconds=1)
        self.assertEqual(1, store.cleanup())
        self.assertFalse(store.contains_images(session_id))

    def test_expiration_removes_awaiting_and_ready_verifications(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=3)
        awaiting = store.create().session_id
        ready = store.create().session_id
        accept_document(store, ready)
        ready_record = store._records[ready]
        for session_id in (awaiting, ready):
            store._records[session_id].expires_at = _now() - timedelta(seconds=1)

        self.assertEqual(2, store.cleanup())
        self.assertNotIn(awaiting, store._records)
        self.assertNotIn(ready, store._records)
        self.assertIsNone(ready_record.document.front)
        self.assertIsNone(ready_record.document.back)
        with self.assertRaises(SessionExpired):
            store.get(ready)

    def test_expiration_invalidates_queued_and_processing_document_work(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=3)
        queued = store.create().session_id
        processing = store.create().session_id
        accept_document(store, queued)
        queued_job = store.queue_document_processing(queued).document.job_id
        assert queued_job is not None
        accept_document(store, processing)
        processing_job = store.queue_document_processing(processing).document.job_id
        assert processing_job is not None
        store.start_document_processing(processing, processing_job)
        processing_record = store._records[processing]
        for session_id in (queued, processing):
            store._records[session_id].expires_at = _now() - timedelta(seconds=1)

        self.assertEqual(2, store.cleanup())
        self.assertIsNone(processing_record.document.front)
        self.assertIsNone(processing_record.document.back)
        self.assertIsNone(store.complete_document_processing(processing, processing_job, extraction_result()))
        with self.assertRaises(SessionExpired):
            store.start_document_processing(queued, queued_job)

    def test_liveness_transitions_are_independent_and_snapshot_safe(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id
        with self.assertRaises(SessionConflict):
            store.start_liveness(session_id)

        accept_document(store, session_id)
        self.assertIsNone(
            store.complete_liveness(
                session_id, LivenessResult(True, 0.9, frames_evaluated=3, real_frames=3)
            )
        )
        running = store.start_liveness(session_id)
        self.assertEqual("processing", running.liveness_status.value)
        with self.assertRaises(SessionConflict):
            store.start_liveness(session_id)

        completed = store.complete_liveness(
            session_id, LivenessResult(True, 0.9, frames_evaluated=3, real_frames=3)
        )
        assert completed is not None
        self.assertEqual("passed", completed.liveness_status.value)
        self.assertEqual("in_progress", completed.verification_status.value)
        self.assertEqual("blocked", completed.face_match_status.value)
        self.assertFalse(hasattr(completed, "liveness_result"))
        with self.assertRaises(SessionConflict):
            store.start_liveness(session_id)

    def test_liveness_failure_and_terminal_verification_do_not_restart(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        failed = store.create().session_id
        accept_document(store, failed)
        store.start_liveness(failed)
        snapshot = store.fail_liveness(failed, "LIVENESS_UNAVAILABLE")
        assert snapshot is not None
        self.assertEqual("failed", snapshot.liveness_status.value)
        self.assertIsNone(store.fail_liveness(failed, "LIVENESS_UNAVAILABLE"))

        terminal = store.create().session_id
        accept_document(store, terminal)
        store._records[terminal].status = VerificationStatus.REJECTED
        with self.assertRaises(SessionConflict):
            store.start_liveness(terminal)
        self.assertIsNone(
            store.complete_liveness(
                terminal, LivenessResult(True, 0.9, frames_evaluated=3, real_frames=3)
            )
        )

    def _wait_with_client(self, client: TestClient, session_id: str) -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            payload = client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS).json()
            if payload["document"]["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.01)
        self.fail("document did not reach a terminal state")


class JobManagerTests(unittest.TestCase):
    def test_submission_capacity_is_bounded(self) -> None:
        started, release = Event(), Event()
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        first, second = store.create().session_id, store.create().session_id
        accept_document(store, first)
        accept_document(store, second)
        coordinator = FakeCoordinator(started=started, release=release)
        with ThreadPoolExecutor(max_workers=1) as executor:
            manager = JobManager(coordinator, store, workers=1, capacity=1, executor=executor)
            manager.submit_document_processing(first)
            self.assertTrue(started.wait(timeout=1))
            with self.assertRaises(JobCapacityExceeded):
                manager.submit_document_processing(second)
            release.set()

    def test_completed_capture_stays_ready_when_automatic_capacity_is_full(self) -> None:
        started, release = Event(), Event()
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        first, second = store.create().session_id, store.create().session_id
        coordinator = FakeCoordinator(started=started, release=release)
        with ThreadPoolExecutor(max_workers=1) as executor:
            jobs = JobManager(coordinator, store, workers=1, capacity=1, executor=executor)
            verification = VerificationManager(
                assessor=DocumentCaptureAssessor(ImageIntake()), store=store, jobs=jobs
            )
            try:
                verification.submit_document_capture(first, DocumentSide.FRONT, PNG_BYTES)
                verification.submit_document_capture(first, DocumentSide.BACK, PNG_BYTES)
                self.assertTrue(started.wait(timeout=1))
                verification.submit_document_capture(second, DocumentSide.FRONT, PNG_BYTES)
                result = verification.submit_document_capture(second, DocumentSide.BACK, PNG_BYTES)
                self.assertEqual("ready", result.snapshot.document.status.value)
                self.assertEqual("ready", result.snapshot.liveness_status.value)
                self.assertTrue(store.contains_images(second))
            finally:
                release.set()
                jobs.shutdown()

    def test_expired_active_job_clears_bytes_and_cannot_recreate_the_session(self) -> None:
        started, release = Event(), Event()
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id
        accept_document(store, session_id)
        coordinator = FakeCoordinator(started=started, release=release)
        with ThreadPoolExecutor(max_workers=1) as executor:
            jobs = JobManager(coordinator, store, workers=1, capacity=1, executor=executor)
            jobs.submit_document_processing(session_id)
            self.assertTrue(started.wait(timeout=1))
            record = store._records[session_id]
            record.expires_at = _now() - timedelta(seconds=1)
            self.assertEqual(1, store.cleanup())
            self.assertIsNone(record.document.front)
            self.assertIsNone(record.document.back)
            release.set()
        self.assertNotIn(session_id, store._records)

    def test_expired_queued_worker_is_discarded_before_processing_starts(self) -> None:
        gate = Event()
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id
        accept_document(store, session_id)
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(gate.wait)
            try:
                jobs = JobManager(FakeCoordinator(), store, workers=1, capacity=1, executor=executor)
                jobs.submit_document_processing(session_id)
                store._records[session_id].expires_at = _now() - timedelta(seconds=1)
                self.assertEqual(1, store.cleanup())
            finally:
                gate.set()
        self.assertNotIn(session_id, store._records)

    def test_verification_manager_publishes_safe_liveness_transitions(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id
        accept_document(store, session_id)
        transitions: list[str] = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            manager = VerificationManager(
                assessor=DocumentCaptureAssessor(ImageIntake()),
                store=store,
                jobs=JobManager(FakeCoordinator(), store, workers=1, capacity=1, executor=executor),
                snapshot_publisher=lambda snapshot, reason: transitions.append(reason),
            )
            manager.start_liveness(session_id)
            manager.complete_liveness(
                session_id, LivenessResult(True, 0.9, frames_evaluated=3, real_frames=3)
            )
        self.assertEqual(["liveness.started", "liveness.passed"], transitions)


if __name__ == "__main__":
    unittest.main()
