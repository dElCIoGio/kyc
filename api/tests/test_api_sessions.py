from __future__ import annotations

import io
import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event

from fastapi.testclient import TestClient
from PIL import Image

from kyc_engine import DocumentCaptureAssessor
from kyc_engine.intake import ImageIntake
from kyc_api.jobs import JobCapacityExceeded, JobManager
from kyc_api.main import create_app
from kyc_api.models import DocumentSide
from kyc_api.sessions import SessionConflict, SessionStore, _now
from kyc_api.verification import VerificationManager
from kyc_api.webhooks import WebhookEvent

from helpers import (
    AUTH_HEADERS,
    PNG_BYTES,
    FakeCoordinator,
    accept_document,
    accepted_capture_assessment,
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
            if payload["document"]["status"] in {"passed", "partial", "failed"}:
                return payload
            time.sleep(0.01)
        self.fail("document did not reach a terminal state")

    def test_new_verification_exposes_independent_initial_states(self) -> None:
        session_id = self.create_session()
        payload = self.client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS).json()
        self.assertEqual("in_progress", payload["verification_status"])
        self.assertEqual("awaiting_capture", payload["document"]["status"])
        self.assertEqual("missing", payload["document"]["front_capture"])
        self.assertEqual("missing", payload["document"]["back_capture"])
        self.assertEqual("blocked", payload["liveness"]["status"])
        self.assertEqual("blocked", payload["face_match"]["status"])
        self.assertNotIn("result", payload)

    def test_rejected_capture_remains_missing_and_can_be_retried(self) -> None:
        session_id = self.create_session()
        rejected = self.upload(session_id, "front", _flat_png())
        self.assertEqual(200, rejected.status_code)
        self.assertFalse(rejected.json()["accepted"])
        self.assertIn("low_contrast", {item["code"] for item in rejected.json()["issues"]})
        self.assertEqual("missing", rejected.json()["verification"]["document"]["front_capture"])

        accepted = self.upload(session_id, "front")
        self.assertTrue(accepted.json()["accepted"])
        self.assertEqual("accepted", accepted.json()["verification"]["document"]["front_capture"])

    def test_one_accepted_side_does_not_start_processing(self) -> None:
        session_id = self.create_session()
        response = self.upload(session_id, "back")
        payload = response.json()["verification"]
        self.assertTrue(response.json()["accepted"])
        self.assertEqual("awaiting_capture", payload["document"]["status"])
        self.assertEqual("blocked", payload["liveness"]["status"])
        self.assertEqual([], self.coordinator.calls)

    def test_front_accepted_while_back_remains_missing(self) -> None:
        session_id = self.create_session()
        response = self.upload(session_id, "front")
        payload = response.json()["verification"]
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
        event = WebhookEvent.from_snapshot(
            snapshot, transition_reason="document.capture_accepted"
        )
        payload = json.loads(event.payload())
        data = payload["data"]
        self.assertEqual("kyc.session.updated", payload["type"])
        self.assertEqual("in_progress", data["verification_status"])
        self.assertEqual("awaiting_capture", data["document_status"])
        self.assertEqual("document.capture_accepted", data["transition_reason"])
        self.assertNotIn("metrics", data)
        self.assertNotIn("image", data)

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
            self.assertEqual("ready", current["liveness"]["status"])
            self.assertEqual("blocked", current["face_match"]["status"])
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
            self.assertEqual("passed", terminal["document"]["status"])
            self.assertEqual("in_progress", terminal["verification_status"])
            self.assertTrue(terminal["document"]["result_available"])
            self.assertFalse(store.contains_images(session_id))
            result = client.get(f"/v1/sessions/{session_id}/result", headers=AUTH_HEADERS)
            self.assertEqual(200, result.status_code)
            self.assertEqual("1.0", result.json()["schema_version"])

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

    def _wait_with_client(self, client: TestClient, session_id: str) -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            payload = client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS).json()
            if payload["document"]["status"] in {"passed", "partial", "failed"}:
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


if __name__ == "__main__":
    unittest.main()
