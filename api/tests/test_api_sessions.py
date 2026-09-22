from __future__ import annotations

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from fastapi.testclient import TestClient

from kyc_engine import ProcessingStatus
from kyc_api.jobs import JobCapacityExceeded, JobManager
from kyc_api.main import create_app
from kyc_api.sessions import SessionStore

from helpers import AUTH_HEADERS, PNG_BYTES, FakeCoordinator, settings


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

    def wait_for_terminal(self, session_id: str) -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            response = self.client.get(
                f"/v1/sessions/{session_id}", headers=AUTH_HEADERS
            )
            payload = response.json()
            if payload["status"] in {"success", "partial", "failed"}:
                return payload
            time.sleep(0.01)
        self.fail("session did not reach a terminal state")

    def test_creates_session_without_sensitive_fields(self) -> None:
        session_id = self.create_session()
        response = self.client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS)
        self.assertEqual("created", response.json()["status"])
        self.assertNotIn("result", response.json())

    def test_uploads_and_replaces_each_side_before_processing(self) -> None:
        session_id = self.create_session()
        self.assertEqual(200, self.upload(session_id, "front").status_code)
        replacement = PNG_BYTES + b"-replacement"
        response = self.upload(session_id, "front", replacement)
        self.assertEqual(200, response.status_code)
        response = self.upload(session_id, "back")
        self.assertEqual(["front", "back"], response.json()["uploaded_sides"])

        queued = self.client.post(
            f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS
        )
        self.assertEqual(202, queued.status_code)
        self.wait_for_terminal(session_id)
        self.assertEqual([(replacement, PNG_BYTES)], self.coordinator.calls)

    def test_front_only_job_returns_partial_library_result(self) -> None:
        coordinator = FakeCoordinator(status=ProcessingStatus.PARTIAL)
        with TestClient(create_app(settings=settings(), coordinator=coordinator)) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            client.post(
                f"/v1/sessions/{session_id}/images/front",
                headers=AUTH_HEADERS,
                files={"image": ("front.png", PNG_BYTES, "image/png")},
            )
            response = client.post(
                f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS
            )
            self.assertEqual("queued", response.json()["status"])
            self._wait_with_client(client, session_id)
            result = client.get(
                f"/v1/sessions/{session_id}/result", headers=AUTH_HEADERS
            )
            self.assertEqual("partial", result.json()["status"])
            self.assertEqual([(PNG_BYTES, None)], coordinator.calls)

    def test_back_only_upload_is_forwarded_to_the_back_side(self) -> None:
        session_id = self.create_session()
        self.assertEqual(200, self.upload(session_id, "back").status_code)
        self.client.post(
            f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS
        )
        self.wait_for_terminal(session_id)
        self.assertEqual([(None, PNG_BYTES)], self.coordinator.calls)

    def test_result_is_unavailable_while_running_and_duplicate_start_is_rejected(self) -> None:
        started = Event()
        release = Event()
        coordinator = FakeCoordinator(started=started, release=release)
        with TestClient(create_app(settings=settings(), coordinator=coordinator)) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            client.post(
                f"/v1/sessions/{session_id}/images/front",
                headers=AUTH_HEADERS,
                files={"image": ("front.png", PNG_BYTES, "image/png")},
            )
            client.post(f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS)
            self.assertTrue(started.wait(timeout=1))
            pending = client.get(
                f"/v1/sessions/{session_id}/result", headers=AUTH_HEADERS
            )
            duplicate = client.post(
                f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS
            )
            upload = client.post(
                f"/v1/sessions/{session_id}/images/back",
                headers=AUTH_HEADERS,
                files={"image": ("back.png", PNG_BYTES, "image/png")},
            )
            self.assertEqual(409, pending.status_code)
            self.assertEqual(409, duplicate.status_code)
            self.assertEqual(409, upload.status_code)
            release.set()
            self._wait_with_client(client, session_id)

    def test_rejects_bad_side_type_signature_and_size(self) -> None:
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

        with TestClient(
            create_app(
                settings=settings(max_upload_bytes=12),
                coordinator=FakeCoordinator(),
            )
        ) as small_client:
            small_id = small_client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            oversized = small_client.post(
                f"/v1/sessions/{small_id}/images/front",
                headers=AUTH_HEADERS,
                files={"image": ("front.png", PNG_BYTES, "image/png")},
            )
            self.assertEqual(413, oversized.status_code)

    def test_requires_an_image_before_processing(self) -> None:
        session_id = self.create_session()
        response = self.client.post(
            f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS
        )
        self.assertEqual(409, response.status_code)

    def test_unknown_session_and_idempotent_delete(self) -> None:
        missing = self.client.get("/v1/sessions/unknown", headers=AUTH_HEADERS)
        deleted = self.client.delete("/v1/sessions/unknown", headers=AUTH_HEADERS)
        self.assertEqual(404, missing.status_code)
        self.assertEqual(200, deleted.status_code)
        self.assertTrue(deleted.json()["deleted"])

    def test_expired_session_is_removed_with_its_image(self) -> None:
        store = SessionStore(ttl_seconds=1, max_sessions=2)
        with TestClient(
            create_app(settings=settings(), coordinator=FakeCoordinator(), session_store=store)
        ) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            client.post(
                f"/v1/sessions/{session_id}/images/front",
                headers=AUTH_HEADERS,
                files={"image": ("front.png", PNG_BYTES, "image/png")},
            )
            time.sleep(1.05)
            response = client.get(
                f"/v1/sessions/{session_id}", headers=AUTH_HEADERS
            )
            self.assertEqual(404, response.status_code)
            self.assertFalse(store.contains_images(session_id))

    def test_failed_job_is_safe_and_clears_uploaded_bytes(self) -> None:
        coordinator = FakeCoordinator(fail=True)
        store = SessionStore(ttl_seconds=60, max_sessions=4)
        with TestClient(
            create_app(settings=settings(), coordinator=coordinator, session_store=store)
        ) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            client.post(
                f"/v1/sessions/{session_id}/images/back",
                headers=AUTH_HEADERS,
                files={"image": ("secret-name.png", PNG_BYTES, "image/png")},
            )
            client.post(f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS)
            terminal = self._wait_with_client(client, session_id)
            self.assertEqual("failed", terminal["status"])
            self.assertFalse(store.contains_images(session_id))
            result = client.get(
                f"/v1/sessions/{session_id}/result", headers=AUTH_HEADERS
            )
            self.assertEqual(500, result.status_code)
            self.assertNotIn("secret-name", result.text)
            self.assertNotIn("sensitive backend failure", result.text)

    def test_completed_job_clears_uploaded_bytes_and_serializes_result(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=4)
        with TestClient(
            create_app(settings=settings(), coordinator=FakeCoordinator(), session_store=store)
        ) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            client.post(
                f"/v1/sessions/{session_id}/images/front",
                headers=AUTH_HEADERS,
                files={"image": ("front.png", PNG_BYTES, "image/png")},
            )
            client.post(f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS)
            terminal = self._wait_with_client(client, session_id)
            self.assertTrue(terminal["result_available"])
            self.assertFalse(store.contains_images(session_id))
            result = client.get(
                f"/v1/sessions/{session_id}/result", headers=AUTH_HEADERS
            )
            self.assertEqual(200, result.status_code)
            self.assertEqual("1.0", result.json()["schema_version"])
            self.assertEqual("front", result.json()["front"]["side"])
            self.assertEqual("back", result.json()["back"]["side"])

    def test_session_capacity_is_bounded(self) -> None:
        with TestClient(
            create_app(
                settings=settings(max_sessions=1),
                coordinator=FakeCoordinator(),
            )
        ) as client:
            self.assertEqual(201, client.post("/v1/sessions", headers=AUTH_HEADERS).status_code)
            response = client.post("/v1/sessions", headers=AUTH_HEADERS)
            self.assertEqual(429, response.status_code)

    def _wait_with_client(self, client: TestClient, session_id: str) -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            payload = client.get(
                f"/v1/sessions/{session_id}", headers=AUTH_HEADERS
            ).json()
            if payload["status"] in {"success", "partial", "failed"}:
                return payload
            time.sleep(0.01)
        self.fail("session did not reach a terminal state")


class JobManagerTests(unittest.TestCase):
    def test_submission_capacity_is_bounded(self) -> None:
        started = Event()
        release = Event()
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        first = store.create().session_id
        second = store.create().session_id
        store.upload(first, side=self._front(), content=PNG_BYTES)
        store.upload(second, side=self._front(), content=PNG_BYTES)
        coordinator = FakeCoordinator(started=started, release=release)
        with ThreadPoolExecutor(max_workers=1) as executor:
            manager = JobManager(
                coordinator,
                store,
                workers=1,
                capacity=1,
                executor=executor,
            )
            manager.submit(first)
            self.assertTrue(started.wait(timeout=1))
            with self.assertRaises(JobCapacityExceeded):
                manager.submit(second)
            release.set()

    @staticmethod
    def _front():
        from kyc_api.models import DocumentSide

        return DocumentSide.FRONT


if __name__ == "__main__":
    unittest.main()
