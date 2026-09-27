from __future__ import annotations

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event

from fastapi.testclient import TestClient

from kyc_engine import LivenessNoFaceError, LivenessResult
from kyc_engine.intake import ImageIntake
from kyc_api.jobs import JobManager
from kyc_api.main import create_app
from kyc_api.sessions import SessionConflict, SessionExpired, SessionStore, _now
from kyc_api.verification import VerificationManager

from helpers import (
    AUTH_HEADERS,
    PNG_BYTES,
    FakeCoordinator,
    FakeLivenessEvaluator,
    accept_document,
    settings,
)


class _ImmediateTimer:
    """Timer double that deterministically makes timeout win before processing."""

    def __init__(self, _delay: float, callback) -> None:
        self.daemon = False
        self._callback = callback

    def start(self) -> None:
        self._callback()

    def cancel(self) -> None:
        pass


class ApiLivenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = FakeLivenessEvaluator()
        self.context = TestClient(
            create_app(
                settings=settings(max_liveness_frame_bytes=1024),
                coordinator=FakeCoordinator(),
                liveness_evaluator=self.evaluator,
                liveness_intake=ImageIntake(),
            )
        )
        self.client = self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)

    def _ready_session(self) -> str:
        session_id = self.client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
        for side in ("front", "back"):
            response = self.client.post(
                f"/v1/sessions/{session_id}/images/{side}",
                headers=AUTH_HEADERS,
                files={"image": (f"{side}.png", PNG_BYTES, "image/png")},
            )
            self.assertEqual(200, response.status_code)
        return session_id

    def _submit(self, session_id: str, frames: list[tuple[str, bytes, str]] | None = None):
        values = frames or [(f"frame-{index}.png", PNG_BYTES, "image/png") for index in range(3)]
        return self.client.post(
            f"/v1/sessions/{session_id}/liveness",
            headers=AUTH_HEADERS,
            files=[("frames", value) for value in values],
        )

    def test_successful_submission_returns_updated_session_only(self) -> None:
        session_id = self._ready_session()
        response = self._submit(session_id)

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(session_id, payload["session"]["session_id"])
        self.assertEqual("passed", payload["session"]["liveness"]["status"])
        self.assertNotIn("passive_score", response.text)
        self.assertNotIn("frames_evaluated", response.text)
        self.assertEqual(1, self.evaluator.calls)
        self.assertEqual(3, len(self.evaluator.frames or ()))

    def test_document_timeout_makes_the_current_session_terminal(self) -> None:
        self.client.app.state.jobs._timer_factory = _ImmediateTimer
        session_id = self._ready_session()

        deadline = time.monotonic() + 3
        current = None
        while time.monotonic() < deadline:
            current = self.client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS).json()
            if current["document"]["status"] == "failed":
                break
            time.sleep(0.01)
        else:
            self.fail("document processing did not time out")

        assert current is not None
        self.assertEqual("failed", current["status"])
        self.assertIsNone(current["next_action"])

        response = self._submit(session_id)
        self.assertEqual(409, response.status_code)
        self.assertEqual(0, self.evaluator.calls)

    def test_failed_decision_is_a_safe_successful_liveness_interaction(self) -> None:
        self.context.__exit__(None, None, None)
        self.evaluator = FakeLivenessEvaluator(LivenessResult(False, 0.2, 3, 0))
        self.context = TestClient(
            create_app(settings=settings(), coordinator=FakeCoordinator(), liveness_evaluator=self.evaluator)
        )
        self.client = self.context.__enter__()
        response = self._submit(self._ready_session())
        self.assertEqual(200, response.status_code)
        self.assertEqual("failed", response.json()["session"]["liveness"]["status"])
        self.assertEqual("failed", response.json()["session"]["status"])

    def test_invalid_count_format_and_state_are_rejected(self) -> None:
        session_id = self._ready_session()
        count = self._submit(session_id, [("one.png", PNG_BYTES, "image/png")])
        self.assertEqual(422, count.status_code)
        self.assertEqual("LIVENESS_FRAME_COUNT", count.json()["error"]["code"])

        malformed = self._submit(
            session_id,
            [(f"frame-{index}.png", b"not-a-png", "image/png") for index in range(3)],
        )
        self.assertEqual(415, malformed.status_code)

        unsupported = self._submit(
            session_id,
            [(f"frame-{index}.gif", PNG_BYTES, "image/gif") for index in range(3)],
        )
        self.assertEqual(415, unsupported.status_code)

        blocked = self.client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
        self.assertEqual(409, self._submit(blocked).status_code)

    def test_unknown_and_terminal_liveness_state_are_safe(self) -> None:
        unknown = self._submit("missing-session")
        self.assertEqual(404, unknown.status_code)
        session_id = self._ready_session()
        self.assertEqual(200, self._submit(session_id).status_code)
        duplicate = self._submit(session_id)
        self.assertEqual(409, duplicate.status_code)

    def test_unconfigured_liveness_returns_a_safe_service_error(self) -> None:
        self.context.__exit__(None, None, None)
        self.context = TestClient(create_app(settings=settings(), coordinator=FakeCoordinator()))
        self.client = self.context.__enter__()
        session_id = self.client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
        response = self._submit(session_id)
        self.assertEqual(503, response.status_code)
        self.assertEqual("LIVENESS_UNAVAILABLE", response.json()["error"]["code"])

    def test_no_face_is_a_safe_capture_failure(self) -> None:
        self.context.__exit__(None, None, None)
        evaluator = FakeLivenessEvaluator(error=LivenessNoFaceError("face crop unavailable"))
        self.context = TestClient(
            create_app(settings=settings(), coordinator=FakeCoordinator(), liveness_evaluator=evaluator)
        )
        self.client = self.context.__enter__()
        session_id = self._ready_session()
        response = self._submit(session_id)
        self.assertEqual(422, response.status_code)
        self.assertEqual("NO_FACE_DETECTED", response.json()["error"]["code"])
        current = self.client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS).json()
        self.assertEqual("failed", current["liveness"]["status"])
        self.assertEqual("failed", current["status"])

    def test_oversized_upload_and_evaluator_error_do_not_expose_details(self) -> None:
        self.context.__exit__(None, None, None)
        evaluator = FakeLivenessEvaluator(error=RuntimeError("private model path: C:/secret"))
        self.context = TestClient(
            create_app(
                settings=settings(max_liveness_frame_bytes=16),
                coordinator=FakeCoordinator(),
                liveness_evaluator=evaluator,
            )
        )
        self.client = self.context.__enter__()
        session_id = self._ready_session()
        oversized = self._submit(
            session_id,
            [(f"frame-{index}.png", b"\x89PNG\r\n\x1a\n" + b"x" * 20, "image/png") for index in range(3)],
        )
        self.assertEqual(413, oversized.status_code)

        self.context.__exit__(None, None, None)
        self.context = TestClient(
            create_app(settings=settings(), coordinator=FakeCoordinator(), liveness_evaluator=evaluator)
        )
        self.client = self.context.__enter__()
        response = self._submit(self._ready_session())
        self.assertEqual(503, response.status_code)
        self.assertEqual("LIVENESS_EVALUATION_FAILED", response.json()["error"]["code"])
        self.assertNotIn("secret", response.text)


class VerificationManagerLivenessTests(unittest.TestCase):
    def _manager(self, evaluator: FakeLivenessEvaluator, events: list[str]):
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        executor = ThreadPoolExecutor(max_workers=1)
        manager = VerificationManager(
            assessor=None,  # Document capture is outside these liveness-only tests.
            store=store,
            jobs=JobManager(FakeCoordinator(), store, workers=1, capacity=1, executor=executor),
            liveness_evaluator=evaluator,
            liveness_intake=ImageIntake(),
            snapshot_publisher=lambda snapshot, event: events.append(event),
        )
        return store, manager, executor

    def test_manager_persists_safe_result_and_events(self) -> None:
        events: list[str] = []
        store, manager, executor = self._manager(FakeLivenessEvaluator(), events)
        try:
            session_id = store.create().session_id
            accept_document(store, session_id)
            submission = manager.submit_liveness(session_id, [PNG_BYTES] * 3)
            self.assertEqual("passed", submission.snapshot.liveness_status.value)
            self.assertEqual(["liveness.started", "liveness.passed"], events)
            self.assertEqual(0.9, store._records[session_id].liveness.result.passive_score)
            self.assertFalse(hasattr(submission.snapshot, "liveness_result"))
            with self.assertRaises(SessionConflict):
                manager.submit_liveness(session_id, [PNG_BYTES] * 3)
        finally:
            executor.shutdown()

    def test_manager_records_a_stable_passive_failure_code(self) -> None:
        events: list[str] = []
        store, manager, executor = self._manager(
            FakeLivenessEvaluator(LivenessResult(False, 0.1, 3, 0)), events
        )
        try:
            session_id = store.create().session_id
            accept_document(store, session_id)
            submission = manager.submit_liveness(session_id, [PNG_BYTES] * 3)
            self.assertFalse(submission.result.passed)
            self.assertEqual("failed", submission.snapshot.liveness_status.value)
            self.assertEqual("PASSIVE_LIVENESS_FAILED", store._records[session_id].liveness.error_code)
            self.assertEqual(
                ["liveness.started", "liveness.failed", "verification.failed"], events
            )
        finally:
            executor.shutdown()

    def test_manager_rejects_expired_and_concurrent_duplicate_submissions(self) -> None:
        class BlockingEvaluator:
            frame_count = 3

            def __init__(self) -> None:
                self.started = Event()
                self.release = Event()

            def evaluate(self, frames) -> LivenessResult:
                self.started.set()
                self.release.wait(timeout=3)
                return LivenessResult(True, 0.9, 3, 3)

        events: list[str] = []
        evaluator = BlockingEvaluator()
        store, manager, executor = self._manager(evaluator, events)  # type: ignore[arg-type]
        try:
            expired = store.create().session_id
            accept_document(store, expired)
            store._records[expired].expires_at = _now() - timedelta(seconds=1)
            with self.assertRaises(SessionExpired):
                manager.submit_liveness(expired, [PNG_BYTES] * 3)

            session_id = store.create().session_id
            accept_document(store, session_id)
            with ThreadPoolExecutor(max_workers=1) as calls:
                first = calls.submit(manager.submit_liveness, session_id, [PNG_BYTES] * 3)
                self.assertTrue(evaluator.started.wait(timeout=1))
                with self.assertRaises(SessionConflict):
                    manager.submit_liveness(session_id, [PNG_BYTES] * 3)
                evaluator.release.set()
                first.result(timeout=1)
            self.assertEqual(["liveness.started", "liveness.passed"], events)
        finally:
            executor.shutdown()

    def test_manager_fails_capture_and_operational_errors_without_resurrection(self) -> None:
        events: list[str] = []
        store, manager, executor = self._manager(
            FakeLivenessEvaluator(error=LivenessNoFaceError("no face")), events
        )
        try:
            session_id = store.create().session_id
            accept_document(store, session_id)
            with self.assertRaisesRegex(Exception, "usable face"):
                manager.submit_liveness(session_id, [PNG_BYTES] * 3)
            self.assertEqual("failed", store.get(session_id).liveness_status.value)
            self.assertEqual(
                ["liveness.started", "liveness.failed", "verification.failed"], events
            )
        finally:
            executor.shutdown()


if __name__ == "__main__":
    unittest.main()
