from __future__ import annotations

import asyncio
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient

from kyc_api.jobs import JobCapacityExceeded, JobManager
from kyc_api.main import create_app
from kyc_api.metrics import MetricsRegistry
from kyc_api.middleware import RequestBodyLimitMiddleware
from kyc_api.models import DocumentSide
from kyc_api.rate_limit import ApiKeyRateLimiter
from kyc_api.sessions import SessionStore, SessionStoreError, _now

from helpers import API_KEY, AUTH_HEADERS, PNG_BYTES, FakeCoordinator, accept_document, settings


class _ImmediateTimer:
    def __init__(self, _interval: float, callback) -> None:
        self.daemon = False
        self._callback = callback

    def start(self) -> None:
        self._callback()

    def cancel(self) -> None:
        pass


class RequestBodyLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_streamed_body_over_limit_before_application(self) -> None:
        called = False
        metrics = MetricsRegistry()

        async def application(_scope, _receive, _send) -> None:
            nonlocal called
            called = True

        messages = iter(
            (
                {"type": "http.request", "body": b"1234", "more_body": True},
                {"type": "http.request", "body": b"567", "more_body": False},
            )
        )
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            return next(messages)

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        middleware = RequestBodyLimitMiddleware(
            application,
            limit_provider=lambda: 6,
            metrics_provider=lambda: metrics,
        )
        await middleware(
            {"type": "http", "path": "/v1/sessions", "headers": []},
            receive,
            send,
        )

        self.assertFalse(called)
        self.assertEqual(413, sent[0]["status"])
        self.assertIn(b"REQUEST_TOO_LARGE", bytes(sent[1]["body"]))
        self.assertEqual(1, metrics.snapshot()["errors_by_code"]["REQUEST_TOO_LARGE"])


class ApiProtectionTests(unittest.TestCase):
    def test_request_content_length_is_rejected_before_authentication(self) -> None:
        configured = settings(max_upload_bytes=32)
        with TestClient(
            create_app(
                settings=configured,
                coordinator=FakeCoordinator(),
            )
        ) as client:
            response = client.post(
                "/v1/sessions",
                headers={"Content-Length": str(configured.max_request_bytes + 1)},
            )
        self.assertEqual(413, response.status_code)
        self.assertEqual("REQUEST_TOO_LARGE", response.json()["error"]["code"])

    def test_multipart_overhead_is_allowed_beyond_image_limit(self) -> None:
        configured = settings(max_upload_bytes=len(PNG_BYTES))
        with TestClient(
            create_app(
                settings=configured,
                coordinator=FakeCoordinator(),
            )
        ) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            response = client.post(
                f"/v1/sessions/{session_id}/images/front",
                headers=AUTH_HEADERS,
                files={"image": ("front.png", PNG_BYTES, "image/png")},
            )
        self.assertEqual(200, response.status_code)

    def test_rate_limit_has_safe_retry_and_health_is_exempt(self) -> None:
        configured = settings(rate_limit_requests=1, rate_limit_window_seconds=60)
        with TestClient(
            create_app(
                settings=configured,
                coordinator=FakeCoordinator(),
            )
        ) as client:
            self.assertEqual(200, client.get("/healthz").status_code)
            self.assertEqual(200, client.get("/healthz").status_code)
            self.assertEqual(201, client.post("/v1/sessions", headers=AUTH_HEADERS).status_code)
            limited = client.post("/v1/sessions", headers=AUTH_HEADERS)
        self.assertEqual(429, limited.status_code)
        self.assertEqual("RATE_LIMITED", limited.json()["error"]["code"])
        self.assertGreaterEqual(int(limited.headers["Retry-After"]), 1)
        self.assertNotIn(API_KEY, limited.text)

    def test_invalid_key_does_not_consume_valid_key_budget(self) -> None:
        configured = settings(rate_limit_requests=1, rate_limit_window_seconds=60)
        with TestClient(
            create_app(
                settings=configured,
                coordinator=FakeCoordinator(),
            )
        ) as client:
            self.assertEqual(401, client.post("/v1/sessions", headers={"X-API-Key": "wrong"}).status_code)
            self.assertEqual(201, client.post("/v1/sessions", headers=AUTH_HEADERS).status_code)

    def test_rate_limit_window_resets(self) -> None:
        with patch("kyc_api.rate_limit.monotonic", side_effect=(10.0, 10.0, 10.5, 71.0)):
            limiter = ApiKeyRateLimiter(max_requests=1, window_seconds=60)
            self.assertIsNone(limiter.allow())
            self.assertEqual(60, limiter.allow())
            self.assertIsNone(limiter.allow())

    def test_metrics_are_authenticated_aggregate_only_and_pii_safe(self) -> None:
        with TestClient(create_app(settings=settings(), coordinator=FakeCoordinator())) as client:
            self.assertEqual(401, client.get("/v1/metrics").status_code)
            client.get("/v1/sessions/missing", headers=AUTH_HEADERS)
            response = client.get("/v1/metrics", headers=AUTH_HEADERS)
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertIn("responses_by_status", payload)
        self.assertIn("request_latency_ms", payload)
        self.assertGreaterEqual(payload["errors_by_code"]["SESSION_NOT_FOUND"], 1)
        self.assertNotIn("missing", response.text)
        self.assertNotIn(API_KEY, response.text)
        self.assertNotIn("image", response.text.lower())


class JobTimeoutTests(unittest.TestCase):
    def test_timeout_clears_images_rejects_late_completion_and_keeps_capacity(self) -> None:
        started = Event()
        release = Event()
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        first = store.create().session_id
        second = store.create().session_id
        accept_document(store, first)
        accept_document(store, second)
        coordinator = FakeCoordinator(started=started, release=release)
        metrics = MetricsRegistry()
        with ThreadPoolExecutor(max_workers=1) as executor:
            manager = JobManager(
                coordinator,
                store,
                workers=1,
                capacity=1,
                timeout_seconds=1,
                metrics=metrics,
                executor=executor,
                timer_factory=_ImmediateTimer,
            )
            manager.submit_document_processing(first)
            self.assertTrue(started.wait(timeout=1))
            self.assertEqual("failed", store.get(first).document.status.value)
            self.assertFalse(store.contains_images(first))
            with self.assertRaises(JobCapacityExceeded):
                manager.submit_document_processing(second)
            release.set()
        with self.assertRaises(SessionStoreError):
            store.result(first)
        self.assertEqual(1, metrics.snapshot()["jobs_by_outcome"]["timeout"])

    def test_cleanup_removes_expired_terminal_state_without_request_access(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=1)
        session_id = store.create().session_id
        record = store._records[session_id]
        record.expires_at = _now() - timedelta(seconds=1)
        self.assertEqual(1, store.cleanup())
        self.assertNotIn(session_id, store._records)

    def test_lifespan_cleanup_purges_expired_session_without_access(self) -> None:
        store = SessionStore(ttl_seconds=1, max_sessions=1)
        configured = settings(session_ttl_seconds=1, session_cleanup_interval_seconds=1)
        with TestClient(create_app(settings=configured, coordinator=FakeCoordinator(), session_store=store)):
            session_id = store.create().session_id
            accept_document(store, session_id)
            time.sleep(1.2)
            self.assertNotIn(session_id, store._records)


if __name__ == "__main__":
    unittest.main()
