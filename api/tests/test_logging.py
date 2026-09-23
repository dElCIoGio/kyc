from __future__ import annotations

import io
import json
import logging
import time
import unittest

from fastapi.testclient import TestClient

from kyc_api.logging import JsonFormatter
from kyc_api.main import create_app
from kyc_api.jobs import JobManager
from kyc_api.models import DocumentSide
from kyc_api.sessions import SessionStore
from kyc_engine.instrumentation import observe_pipeline_stage

from helpers import AUTH_HEADERS, PNG_BYTES, extraction_result, settings


class _SensitiveFailingCoordinator:
    def process(self, **_kwargs):
        raise RuntimeError("synthetic-private-id-123456789")


class _FailingPipeline:
    def process(self, _source):
        raise RuntimeError("synthetic-private-id-123456789")


class _DeepLoggingCoordinator:
    def process(self, **_kwargs):
        with observe_pipeline_stage("test_stage"):
            pass
        return extraction_result()


class StructuredLoggingTests(unittest.TestCase):
    def test_responses_get_distinct_server_generated_request_ids(self) -> None:
        with TestClient(create_app(settings=settings(), coordinator=_DeepLoggingCoordinator())) as client:
            first = client.get("/healthz", headers={"X-Request-ID": "client-value"})
            second = client.post("/v1/sessions")

        self.assertIn("X-Request-ID", first.headers)
        self.assertIn("X-Request-ID", second.headers)
        self.assertNotEqual("client-value", first.headers["X-Request-ID"])
        self.assertNotEqual(first.headers["X-Request-ID"], second.headers["X-Request-ID"])

    def test_handled_error_and_log_share_request_context(self) -> None:
        class ExplodingStore(SessionStore):
            def create(self):
                raise RuntimeError("synthetic-private-id-123456789")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(environment="test"))
        logger = logging.getLogger("kyc_api.main")
        original_handlers, original_level, original_propagate = (
            logger.handlers[:], logger.level, logger.propagate
        )
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            with TestClient(
                create_app(
                    settings=settings(environment="test"),
                    coordinator=_DeepLoggingCoordinator(),
                    session_store=ExplodingStore(ttl_seconds=60, max_sessions=2),
                ),
                raise_server_exceptions=False,
            ) as client:
                first_response = client.post("/v1/sessions", headers=AUTH_HEADERS)
                second_response = client.post("/v1/sessions", headers=AUTH_HEADERS)
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        self.assertEqual(500, first_response.status_code)
        self.assertEqual(500, second_response.status_code)
        self.assertIn("X-Request-ID", first_response.headers)
        self.assertIn("X-Request-ID", second_response.headers)
        self.assertNotEqual(
            first_response.headers["X-Request-ID"],
            second_response.headers["X-Request-ID"],
        )
        records = [
            json.loads(line)
            for line in stream.getvalue().splitlines()
            if json.loads(line)["event"] == "http_request_failed"
        ]
        self.assertEqual(
            {first_response.headers["X-Request-ID"], second_response.headers["X-Request-ID"]},
            {record["request_id"] for record in records},
        )
        self.assertNotIn("synthetic-private-id-123456789", stream.getvalue())

    def test_queue_event_has_request_session_and_job_context(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(environment="test"))
        logger = logging.getLogger("kyc_api.main")
        original_handlers, original_level, original_propagate = (
            logger.handlers[:], logger.level, logger.propagate
        )
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            with TestClient(
                create_app(settings=settings(environment="test"), coordinator=_DeepLoggingCoordinator())
            ) as client:
                session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
                client.post(
                    f"/v1/sessions/{session_id}/images/front",
                    headers=AUTH_HEADERS,
                    files={"image": ("front.png", PNG_BYTES, "image/png")},
                )
                response = client.post(f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS)
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        record = next(
            json.loads(line)
            for line in stream.getvalue().splitlines()
            if json.loads(line)["event"] == "processing_job_queued"
        )
        self.assertEqual(response.headers["X-Request-ID"], record["request_id"])
        self.assertEqual(session_id, record["session_id"])
        self.assertEqual(response.json()["job_id"], record["job_id"])

    def test_worker_context_reaches_engine_and_does_not_leak_between_jobs(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(environment="test"))
        logger = logging.getLogger("kyc_engine")
        original_handlers, original_level, original_propagate = (
            logger.handlers[:], logger.level, logger.propagate
        )
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        first = store.create().session_id
        second = store.create().session_id
        store.upload(first, DocumentSide.FRONT, PNG_BYTES)
        store.upload(second, DocumentSide.FRONT, PNG_BYTES)
        try:
            manager = JobManager(
                _DeepLoggingCoordinator(),
                store,
                workers=1,
                capacity=2,
            )
            first_job = manager.submit(first).job_id
            second_job = manager.submit(second).job_id
            self._wait_for_success(store, first)
            self._wait_for_success(store, second)
            manager.shutdown()
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        records = [
            json.loads(line)
            for line in stream.getvalue().splitlines()
            if json.loads(line)["event"] == "pipeline_stage_completed"
        ]
        self.assertEqual(2, len(records))
        self.assertEqual({(first, first_job), (second, second_job)}, {
            (record["session_id"], record["job_id"]) for record in records
        })
        self.assertTrue(all("request_id" not in record for record in records))
    def test_formatter_emits_valid_json_and_ignores_unapproved_values(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(environment="test"))
        logger = logging.getLogger("logging-test.formatter")
        original_handlers, original_level, original_propagate = (
            logger.handlers[:],
            logger.level,
            logger.propagate,
        )
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            logger.info(
                "this message is intentionally not serialized",
                extra={
                    "event": "processing_job_completed",
                    "session_id": "session-test-1",
                    "status": "success",
                    "extracted_value": "synthetic-private-id-123456789",
                },
            )
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        payload = json.loads(stream.getvalue())
        self.assertEqual("processing_job_completed", payload["event"])
        self.assertEqual("kyc-api", payload["service"])
        self.assertEqual("test", payload["environment"])
        self.assertEqual("session-test-1", payload["session_id"])
        self.assertNotIn("extracted_value", payload)
        self.assertNotIn("synthetic-private-id-123456789", stream.getvalue())

    def test_stage_failure_json_omits_exception_message(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(environment="test"))
        logger = logging.getLogger("kyc_engine.pipeline")
        original_handlers, original_level, original_propagate = (
            logger.handlers[:], logger.level, logger.propagate
        )
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            with self.assertRaises(RuntimeError):
                with observe_pipeline_stage("ocr", error_code="OCR_ENGINE_FAILED"):
                    raise RuntimeError("synthetic-private-id-123456789")
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual("pipeline_stage_started", records[0]["event"])
        self.assertEqual("pipeline_stage_failed", records[1]["event"])
        self.assertEqual("OCR_ENGINE_FAILED", records[1]["error_code"])
        self.assertNotIn("synthetic-private-id-123456789", stream.getvalue())

    def test_failed_background_job_logs_traceback_but_keeps_public_error_generic(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(environment="test"))
        logger = logging.getLogger("kyc_api.jobs")
        original_handlers, original_level, original_propagate = (
            logger.handlers[:],
            logger.level,
            logger.propagate,
        )
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            with TestClient(
                create_app(
                    settings=settings(environment="test"),
                    coordinator=_SensitiveFailingCoordinator(),
                )
            ) as client:
                session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
                client.post(
                    f"/v1/sessions/{session_id}/images/front",
                    headers=AUTH_HEADERS,
                    files={"image": ("private-name.png", PNG_BYTES, "image/png")},
                )
                self.assertEqual(
                    202,
                    client.post(f"/v1/sessions/{session_id}/process", headers=AUTH_HEADERS).status_code,
                )
                self._wait_for_failed_job(client, session_id)
                response = client.get(f"/v1/sessions/{session_id}/result", headers=AUTH_HEADERS)
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        self.assertEqual(500, response.status_code)
        self.assertEqual("JOB_FAILED", response.json()["error"]["code"])
        self.assertNotIn("synthetic-private-id-123456789", response.text)
        self.assertNotIn("private-name", response.text)

        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        failure = next(record for record in records if record["event"] == "processing_job_failed")
        self.assertEqual("RuntimeError", failure["exception_type"])
        self.assertTrue(failure["traceback"])
        self.assertNotIn("synthetic-private-id-123456789", stream.getvalue())
        self.assertNotIn("private-name", stream.getvalue())

    def test_coordinator_side_failure_logs_safe_metadata_and_preserves_issue_code(self) -> None:
        from kyc_engine.coordinator import DocumentCoordinator

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(environment="test"))
        logger = logging.getLogger("kyc_engine.coordinator")
        original_handlers, original_level, original_propagate = (
            logger.handlers[:],
            logger.level,
            logger.propagate,
        )
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            output = DocumentCoordinator(_FailingPipeline(), _FailingPipeline()).process(front=b"image")
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        self.assertEqual("FRONT_PROCESSING_FAILED", output.issues[0].code)
        record = json.loads(stream.getvalue())
        self.assertEqual("document_side_processing_failed", record["event"])
        self.assertEqual("front", record["side"])
        self.assertEqual("RuntimeError", record["exception_type"])
        self.assertNotIn("synthetic-private-id-123456789", stream.getvalue())

    def _wait_for_failed_job(self, client: TestClient, session_id: str) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            response = client.get(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS)
            if response.json()["status"] == "failed":
                return
            time.sleep(0.01)
        self.fail("job did not fail")

    def _wait_for_success(self, store: SessionStore, session_id: str) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if store.get(session_id).status.value == "success":
                return
            time.sleep(0.01)
        self.fail("job did not complete")


if __name__ == "__main__":
    unittest.main()
