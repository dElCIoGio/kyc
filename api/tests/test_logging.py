from __future__ import annotations

import io
import json
import logging
import time
import unittest

from fastapi.testclient import TestClient

from kyc_api.logging import JsonFormatter
from kyc_api.main import create_app

from helpers import AUTH_HEADERS, PNG_BYTES, settings


class _SensitiveFailingCoordinator:
    def process(self, **_kwargs):
        raise RuntimeError("synthetic-private-id-123456789")


class _FailingPipeline:
    def process(self, _source):
        raise RuntimeError("synthetic-private-id-123456789")


class StructuredLoggingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
