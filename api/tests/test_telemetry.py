from __future__ import annotations

import io
import json
import logging
import time
import unittest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from kyc_api.jobs import JobManager
from kyc_api.logging import JsonFormatter
from kyc_api.models import DocumentSide
from kyc_api.sessions import SessionStore
from kyc_api.telemetry import configure_tracing
from kyc_engine.contracts import KycExtractionResult, ProcessingStatus
from kyc_engine.coordinator import DocumentCoordinator
from kyc_engine.instrumentation import observe_pipeline_stage

from helpers import PNG_BYTES


class _ObservedPipeline:
    def __init__(self, side: str) -> None:
        self._side = side

    def process(self, _source: bytes) -> KycExtractionResult:
        with observe_pipeline_stage("intake"):
            pass
        with observe_pipeline_stage("ocr"):
            pass
        return KycExtractionResult(
            schema_version="1.1",
            processing_id=f"safe-{self._side}-processing-id",
            status=ProcessingStatus.SUCCESS,
            document_type="ao_id_card",
            side=self._side,
            profile_id=f"ao_id_card/{self._side}/v1",
            detection=None,
            fields={},
            issues=(),
            timings_ms={},
        )


class _FailingPipeline:
    def process(self, _source: bytes) -> KycExtractionResult:
        raise RuntimeError("synthetic-private-id-123456789")


class TelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_tracing(environment="test", service_version="test")
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            raise AssertionError("OpenTelemetry SDK provider was not installed")
        cls.exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(cls.exporter))

    def setUp(self) -> None:
        self.exporter.clear()

    def test_job_side_and_stage_spans_form_one_hierarchy(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id
        store.upload(session_id, DocumentSide.FRONT, PNG_BYTES)
        store.upload(session_id, DocumentSide.BACK, PNG_BYTES)
        manager = JobManager(
            DocumentCoordinator(_ObservedPipeline("front"), _ObservedPipeline("back")),
            store,
            workers=1,
            capacity=2,
        )
        try:
            job_id = manager.submit(session_id).job_id
            self._wait_for_success(store, session_id)
        finally:
            manager.shutdown()

        spans = self.exporter.get_finished_spans()
        roots = [span for span in spans if span.name == "kyc.process_job"]
        sides = [span for span in spans if span.name == "kyc.process_side"]
        stages = [span for span in spans if span.name == "kyc.stage"]

        self.assertEqual(1, len(roots))
        root = roots[0]
        self.assertEqual(session_id, root.attributes["kyc.session_id"])
        self.assertEqual(job_id, root.attributes["kyc.job_id"])
        self.assertEqual("success", root.attributes["kyc.processing_status"])
        self.assertEqual({"front", "back"}, {span.attributes["kyc.side"] for span in sides})
        self.assertTrue(
            all(span.parent.span_id == root.context.span_id for span in sides)
        )
        self.assertTrue(
            all(span.context.trace_id == root.context.trace_id for span in sides)
        )
        self.assertEqual(2, len({span.context.span_id for span in sides}))
        self.assertEqual({"intake", "ocr"}, {span.attributes["kyc.stage"] for span in stages})
        self.assertTrue(
            all(span.attributes["kyc.side"] in {"front", "back"} for span in stages)
        )
        side_ids = {span.context.span_id for span in sides}
        self.assertTrue(all(span.parent.span_id in side_ids for span in stages))
        self.assertTrue(all(span.status.status_code != StatusCode.ERROR for span in stages))

    def test_stage_failure_is_pii_safe_and_marks_only_the_stage_error(self) -> None:
        secret = "synthetic-private-id-123456789"
        with self.assertRaisesRegex(RuntimeError, secret):
            with observe_pipeline_stage("ocr", error_code="OCR_ENGINE_FAILED"):
                raise RuntimeError(secret)

        spans = self.exporter.get_finished_spans()
        self.assertEqual(1, len(spans))
        span = spans[0]
        self.assertEqual("kyc.stage", span.name)
        self.assertEqual(StatusCode.ERROR, span.status.status_code)
        self.assertEqual("RuntimeError", span.attributes["exception.type"])
        self.assertEqual((), span.events)
        self.assertNotIn(secret, repr((span.attributes, span.events, span.status)))

    def test_side_failure_remains_generic_and_has_safe_error_span(self) -> None:
        secret = "synthetic-private-id-123456789"
        result = DocumentCoordinator(_FailingPipeline(), _ObservedPipeline("back")).process(
            front=b"source"
        )

        self.assertEqual("FRONT_PROCESSING_FAILED", result.issues[0].code)
        side = next(span for span in self.exporter.get_finished_spans() if span.name == "kyc.process_side")
        self.assertEqual(StatusCode.ERROR, side.status.status_code)
        self.assertEqual("RuntimeError", side.attributes["exception.type"])
        self.assertEqual("failed", side.attributes["kyc.processing_status"])
        self.assertEqual((), side.events)
        self.assertNotIn(secret, repr((side.attributes, side.events, side.status)))

    def test_stage_log_contains_its_active_trace_identifiers(self) -> None:
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
            with observe_pipeline_stage("detection"):
                pass
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        completed = next(
            json.loads(line)
            for line in stream.getvalue().splitlines()
            if json.loads(line)["event"] == "pipeline_stage_completed"
        )
        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(trace.format_trace_id(span.context.trace_id), completed["trace_id"])
        self.assertEqual(trace.format_span_id(span.context.span_id), completed["span_id"])

    def test_sequential_jobs_create_distinct_traces(self) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=3)
        first_session = store.create().session_id
        second_session = store.create().session_id
        store.upload(first_session, DocumentSide.FRONT, PNG_BYTES)
        store.upload(first_session, DocumentSide.BACK, PNG_BYTES)
        store.upload(second_session, DocumentSide.FRONT, PNG_BYTES)
        store.upload(second_session, DocumentSide.BACK, PNG_BYTES)
        manager = JobManager(
            DocumentCoordinator(_ObservedPipeline("front"), _ObservedPipeline("back")),
            store,
            workers=1,
            capacity=3,
        )
        try:
            manager.submit(first_session)
            manager.submit(second_session)
            self._wait_for_success(store, first_session)
            self._wait_for_success(store, second_session)
        finally:
            manager.shutdown()

        roots = [span for span in self.exporter.get_finished_spans() if span.name == "kyc.process_job"]
        self.assertEqual(2, len(roots))
        self.assertEqual(2, len({span.context.trace_id for span in roots}))

    def _wait_for_success(self, store: SessionStore, session_id: str) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if store.get(session_id).status.value == "success":
                return
            time.sleep(0.01)
        self.fail("job did not complete")


if __name__ == "__main__":
    unittest.main()
