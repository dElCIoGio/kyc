from __future__ import annotations

import time
import unittest
from threading import Timer

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from kyc_api.jobs import JobManager
from kyc_api.metrics import MetricsRegistry
from kyc_api.models import DocumentSide
from kyc_api.sessions import SessionStore
from kyc_engine import DocumentExtractionResult, KycExtractionResult, ProcessingStatus
from kyc_engine.contracts import ExtractedField, FieldStatus
from kyc_engine.instrumentation import (
    observe_pipeline_side,
    observe_pipeline_stage,
    pipeline_metrics_context,
)

from helpers import FakeCoordinator, PNG_BYTES


class _ImmediateTimer:
    def __init__(self, _delay: float, callback) -> None:
        self.daemon = False
        self._callback = callback

    def start(self) -> None:
        self._callback()

    def cancel(self) -> None:
        pass


class OperationalMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = InMemoryMetricReader()
        self.provider = MeterProvider(metric_readers=[self.reader])
        self.registry = MetricsRegistry(meter_provider=self.provider)

    def tearDown(self) -> None:
        self.provider.shutdown()

    def test_terminal_job_outcomes_are_counted_once_and_have_duration(self) -> None:
        self._run_job(FakeCoordinator(status=ProcessingStatus.SUCCESS))
        self._run_job(FakeCoordinator(status=ProcessingStatus.PARTIAL))
        self._run_job(FakeCoordinator(fail=True))

        outcomes = self._counter_values("kyc.jobs")
        self.assertEqual(1, outcomes[("status", "success")])
        self.assertEqual(1, outcomes[("status", "partial")])
        self.assertEqual(1, outcomes[("status", "failed")])
        durations = self._histogram_points("kyc.job.duration")
        self.assertEqual(3, sum(point.count for point in durations))
        self.assertTrue(all(point.sum >= 0 for point in durations))

    def test_timeout_winning_the_transition_records_only_timeout(self) -> None:
        self._run_job(FakeCoordinator(), timer_factory=_ImmediateTimer)

        outcomes = self._counter_values("kyc.jobs")
        self.assertEqual({("status", "timeout"): 1}, outcomes)
        self.assertEqual(1, sum(point.count for point in self._histogram_points("kyc.job.duration")))

    def test_stage_metrics_track_success_failure_duration_and_side(self) -> None:
        with pipeline_metrics_context(self.registry):
            with observe_pipeline_side("front"):
                with observe_pipeline_stage("ocr"):
                    pass
                with self.assertRaises(RuntimeError):
                    with observe_pipeline_stage("detection"):
                        raise RuntimeError("synthetic-private-id-123456789")

        executions = self._counter_values("kyc.stage.executions")
        self.assertEqual(1, executions[("side", "front", "stage", "ocr", "status", "success")])
        self.assertEqual(1, executions[("side", "front", "stage", "detection", "status", "failed")])
        failures = self._counter_values("kyc.stage.failures")
        self.assertEqual(1, failures[("side", "front", "stage", "detection")])
        durations = self._histogram_points("kyc.stage.duration")
        self.assertEqual(2, sum(point.count for point in durations))

    def test_ocr_fallback_semantics_record_one_successful_stage(self) -> None:
        with pipeline_metrics_context(self.registry):
            with observe_pipeline_stage("ocr"):
                try:
                    raise RuntimeError("synthetic-private-id-123456789")
                except RuntimeError:
                    pass

        executions = self._counter_values("kyc.stage.executions")
        self.assertEqual({("side", "unknown", "stage", "ocr", "status", "success"): 1}, executions)
        self.assertEqual({}, self._counter_values("kyc.stage.failures"))

    def test_final_field_statuses_are_allowlisted_and_value_free(self) -> None:
        secret = "synthetic-private-id-123456789"
        front = KycExtractionResult(
            schema_version="1.1",
            processing_id="private-processing-id",
            status=ProcessingStatus.SUCCESS,
            document_type="ao_id_card",
            side="front",
            profile_id="ao_id_card/front/v1",
            detection=None,
            fields={
                "id_number": ExtractedField(
                    name="id_number",
                    status=FieldStatus.INVALID,
                    raw_value=secret,
                    normalized_value=secret,
                    confidence=0.1,
                    selected_candidate=None,
                ),
                "untrusted_field": ExtractedField(
                    name="untrusted_field",
                    status=FieldStatus.ERROR,
                    raw_value=secret,
                    normalized_value=secret,
                    confidence=0.0,
                    selected_candidate=None,
                ),
            },
            issues=(),
            timings_ms={},
        )
        coordinator = FakeCoordinator()
        coordinator.output = DocumentExtractionResult(
            schema_version="1.0",
            status=ProcessingStatus.SUCCESS,
            front=front,
            back=None,
            issues=(),
            timings_ms={},
        )
        self._run_job(coordinator)

        fields = self._counter_values("kyc.field.status")
        self.assertEqual(
            {("field", "id_number", "side", "front", "status", "invalid"): 1},
            fields,
        )
        self.assertNotIn(secret, repr(self.reader.get_metrics_data()))
        self.assertNotIn("private-processing-id", repr(self.reader.get_metrics_data()))
        self.assertNotIn("untrusted_field", repr(self.reader.get_metrics_data()))

    def test_sequential_jobs_aggregate_without_identifier_attributes(self) -> None:
        self._run_job(FakeCoordinator())
        self._run_job(FakeCoordinator())

        self.assertEqual({("status", "success"): 2}, self._counter_values("kyc.jobs"))
        for resource_metrics in self.reader.get_metrics_data().resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    for point in metric.data.data_points:
                        self.assertTrue(
                            {"request_id", "session_id", "job_id", "trace_id", "span_id"}
                            .isdisjoint(point.attributes)
                        )

    def _run_job(self, coordinator, *, timer_factory=None) -> None:
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id
        store.upload(session_id, DocumentSide.FRONT, PNG_BYTES)
        manager = JobManager(
            coordinator,
            store,
            workers=1,
            capacity=2,
            timeout_seconds=1,
            metrics=self.registry,
            timer_factory=timer_factory or Timer,
        )
        try:
            manager.submit(session_id)
            self._wait_for_terminal(store, session_id)
        finally:
            manager.shutdown()

    def _wait_for_terminal(self, store: SessionStore, session_id: str) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if store.get(session_id).status.value in {"success", "partial", "failed"}:
                return
            time.sleep(0.01)
        self.fail("job did not reach a terminal state")

    def _counter_values(self, name: str) -> dict[tuple[str, ...], int]:
        values: dict[tuple[str, ...], int] = {}
        for point in self._metric_points(name):
            key = tuple(item for pair in sorted(point.attributes.items()) for item in pair)
            values[key] = point.value
        return values

    def _histogram_points(self, name: str):
        return self._metric_points(name)

    def _metric_points(self, name: str):
        return [
            point
            for resource_metrics in self.reader.get_metrics_data().resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
            if metric.name == name
            for point in metric.data.data_points
        ]


if __name__ == "__main__":
    unittest.main()
