from __future__ import annotations

import io
import json
import logging
import time
import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from pydantic import ValidationError

from helpers import FakeCoordinator, PNG_BYTES, settings
from kyc_api.jobs import JobManager
from kyc_api.logging import JsonFormatter
from kyc_api.models import DocumentSide
from kyc_api.sessions import SessionStore
from kyc_api.telemetry import (
    OtlpExportConfiguration,
    build_meter_provider,
    build_otlp_export_configuration,
    build_tracer_provider,
)


class _FailingSpanExporter(SpanExporter):
    def export(self, spans) -> SpanExportResult:
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        return None


class OtlpExportConfigurationTests(unittest.TestCase):
    def test_disabled_export_requires_no_endpoint_or_exporters(self) -> None:
        self.assertIsNone(build_otlp_export_configuration(settings()))
        with (
            patch("kyc_api.telemetry.OTLPSpanExporter") as span_exporter,
            patch("kyc_api.telemetry.OTLPMetricExporter") as metric_exporter,
        ):
            tracer_provider = build_tracer_provider(
                environment="test", service_version="test"
            )
            meter_provider = build_meter_provider(
                environment="test", service_version="test"
            )
            tracer_provider.shutdown()
            meter_provider.shutdown()
        span_exporter.assert_not_called()
        metric_exporter.assert_not_called()

    def test_enabled_configuration_derives_signal_endpoints_and_keeps_headers_private(self) -> None:
        secret = "synthetic-otel-secret-123456789"
        export = build_otlp_export_configuration(
            settings(
                otel_enabled=True,
                otel_endpoint="https://collector.example.test/base/",
                otel_headers=f"Authorization=Bearer {secret},X-Tenant=beta",
                otel_trace_sample_ratio=0.25,
                otel_metric_export_interval_seconds=45,
                otel_export_timeout_seconds=7,
            )
        )

        assert export is not None
        self.assertEqual("https://collector.example.test/base/v1/traces", export.traces_endpoint)
        self.assertEqual("https://collector.example.test/base/v1/metrics", export.metrics_endpoint)
        self.assertEqual({"Authorization": f"Bearer {secret}", "X-Tenant": "beta"}, export.headers)
        self.assertEqual(0.25, export.trace_sample_ratio)
        self.assertEqual(45_000, export.metric_export_interval_millis)
        self.assertEqual(7_000, export.export_timeout_millis)
        self.assertNotIn(secret, repr(export).replace(repr(export.headers), ""))

    def test_enabled_configuration_rejects_invalid_static_values(self) -> None:
        invalid = (
            {"otel_enabled": True},
            {"otel_enabled": True, "otel_endpoint": "collector.example.test"},
            {"otel_enabled": True, "otel_endpoint": "https://user:pass@collector.example.test"},
            {"otel_enabled": True, "otel_endpoint": "https://collector.example.test/v1/traces"},
            {"otel_trace_sample_ratio": -0.1},
            {"otel_trace_sample_ratio": 1.1},
            {"otel_metric_export_interval_seconds": 0},
            {"otel_export_timeout_seconds": 0},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                settings(**overrides)

    def test_header_syntax_is_validated_without_echoing_secret(self) -> None:
        with self.assertRaisesRegex(ValueError, "KYC_OTEL_HEADERS") as raised:
            build_otlp_export_configuration(
                settings(
                    otel_enabled=True,
                    otel_endpoint="https://collector.example.test",
                    otel_headers="Authorization synthetic-otel-secret-123456789",
                )
            )
        self.assertNotIn("synthetic-otel-secret-123456789", str(raised.exception))

    def test_telemetry_configuration_log_cannot_serialize_header_secret(self) -> None:
        secret = "synthetic-otel-secret-123456789"
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(environment="test"))
        logger = logging.getLogger("kyc_api.test_otlp")
        original_handlers, original_level, original_propagate = (
            logger.handlers[:], logger.level, logger.propagate
        )
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            logger.info(
                "OTLP telemetry export configured",
                extra={
                    "event": "telemetry_export_configured",
                    "protocol": "http/protobuf",
                    "headers": {"Authorization": f"Bearer {secret}"},
                },
            )
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        payload = json.loads(stream.getvalue())
        self.assertEqual("telemetry_export_configured", payload["event"])
        self.assertEqual("http/protobuf", payload["protocol"])
        self.assertNotIn("headers", payload)
        self.assertNotIn(secret, stream.getvalue())

    def test_enabled_factories_install_batched_and_periodic_export_paths(self) -> None:
        export = OtlpExportConfiguration(
            traces_endpoint="https://collector.example.test/v1/traces",
            metrics_endpoint="https://collector.example.test/v1/metrics",
            headers={"Authorization": "Bearer synthetic-otel-secret-123456789"},
            trace_sample_ratio=0.25,
            metric_export_interval_millis=45_000,
            export_timeout_millis=7_000,
        )
        fake_span_exporter = MagicMock()
        fake_processor = MagicMock()
        with (
            patch("kyc_api.telemetry.OTLPSpanExporter", return_value=fake_span_exporter) as span_exporter,
            patch("kyc_api.telemetry.BatchSpanProcessor", return_value=fake_processor) as processor,
            patch("kyc_api.telemetry.TraceIdRatioBased") as ratio_sampler,
            patch("kyc_api.telemetry.ParentBased") as parent_sampler,
        ):
            provider = build_tracer_provider(
                environment="beta", service_version="test", export=export
            )
        span_exporter.assert_called_once_with(
            endpoint=export.traces_endpoint,
            headers=export.headers,
            timeout=7.0,
        )
        processor.assert_called_once_with(fake_span_exporter, export_timeout_millis=7_000)
        ratio_sampler.assert_called_once_with(0.25)
        parent_sampler.assert_called_once_with(ratio_sampler.return_value)
        self.assertEqual("kyc-api", provider.resource.attributes["service.name"])
        self.assertEqual("beta", provider.resource.attributes["deployment.environment.name"])

        fake_metric_exporter = MagicMock()
        fake_reader = MagicMock()
        with (
            patch("kyc_api.telemetry.OTLPMetricExporter", return_value=fake_metric_exporter) as metric_exporter,
            patch("kyc_api.telemetry.PeriodicExportingMetricReader", return_value=fake_reader) as reader,
            patch("kyc_api.telemetry.MeterProvider") as meter_provider,
        ):
            build_meter_provider(environment="beta", service_version="test", export=export)
        metric_exporter.assert_called_once_with(
            endpoint=export.metrics_endpoint,
            headers=export.headers,
            timeout=7.0,
        )
        reader.assert_called_once_with(
            fake_metric_exporter,
            export_interval_millis=45_000,
            export_timeout_millis=7_000,
        )
        meter_provider.assert_called_once()
        self.assertEqual((fake_reader,), meter_provider.call_args.kwargs["metric_readers"])

    def test_async_span_export_failure_does_not_change_a_successful_job(self) -> None:
        provider = TracerProvider()
        provider.add_span_processor(
            BatchSpanProcessor(_FailingSpanExporter(), schedule_delay_millis=1)
        )
        store = SessionStore(ttl_seconds=60, max_sessions=2)
        session_id = store.create().session_id
        store.upload(session_id, DocumentSide.FRONT, PNG_BYTES)
        manager = JobManager(
            FakeCoordinator(), store, workers=1, capacity=2, timeout_seconds=1
        )
        try:
            with patch(
                "kyc_api.jobs.trace.get_tracer",
                return_value=provider.get_tracer("test.failing_export"),
            ):
                manager.submit(session_id)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if store.get(session_id).status.value == "success":
                        break
                    time.sleep(0.01)
                else:
                    self.fail("job did not succeed")
            self.assertTrue(provider.force_flush(timeout_millis=1_000))
            self.assertEqual("success", store.get(session_id).status.value)
        finally:
            manager.shutdown()
            provider.shutdown()


if __name__ == "__main__":
    unittest.main()
