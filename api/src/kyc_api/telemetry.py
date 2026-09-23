"""Application-owned, vendor-neutral OpenTelemetry SDK configuration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import ProxyTracerProvider

from .settings import ApiSettings


logger = logging.getLogger(__name__)
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


@dataclass(frozen=True)
class OtlpExportConfiguration:
    """Validated configuration passed privately to OTLP exporters."""

    traces_endpoint: str
    metrics_endpoint: str
    headers: Mapping[str, str]
    trace_sample_ratio: float
    metric_export_interval_millis: int
    export_timeout_millis: int


def build_otlp_export_configuration(settings: ApiSettings) -> OtlpExportConfiguration | None:
    """Build signal endpoints from an enabled settings object without logging secrets."""
    if not settings.otel_enabled:
        return None
    if not settings.otel_endpoint:
        raise ValueError("KYC_OTEL_ENDPOINT is required when KYC_OTEL_ENABLED is true")

    return OtlpExportConfiguration(
        traces_endpoint=_signal_endpoint(settings.otel_endpoint, "traces"),
        metrics_endpoint=_signal_endpoint(settings.otel_endpoint, "metrics"),
        headers=_parse_headers(
            settings.otel_headers.get_secret_value() if settings.otel_headers else ""
        ),
        trace_sample_ratio=settings.otel_trace_sample_ratio,
        metric_export_interval_millis=_seconds_to_millis(
            settings.otel_metric_export_interval_seconds
        ),
        export_timeout_millis=_seconds_to_millis(settings.otel_export_timeout_seconds),
    )


def configure_tracing(
    *,
    environment: str,
    service_version: str,
    export: OtlpExportConfiguration | None = None,
) -> TracerProvider | None:
    """Install an SDK provider once, preserving a host-supplied provider if present."""
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return current
    if not isinstance(current, ProxyTracerProvider):
        return None

    provider = build_tracer_provider(
        environment=environment,
        service_version=service_version,
        export=export,
    )
    trace.set_tracer_provider(provider)
    installed = trace.get_tracer_provider()
    return installed if isinstance(installed, TracerProvider) else None


def configure_metrics(
    *,
    environment: str,
    service_version: str,
    export: OtlpExportConfiguration | None = None,
) -> MeterProvider | None:
    """Install an SDK meter provider once, with OTLP export only when enabled."""
    current = metrics.get_meter_provider()
    if isinstance(current, MeterProvider):
        return current
    if type(current).__name__ != "_ProxyMeterProvider":
        return None

    provider = build_meter_provider(
        environment=environment,
        service_version=service_version,
        export=export,
    )
    metrics.set_meter_provider(provider)
    installed = metrics.get_meter_provider()
    return installed if isinstance(installed, MeterProvider) else None


def build_tracer_provider(
    *,
    environment: str,
    service_version: str,
    export: OtlpExportConfiguration | None = None,
) -> TracerProvider:
    """Create an uninstalled provider for startup and isolated configuration tests."""
    provider = TracerProvider(
        resource=_resource(environment=environment, service_version=service_version),
        sampler=(
            ParentBased(TraceIdRatioBased(export.trace_sample_ratio))
            if export is not None
            else None
        ),
    )
    if export is not None:
        exporter = OTLPSpanExporter(
            endpoint=export.traces_endpoint,
            headers=export.headers,
            timeout=export.export_timeout_millis / 1000,
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                export_timeout_millis=export.export_timeout_millis,
            )
        )
    return provider


def build_meter_provider(
    *,
    environment: str,
    service_version: str,
    export: OtlpExportConfiguration | None = None,
) -> MeterProvider:
    """Create an uninstalled meter provider with an optional periodic OTLP reader."""
    readers = ()
    if export is not None:
        exporter = OTLPMetricExporter(
            endpoint=export.metrics_endpoint,
            headers=export.headers,
            timeout=export.export_timeout_millis / 1000,
        )
        readers = (
            PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=export.metric_export_interval_millis,
                export_timeout_millis=export.export_timeout_millis,
            ),
        )
    return MeterProvider(
        resource=_resource(environment=environment, service_version=service_version),
        metric_readers=readers,
    )


def flush_tracing(*, timeout_millis: int = 10_000) -> None:
    """Bounded processor flush without shutting down the process-global provider."""
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        return
    try:
        provider.force_flush(timeout_millis=timeout_millis)
    except Exception as exc:
        logger.warning(
            "tracing flush failed",
            extra={"event": "tracing_flush_failed", "exception_type": type(exc).__name__},
        )


def flush_metrics(*, timeout_millis: int = 10_000) -> None:
    """Bounded reader flush without shutting down the process-global provider."""
    provider = metrics.get_meter_provider()
    if not isinstance(provider, MeterProvider):
        return
    try:
        provider.force_flush(timeout_millis=timeout_millis)
    except Exception as exc:
        logger.warning(
            "metrics flush failed",
            extra={"event": "metrics_flush_failed", "exception_type": type(exc).__name__},
        )


def _resource(*, environment: str, service_version: str) -> Resource:
    return Resource.create(
        {
            "service.name": "kyc-api",
            "service.version": service_version,
            "deployment.environment.name": environment,
        }
    )


def _signal_endpoint(base_endpoint: str, signal: str) -> str:
    parsed = urlsplit(base_endpoint)
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/v1/{signal}", "", ""))


def _parse_headers(raw_headers: str) -> Mapping[str, str]:
    if not raw_headers.strip():
        return {}
    headers: dict[str, str] = {}
    for item in raw_headers.split(","):
        name, separator, value = item.partition("=")
        name = name.strip()
        if not separator or not name or not _HEADER_NAME.fullmatch(name):
            raise ValueError("KYC_OTEL_HEADERS must use comma-separated Header=Value entries")
        if "\r" in value or "\n" in value:
            raise ValueError("KYC_OTEL_HEADERS must not contain newline characters")
        headers[name] = value.strip()
    return headers


def _seconds_to_millis(value: float) -> int:
    milliseconds = round(value * 1000)
    if milliseconds <= 0:
        raise ValueError("OTLP timing settings must be at least one millisecond")
    return milliseconds
