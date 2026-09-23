"""Application-owned OpenTelemetry SDK setup without an external exporter."""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider


logger = logging.getLogger(__name__)


def configure_tracing(*, environment: str, service_version: str) -> TracerProvider | None:
    """Install the SDK provider once, preserving a host-supplied provider if present."""
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return current
    if not isinstance(current, ProxyTracerProvider):
        return None

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "kyc-api",
                "service.version": service_version,
                "deployment.environment.name": environment,
            }
        )
    )
    trace.set_tracer_provider(provider)
    installed = trace.get_tracer_provider()
    return installed if isinstance(installed, TracerProvider) else None


def flush_tracing() -> None:
    """Flush configured processors without shutting down the process-global provider."""
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        return
    try:
        provider.force_flush()
    except Exception as exc:
        logger.warning(
            "tracing flush failed",
            extra={
                "event": "tracing_flush_failed",
                "exception_type": type(exc).__name__,
            },
        )
