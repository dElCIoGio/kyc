from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Iterator, Protocol

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode


_side: ContextVar[str | None] = ContextVar("kyc_pipeline_side", default=None)
_metrics_recorder: ContextVar[PipelineMetricsRecorder | None] = ContextVar(
    "kyc_pipeline_metrics_recorder",
    default=None,
)
_logger = logging.getLogger("kyc_engine.pipeline")
_tracer = trace.get_tracer("kyc_engine.pipeline")


class PipelineMetricsRecorder(Protocol):
    """Dependency-neutral sink for semantic pipeline-stage aggregates."""

    def record_stage(
        self,
        *,
        stage: str,
        side: str,
        status: str,
        duration_seconds: float,
    ) -> None: ...


@contextmanager
def pipeline_metrics_context(recorder: PipelineMetricsRecorder) -> Iterator[None]:
    """Bind one application-owned metrics sink for synchronous pipeline work."""
    token = _metrics_recorder.set(recorder)
    try:
        yield
    finally:
        _metrics_recorder.reset(token)


@contextmanager
def observe_pipeline_side(side: str) -> Iterator[Span]:
    """Bind one side and create its semantic processing span."""
    token = _side.set(side)
    try:
        with _tracer.start_as_current_span(
            "kyc.process_side",
            attributes={"kyc.side": side},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except Exception as exc:
                mark_span_failed(span, exc)
                span.set_attribute("kyc.processing_status", "failed")
                raise
    finally:
        _side.reset(token)


def mark_span_failed(span: Span, exc: Exception) -> None:
    """Record only safe exception metadata on an active span."""
    span.set_attribute("exception.type", type(exc).__name__)
    span.set_status(Status(StatusCode.ERROR))


def _record_stage_metric(
    *, stage: str, side: str | None, status: str, duration_seconds: float
) -> None:
    recorder = _metrics_recorder.get()
    if recorder is None:
        return
    try:
        recorder.record_stage(
            stage=stage,
            side=side or "unknown",
            status=status,
            duration_seconds=max(0.0, duration_seconds),
        )
    except Exception as exc:
        _logger.warning(
            "pipeline metrics recording failed",
            extra={
                "event": "pipeline_metrics_recording_failed",
                "stage": stage,
                "side": side or "unknown",
                "exception_type": type(exc).__name__,
            },
        )


@contextmanager
def observe_pipeline_stage(
    stage: str,
    *,
    error_code: str | None = None,
) -> Iterator[None]:
    """Log one meaningful pipeline operation without changing its behavior."""
    side = _side.get()
    started = perf_counter()
    metadata: dict[str, object] = {"event": "pipeline_stage_started", "stage": stage}
    if side is not None:
        metadata["side"] = side
    attributes: dict[str, str] = {"kyc.stage": stage}
    if side is not None:
        attributes["kyc.side"] = side
    with _tracer.start_as_current_span(
        "kyc.stage",
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        _logger.info("pipeline stage started", extra=metadata)
        try:
            yield
        except Exception as exc:
            mark_span_failed(span, exc)
            duration_seconds = perf_counter() - started
            _record_stage_metric(
                stage=stage,
                side=side,
                status="failed",
                duration_seconds=duration_seconds,
            )
            failure = {
                "event": "pipeline_stage_failed",
                "stage": stage,
                "status": "failed",
                "duration_ms": round(duration_seconds * 1000.0, 3),
                "exception_type": type(exc).__name__,
            }
            resolved_error_code = error_code or getattr(exc, "code", None)
            if isinstance(resolved_error_code, str):
                failure["error_code"] = resolved_error_code
            if side is not None:
                failure["side"] = side
            _logger.exception("pipeline stage failed", extra=failure)
            raise
        else:
            duration_seconds = perf_counter() - started
            _record_stage_metric(
                stage=stage,
                side=side,
                status="success",
                duration_seconds=duration_seconds,
            )
            completed = {
                "event": "pipeline_stage_completed",
                "stage": stage,
                "status": "success",
                "duration_ms": round(duration_seconds * 1000.0, 3),
            }
            if side is not None:
                completed["side"] = side
            _logger.info("pipeline stage completed", extra=completed)
