from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode


_side: ContextVar[str | None] = ContextVar("kyc_pipeline_side", default=None)
_logger = logging.getLogger("kyc_engine.pipeline")
_tracer = trace.get_tracer("kyc_engine.pipeline")


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
            failure = {
                "event": "pipeline_stage_failed",
                "stage": stage,
                "status": "failed",
                "duration_ms": round((perf_counter() - started) * 1000.0, 3),
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
            completed = {
                "event": "pipeline_stage_completed",
                "stage": stage,
                "status": "success",
                "duration_ms": round((perf_counter() - started) * 1000.0, 3),
            }
            if side is not None:
                completed["side"] = side
            _logger.info("pipeline stage completed", extra=completed)
