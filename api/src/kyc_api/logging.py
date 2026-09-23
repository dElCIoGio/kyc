from __future__ import annotations

import json
import logging
import sys
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace


_SAFE_FIELDS = (
    "request_id", "session_id", "job_id", "side", "sides", "stage", "status", "duration_ms",
    "error_code", "exception_type", "field",
)
_HANDLER_MARKER = "_kyc_json_handler"


@dataclass(frozen=True)
class LoggingContext:
    request_id: str | None = None
    session_id: str | None = None
    job_id: str | None = None


_logging_context: ContextVar[LoggingContext] = ContextVar(
    "kyc_logging_context",
    default=LoggingContext(),
)


def get_logging_context() -> LoggingContext:
    return _logging_context.get()


@contextmanager
def logging_context(
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    job_id: str | None = None,
):
    """Add safe correlation identifiers for one synchronous or async scope."""
    current = get_logging_context()
    token = _logging_context.set(
        LoggingContext(
            request_id=request_id if request_id is not None else current.request_id,
            session_id=session_id if session_id is not None else current.session_id,
            job_id=job_id if job_id is not None else current.job_id,
        )
    )
    try:
        yield
    finally:
        _logging_context.reset(token)


@contextmanager
def job_logging_context(*, session_id: str, job_id: str):
    """Bind only durable job context, excluding any request context."""
    token = _logging_context.set(LoggingContext(session_id=session_id, job_id=job_id))
    try:
        yield
    finally:
        _logging_context.reset(token)


class JsonFormatter(logging.Formatter):
    """Render application records without serializing arbitrary log values."""

    def __init__(self, *, environment: str) -> None:
        super().__init__()
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        context = get_logging_context()
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "service": "kyc-api",
            "environment": self._environment,
            "event": getattr(record, "event", "log"),
        }
        for name in ("request_id", "session_id", "job_id"):
            value = getattr(context, name) or getattr(record, name, None)
            if value is not None:
                payload[name] = value
        for name in _SAFE_FIELDS:
            if name in {"request_id", "session_id", "job_id"}:
                continue
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        span = trace.get_current_span()
        span_context = span.get_span_context()
        if span.is_recording() and span_context.is_valid:
            payload["trace_id"] = trace.format_trace_id(span_context.trace_id)
            payload["span_id"] = trace.format_span_id(span_context.span_id)
        if record.exc_info is not None:
            exception_type, _exception, traceback_value = record.exc_info
            if exception_type is not None:
                payload.setdefault("exception_type", exception_type.__name__)
            if traceback_value is not None:
                payload["traceback"] = [
                    {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                    for frame in traceback.extract_tb(traceback_value)
                ]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def configure_logging(*, level: str, environment: str) -> None:
    """Install JSON stdout handlers for API and engine application loggers."""
    resolved_level = logging.getLevelName(level.upper())
    if not isinstance(resolved_level, int):
        raise ValueError("log level must be a standard Python logging level")
    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(JsonFormatter(environment=environment))
    for name in ("kyc_api", "kyc_engine"):
        logger = logging.getLogger(name)
        for existing in tuple(logger.handlers):
            if getattr(existing, _HANDLER_MARKER, False):
                logger.removeHandler(existing)
                existing.close()
        logger.addHandler(handler)
        logger.setLevel(resolved_level)
        logger.propagate = False
