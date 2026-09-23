from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Iterator


_side: ContextVar[str | None] = ContextVar("kyc_pipeline_side", default=None)
_logger = logging.getLogger("kyc_engine.pipeline")


@contextmanager
def pipeline_side_context(side: str) -> Iterator[None]:
    """Bind the coordinator's side while a synchronous pipeline runs."""
    token = _side.set(side)
    try:
        yield
    finally:
        _side.reset(token)


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
    _logger.info("pipeline stage started", extra=metadata)
    try:
        yield
    except Exception as exc:
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
