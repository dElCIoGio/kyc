from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_SAFE_FIELDS = (
    "session_id", "job_id", "side", "sides", "stage", "status", "duration_ms",
    "error_code", "exception_type", "field",
)
_HANDLER_MARKER = "_kyc_json_handler"


class JsonFormatter(logging.Formatter):
    """Render application records without serializing arbitrary log values."""

    def __init__(self, *, environment: str) -> None:
        super().__init__()
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "service": "kyc-api",
            "environment": self._environment,
            "event": getattr(record, "event", "log"),
        }
        for name in _SAFE_FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
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
