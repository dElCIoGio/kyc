from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Mapping


@dataclass
class _Latency:
    count: int = 0
    total_ms: float = 0.0
    minimum_ms: float | None = None
    maximum_ms: float | None = None

    def record(self, duration_ms: float) -> None:
        value = max(0.0, duration_ms)
        self.count += 1
        self.total_ms += value
        self.minimum_ms = value if self.minimum_ms is None else min(self.minimum_ms, value)
        self.maximum_ms = value if self.maximum_ms is None else max(self.maximum_ms, value)

    def snapshot(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "min_ms": _rounded(self.minimum_ms),
            "max_ms": _rounded(self.maximum_ms),
            "mean_ms": _rounded(self.total_ms / self.count) if self.count else None,
        }


class MetricsRegistry:
    """Thread-safe process-lifetime aggregates with no request identifiers."""

    def __init__(self) -> None:
        self._started_at = monotonic()
        self._response_statuses: Counter[str] = Counter()
        self._error_codes: Counter[str] = Counter()
        self._job_outcomes: Counter[str] = Counter()
        self._request_latency = _Latency()
        self._job_latency = _Latency()
        self._lock = RLock()

    def record_response(self, status_code: int, duration_ms: float) -> None:
        with self._lock:
            self._response_statuses[str(status_code)] += 1
            self._request_latency.record(duration_ms)

    def record_error(self, code: str) -> None:
        with self._lock:
            self._error_codes[code] += 1

    def record_job(self, outcome: str, duration_ms: float) -> None:
        with self._lock:
            self._job_outcomes[outcome] += 1
            self._job_latency.record(duration_ms)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "uptime_seconds": round(max(0.0, monotonic() - self._started_at), 3),
                "responses_by_status": dict(sorted(self._response_statuses.items())),
                "errors_by_code": dict(sorted(self._error_codes.items())),
                "jobs_by_outcome": dict(sorted(self._job_outcomes.items())),
                "request_latency_ms": self._request_latency.snapshot(),
                "job_latency_ms": self._job_latency.snapshot(),
            }


def _rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None
