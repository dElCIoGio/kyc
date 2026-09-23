from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Mapping

from opentelemetry import metrics as otel_metrics
from opentelemetry.metrics import MeterProvider

from kyc_engine import DocumentExtractionResult
from kyc_engine.profiles import load_default_profiles


_CONTROLLED_FIELD_NAMES = frozenset(
    field.name
    for profile in load_default_profiles()
    for field in profile.fields
)
_KNOWN_SIDES = frozenset({"front", "back"})
_KNOWN_JOB_OUTCOMES = frozenset({"success", "partial", "failed", "timeout"})
_KNOWN_STAGE_NAMES = frozenset(
    {
        "intake",
        "detection",
        "profile",
        "normalization",
        "variants",
        "quality",
        "qr",
        "field_localization",
        "ocr",
        "reconciliation",
    }
)

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

    def __init__(self, *, meter_provider: MeterProvider | None = None) -> None:
        self._started_at = monotonic()
        self._response_statuses: Counter[str] = Counter()
        self._error_codes: Counter[str] = Counter()
        self._job_outcomes: Counter[str] = Counter()
        self._request_latency = _Latency()
        self._job_latency = _Latency()
        self._lock = RLock()
        meter = (meter_provider or otel_metrics.get_meter_provider()).get_meter(
            "kyc_api.metrics"
        )
        self._job_outcome_counter = meter.create_counter(
            "kyc.jobs",
            description="Completed KYC processing jobs by terminal outcome",
        )
        self._job_duration = meter.create_histogram(
            "kyc.job.duration",
            unit="s",
            description="KYC processing job duration",
        )
        self._stage_execution_counter = meter.create_counter(
            "kyc.stage.executions",
            description="KYC semantic pipeline stage executions",
        )
        self._stage_failure_counter = meter.create_counter(
            "kyc.stage.failures",
            description="Failed KYC semantic pipeline stages",
        )
        self._stage_duration = meter.create_histogram(
            "kyc.stage.duration",
            unit="s",
            description="KYC semantic pipeline stage duration",
        )
        self._field_status_counter = meter.create_counter(
            "kyc.field.status",
            description="Final KYC extracted-field statuses",
        )

    def record_response(self, status_code: int, duration_ms: float) -> None:
        with self._lock:
            self._response_statuses[str(status_code)] += 1
            self._request_latency.record(duration_ms)

    def record_error(self, code: str) -> None:
        with self._lock:
            self._error_codes[code] += 1

    def record_job(self, outcome: str, duration_ms: float) -> None:
        safe_outcome = outcome if outcome in _KNOWN_JOB_OUTCOMES else "unknown"
        with self._lock:
            self._job_outcomes[safe_outcome] += 1
            self._job_latency.record(duration_ms)
        attributes = {"status": safe_outcome}
        self._record_otel(lambda: self._job_outcome_counter.add(1, attributes))
        self._record_otel(
            lambda: self._job_duration.record(max(0.0, duration_ms) / 1000.0, attributes)
        )

    def record_stage(
        self,
        *,
        stage: str,
        side: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        execution_attributes = {
            "stage": stage if stage in _KNOWN_STAGE_NAMES else "unknown",
            "side": side if side in _KNOWN_SIDES else "unknown",
            "status": status if status in {"success", "failed"} else "unknown",
        }
        duration_attributes = {
            "stage": execution_attributes["stage"],
            "side": execution_attributes["side"],
        }
        self._record_otel(lambda: self._stage_execution_counter.add(1, execution_attributes))
        self._record_otel(
            lambda: self._stage_duration.record(max(0.0, duration_seconds), duration_attributes)
        )
        if status == "failed":
            self._record_otel(lambda: self._stage_failure_counter.add(1, duration_attributes))

    def record_field_statuses(self, result: DocumentExtractionResult) -> None:
        for default_side, side_result in (("front", result.front), ("back", result.back)):
            if side_result is None:
                continue
            side = side_result.side if side_result.side in _KNOWN_SIDES else default_side
            for field in side_result.fields.values():
                if field.name not in _CONTROLLED_FIELD_NAMES:
                    continue
                attributes = {
                    "side": side,
                    "field": field.name,
                    "status": field.status.value,
                }
                self._record_otel(lambda: self._field_status_counter.add(1, attributes))

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

    @staticmethod
    def _record_otel(operation) -> None:
        try:
            operation()
        except Exception:
            # Metrics are observational and must never change KYC behavior.
            return


def _rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None
