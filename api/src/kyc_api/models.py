from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Mapping

from pydantic import BaseModel, ConfigDict


class VerificationStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DocumentStatus(StrEnum):
    AWAITING_CAPTURE = "awaiting_capture"
    READY = "ready"
    QUEUED = "queued"
    PROCESSING = "processing"
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"


class LivenessStatus(StrEnum):
    BLOCKED = "blocked"
    READY = "ready"
    PROCESSING = "processing"
    PASSED = "passed"
    FAILED = "failed"


class FaceMatchStatus(StrEnum):
    BLOCKED = "blocked"
    READY = "ready"
    PROCESSING = "processing"
    PASSED = "passed"
    FAILED = "failed"


class CaptureStatus(StrEnum):
    MISSING = "missing"
    ACCEPTED = "accepted"


class DocumentSide(StrEnum):
    FRONT = "front"
    BACK = "back"


class CaptureIssueResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str


class DocumentStateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: DocumentStatus
    front_capture: CaptureStatus
    back_capture: CaptureStatus
    job_id: str | None = None
    result_available: bool = False
    error_code: str | None = None


class SubsystemStateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: LivenessStatus | FaceMatchStatus


class SessionResponse(BaseModel):
    """Public representation of one verification session."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    verification_status: VerificationStatus
    created_at: datetime
    expires_at: datetime
    document: DocumentStateResponse
    liveness: SubsystemStateResponse
    face_match: SubsystemStateResponse


class CaptureResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    side: DocumentSide
    accepted: bool
    issues: tuple[CaptureIssueResponse, ...] = ()
    verification: SessionResponse


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    job_id: str | None = None
    document_status: DocumentStatus


class DeleteResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    deleted: bool = True


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    service: str = "angolan-kyc-api"
    version: str


class LatencyMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    min_ms: float | None
    max_ms: float | None
    mean_ms: float | None


class MetricsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    service: str = "angolan-kyc-api"
    version: str
    uptime_seconds: float
    responses_by_status: Mapping[str, int]
    errors_by_code: Mapping[str, int]
    jobs_by_outcome: Mapping[str, int]
    request_latency_ms: LatencyMetrics
    job_latency_ms: LatencyMetrics


class ErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: ErrorDetail
