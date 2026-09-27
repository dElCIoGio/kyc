from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Mapping

from pydantic import BaseModel, ConfigDict


class VerificationStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SessionStatus(StrEnum):
    """Customer-facing technical workflow status, never an identity decision."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class NextAction(StrEnum):
    """The only actions a customer can take on the current session."""

    SUBMIT_DOCUMENT_FRONT = "submit_document_front"
    SUBMIT_DOCUMENT_BACK = "submit_document_back"
    SUBMIT_LIVENESS = "submit_liveness"
    WAIT = "wait"


class PublicDocumentStatus(StrEnum):
    AWAITING_CAPTURE = "awaiting_capture"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PublicLivenessStatus(StrEnum):
    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    PASSED = "passed"
    FAILED = "failed"
    NOT_AVAILABLE = "not_available"


class PublicFaceComparisonStatus(StrEnum):
    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_AVAILABLE = "not_available"


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
    COMPLETED = "completed"
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

    status: PublicDocumentStatus
    front_capture: CaptureStatus
    back_capture: CaptureStatus
    result_available: bool = False


class LivenessStateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: PublicLivenessStatus


class FaceComparisonStateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: PublicFaceComparisonStatus


class SessionResponse(BaseModel):
    """Public representation of one verification session."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    status: SessionStatus
    created_at: datetime
    expires_at: datetime
    next_action: NextAction | None
    document: DocumentStateResponse
    liveness: LivenessStateResponse
    face_comparison: FaceComparisonStateResponse


class CaptureResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    side: DocumentSide
    accepted: bool
    issues: tuple[CaptureIssueResponse, ...] = ()
    session: SessionResponse


class LivenessSubmissionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session: SessionResponse


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session: SessionResponse


class ResultFieldResponse(BaseModel):
    """Canonical document value without OCR candidates or provenance internals."""

    model_config = ConfigDict(frozen=True)

    status: str
    value: str | None = None


class ResultIssueResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    field_name: str | None = None


class ResultDocumentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    document_type: str | None = None
    fields: Mapping[str, ResultFieldResponse]
    issues: tuple[ResultIssueResponse, ...] = ()


class VerificationResultResponse(BaseModel):
    """Public normalized document result; biometric and processing details stay internal."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    document: ResultDocumentResponse
    liveness: LivenessStateResponse
    face_comparison: FaceComparisonStateResponse


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
