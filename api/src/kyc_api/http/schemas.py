from __future__ import annotations

from datetime import datetime
from typing import Mapping

from pydantic import BaseModel, ConfigDict
from ..domain.verification import (
    CaptureStatus,
    DocumentSide,
    FaceMatchStatus,
    LivenessStatus,
    NextAction,
    NifVerificationStatus,
    PublicDocumentStatus,
    PublicFaceComparisonStatus,
    PublicLivenessStatus,
    SessionStatus,
)


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


class NifVerificationResponse(BaseModel):
    """Public-safe external registry outcome; it deliberately contains no PII."""

    model_config = ConfigDict(frozen=True)

    status: NifVerificationStatus
    source: str | None = None
    name_match: bool | None = None


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
    nif_verification: NifVerificationResponse


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


class BrowserTokenResponse(BaseModel):
    """Hosted verifier URL; the raw browser credential appears only in its fragment."""

    model_config = ConfigDict(frozen=True)

    verification_url: str
    expires_at: datetime


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
    nif_verification: NifVerificationResponse


class NifVerificationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    nif: str
    claimed_name: str | None = None


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
