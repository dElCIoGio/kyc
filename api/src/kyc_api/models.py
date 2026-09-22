from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class SessionStatus(StrEnum):
    CREATED = "created"
    UPLOADING = "uploading"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    EXPIRED = "expired"


class DocumentSide(StrEnum):
    FRONT = "front"
    BACK = "back"


class SessionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    status: SessionStatus
    created_at: datetime
    expires_at: datetime
    job_id: str | None = None
    uploaded_sides: tuple[DocumentSide, ...] = ()
    result_available: bool = False


class UploadResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    status: SessionStatus
    side: DocumentSide
    size_bytes: int
    uploaded_sides: tuple[DocumentSide, ...]
    expires_at: datetime


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    job_id: str
    status: SessionStatus


class DeleteResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    deleted: bool = True


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    service: str = "angolan-kyc-api"
    version: str


class ErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: ErrorDetail
