from __future__ import annotations

from enum import StrEnum


class VerificationStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
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


class NifVerificationStatus(StrEnum):
    NOT_RUN = "not_run"
    PROCESSING = "processing"
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


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
