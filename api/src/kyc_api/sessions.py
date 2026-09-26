from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock
from uuid import uuid4

from kyc_engine import CaptureAssessment, DocumentExtractionResult, ProcessingStatus

from .models import (
    CaptureStatus,
    DocumentSide,
    DocumentStatus,
    FaceMatchStatus,
    LivenessStatus,
    VerificationStatus,
)


class SessionStoreError(RuntimeError):
    code = "SESSION_ERROR"


class SessionNotFound(SessionStoreError):
    code = "SESSION_NOT_FOUND"


class SessionConflict(SessionStoreError):
    code = "INVALID_SESSION_STATE"


class SessionCapacityExceeded(SessionStoreError):
    code = "SESSION_CAPACITY_EXCEEDED"


@dataclass(frozen=True)
class DocumentSnapshot:
    status: DocumentStatus
    front_capture: CaptureStatus
    back_capture: CaptureStatus
    job_id: str | None
    result_available: bool
    error_code: str | None


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    verification_status: VerificationStatus
    created_at: datetime
    expires_at: datetime
    document: DocumentSnapshot
    liveness_status: LivenessStatus
    face_match_status: FaceMatchStatus
    event_sequence: int = 0


@dataclass(frozen=True)
class DocumentCaptureTransition:
    snapshot: SessionSnapshot
    capture_accepted: SessionSnapshot
    capture_completed: SessionSnapshot | None


@dataclass
class _DocumentState:
    status: DocumentStatus = DocumentStatus.AWAITING_CAPTURE
    front: bytes | None = None
    back: bytes | None = None
    front_capture: CaptureAssessment | None = None
    back_capture: CaptureAssessment | None = None
    job_id: str | None = None
    result: DocumentExtractionResult | None = None
    error_code: str | None = None


@dataclass
class _LivenessState:
    status: LivenessStatus = LivenessStatus.BLOCKED


@dataclass
class _FaceMatchState:
    status: FaceMatchStatus = FaceMatchStatus.BLOCKED


@dataclass
class _SessionRecord:
    session_id: str
    status: VerificationStatus
    created_at: datetime
    expires_at: datetime
    document: _DocumentState
    liveness: _LivenessState
    face_match: _FaceMatchState
    event_sequence: int = 0


class SessionStore:
    """Thread-safe, memory-only storage for complete verification sessions."""

    _DOCUMENT_PROCESSING = {DocumentStatus.QUEUED, DocumentStatus.PROCESSING}
    _DOCUMENT_RESULTS = {DocumentStatus.PASSED, DocumentStatus.PARTIAL, DocumentStatus.FAILED}
    _VERIFICATION_TERMINAL = {
        VerificationStatus.VERIFIED,
        VerificationStatus.REJECTED,
        VerificationStatus.EXPIRED,
    }

    def __init__(self, *, ttl_seconds: int, max_sessions: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_sessions = max_sessions
        self._records: dict[str, _SessionRecord] = {}
        self._lock = RLock()

    def create(self) -> SessionSnapshot:
        with self._lock:
            now = _now()
            self._purge_expired(now)
            if len(self._records) >= self._max_sessions:
                raise SessionCapacityExceeded("Session capacity has been reached")
            record = _SessionRecord(
                session_id=uuid4().hex,
                status=VerificationStatus.IN_PROGRESS,
                created_at=now,
                expires_at=now + self._ttl,
                document=_DocumentState(),
                liveness=_LivenessState(),
                face_match=_FaceMatchState(),
            )
            self._records[record.session_id] = record
            return _snapshot(record)

    def get(self, session_id: str) -> SessionSnapshot:
        with self._lock:
            return _snapshot(self._get_record(session_id))

    def accept_document_capture(
        self,
        session_id: str,
        side: DocumentSide,
        content: bytes,
        assessment: CaptureAssessment,
    ) -> DocumentCaptureTransition:
        if not assessment.accepted:
            raise ValueError("Only accepted captures can be stored")
        with self._lock:
            record = self._get_record(session_id)
            document = record.document
            if record.status != VerificationStatus.IN_PROGRESS:
                raise SessionConflict("Captures cannot be changed after verification ends")
            if document.status != DocumentStatus.AWAITING_CAPTURE:
                raise SessionConflict("Captures cannot be changed after capture completion")
            if side == DocumentSide.FRONT:
                document.front = bytes(content)
                document.front_capture = assessment
            else:
                document.back = bytes(content)
                document.back_capture = assessment
            document.error_code = None
            self._refresh_expiry(record)
            record.event_sequence += 1
            accepted = _snapshot(record)

            completed: SessionSnapshot | None = None
            if document.front is not None and document.back is not None:
                document.status = DocumentStatus.READY
                record.liveness.status = LivenessStatus.READY
                record.event_sequence += 1
                completed = _snapshot(record)
            return DocumentCaptureTransition(
                snapshot=completed or accepted,
                capture_accepted=accepted,
                capture_completed=completed,
            )

    def queue_document_processing(self, session_id: str) -> SessionSnapshot:
        with self._lock:
            record = self._get_record(session_id)
            document = record.document
            if record.status != VerificationStatus.IN_PROGRESS:
                raise SessionConflict("Verification is no longer active")
            if document.status != DocumentStatus.READY:
                raise SessionConflict("Document processing is not ready to queue")
            if document.front is None or document.back is None:
                raise SessionConflict("Both document sides must be accepted")
            document.job_id = uuid4().hex
            document.status = DocumentStatus.QUEUED
            document.error_code = None
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def start_document_processing(
        self, session_id: str, job_id: str
    ) -> tuple[bytes, bytes, SessionSnapshot]:
        with self._lock:
            record = self._get_record(session_id)
            document = record.document
            if (
                record.status != VerificationStatus.IN_PROGRESS
                or document.status != DocumentStatus.QUEUED
                or document.job_id != job_id
                or document.front is None
                or document.back is None
            ):
                raise SessionConflict("The queued document job is no longer available")
            document.status = DocumentStatus.PROCESSING
            record.event_sequence += 1
            self._refresh_expiry(record)
            return document.front, document.back, _snapshot(record)

    def complete_document_processing(
        self,
        session_id: str,
        job_id: str,
        result: DocumentExtractionResult,
    ) -> SessionSnapshot | None:
        with self._lock:
            record = self._records.get(session_id)
            if not self._can_finish_document_job(record, job_id):
                return None
            assert record is not None
            document = record.document
            _clear_document_sources(document)
            document.result = result
            document.error_code = None
            document.status = {
                ProcessingStatus.SUCCESS: DocumentStatus.PASSED,
                ProcessingStatus.PARTIAL: DocumentStatus.PARTIAL,
                ProcessingStatus.FAILED: DocumentStatus.FAILED,
            }[result.status]
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def fail_document_processing(
        self, session_id: str, job_id: str, code: str
    ) -> SessionSnapshot | None:
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.status != VerificationStatus.IN_PROGRESS
                or record.document.job_id != job_id
                or record.document.status not in self._DOCUMENT_PROCESSING
            ):
                return None
            document = record.document
            _clear_document_sources(document)
            document.result = None
            document.error_code = code
            document.status = DocumentStatus.FAILED
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def timeout_document_processing(
        self, session_id: str, job_id: str
    ) -> SessionSnapshot | None:
        """Fail an overdue running job and invalidate any later completion."""
        with self._lock:
            record = self._records.get(session_id)
            if not self._can_finish_document_job(record, job_id):
                return None
            assert record is not None
            document = record.document
            _clear_document_sources(document)
            document.result = None
            document.error_code = "JOB_TIMEOUT"
            document.status = DocumentStatus.FAILED
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def result(self, session_id: str) -> DocumentExtractionResult:
        with self._lock:
            record = self._get_record(session_id)
            document = record.document
            if document.status not in self._DOCUMENT_RESULTS:
                raise SessionConflict("The document extraction result is not ready")
            if document.result is None:
                raise SessionStoreError(document.error_code or "JOB_FAILED")
            return document.result

    def delete(self, session_id: str) -> None:
        with self._lock:
            record = self._records.pop(session_id, None)
            if record is not None:
                _clear_sensitive_state(record)

    def contains_images(self, session_id: str) -> bool:
        """Testing and diagnostics helper; never exposes image bytes."""
        with self._lock:
            record = self._records.get(session_id)
            return bool(record and (record.document.front is not None or record.document.back is not None))

    def cleanup(self) -> int:
        with self._lock:
            return self._purge_expired(_now())

    def _get_record(self, session_id: str) -> _SessionRecord:
        self._purge_expired(_now())
        try:
            return self._records[session_id]
        except KeyError as exc:
            raise SessionNotFound("Session was not found") from exc

    def _can_finish_document_job(self, record: _SessionRecord | None, job_id: str) -> bool:
        return bool(
            record is not None
            and record.status == VerificationStatus.IN_PROGRESS
            and record.document.job_id == job_id
            and record.document.status == DocumentStatus.PROCESSING
        )

    def _purge_expired(self, now: datetime) -> int:
        expired = [
            session_id
            for session_id, record in self._records.items()
            if record.document.status not in self._DOCUMENT_PROCESSING
            and record.expires_at <= now
        ]
        for session_id in expired:
            record = self._records.pop(session_id)
            record.status = VerificationStatus.EXPIRED
            _clear_sensitive_state(record)
        return len(expired)

    def _refresh_expiry(self, record: _SessionRecord) -> None:
        record.expires_at = _now() + self._ttl


def _snapshot(record: _SessionRecord) -> SessionSnapshot:
    document = record.document
    return SessionSnapshot(
        session_id=record.session_id,
        verification_status=record.status,
        created_at=record.created_at,
        expires_at=record.expires_at,
        document=DocumentSnapshot(
            status=document.status,
            front_capture=(
                CaptureStatus.ACCEPTED
                if document.front_capture is not None
                else CaptureStatus.MISSING
            ),
            back_capture=(
                CaptureStatus.ACCEPTED
                if document.back_capture is not None
                else CaptureStatus.MISSING
            ),
            job_id=document.job_id,
            result_available=document.result is not None,
            error_code=document.error_code,
        ),
        liveness_status=record.liveness.status,
        face_match_status=record.face_match.status,
        event_sequence=record.event_sequence,
    )


def _clear_document_sources(document: _DocumentState) -> None:
    document.front = None
    document.back = None


def _clear_sensitive_state(record: _SessionRecord) -> None:
    _clear_document_sources(record.document)
    record.document.front_capture = None
    record.document.back_capture = None
    record.document.result = None


def _now() -> datetime:
    return datetime.now(UTC)
