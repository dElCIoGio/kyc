from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock
from uuid import uuid4

from kyc_engine import DocumentExtractionResult, ProcessingStatus

from .models import DocumentSide, SessionStatus


class SessionStoreError(RuntimeError):
    code = "SESSION_ERROR"


class SessionNotFound(SessionStoreError):
    code = "SESSION_NOT_FOUND"


class SessionConflict(SessionStoreError):
    code = "INVALID_SESSION_STATE"


class SessionCapacityExceeded(SessionStoreError):
    code = "SESSION_CAPACITY_EXCEEDED"


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    status: SessionStatus
    created_at: datetime
    expires_at: datetime
    job_id: str | None
    uploaded_sides: tuple[DocumentSide, ...]
    result_available: bool
    error_code: str | None


@dataclass
class _SessionRecord:
    session_id: str
    status: SessionStatus
    created_at: datetime
    expires_at: datetime
    front: bytes | None = None
    back: bytes | None = None
    job_id: str | None = None
    result: DocumentExtractionResult | None = None
    error_code: str | None = None


class SessionStore:
    """Thread-safe, memory-only storage for document extraction sessions."""

    _ACTIVE = {SessionStatus.QUEUED, SessionStatus.RUNNING}
    _TERMINAL = {
        SessionStatus.SUCCESS,
        SessionStatus.PARTIAL,
        SessionStatus.FAILED,
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
            session_id = uuid4().hex
            record = _SessionRecord(
                session_id=session_id,
                status=SessionStatus.CREATED,
                created_at=now,
                expires_at=now + self._ttl,
            )
            self._records[session_id] = record
            return _snapshot(record)

    def get(self, session_id: str) -> SessionSnapshot:
        with self._lock:
            record = self._get_record(session_id)
            return _snapshot(record)

    def upload(
        self,
        session_id: str,
        side: DocumentSide,
        content: bytes,
    ) -> SessionSnapshot:
        with self._lock:
            record = self._get_record(session_id)
            if record.status not in {SessionStatus.CREATED, SessionStatus.UPLOADING}:
                raise SessionConflict("Images cannot be changed after processing starts")
            if side == DocumentSide.FRONT:
                record.front = bytes(content)
            else:
                record.back = bytes(content)
            record.status = SessionStatus.UPLOADING
            self._refresh_expiry(record)
            return _snapshot(record)

    def queue(self, session_id: str) -> SessionSnapshot:
        with self._lock:
            record = self._get_record(session_id)
            if record.status not in {SessionStatus.CREATED, SessionStatus.UPLOADING}:
                raise SessionConflict("Processing has already started")
            if record.front is None and record.back is None:
                raise SessionConflict("At least one document side must be uploaded")
            record.job_id = uuid4().hex
            record.status = SessionStatus.QUEUED
            record.error_code = None
            self._refresh_expiry(record)
            return _snapshot(record)

    def start(self, session_id: str, job_id: str) -> tuple[bytes | None, bytes | None]:
        with self._lock:
            record = self._get_record(session_id)
            if record.status != SessionStatus.QUEUED or record.job_id != job_id:
                raise SessionConflict("The queued job is no longer available")
            record.status = SessionStatus.RUNNING
            self._refresh_expiry(record)
            return record.front, record.back

    def complete(
        self,
        session_id: str,
        job_id: str,
        result: DocumentExtractionResult,
    ) -> bool:
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.job_id != job_id
                or record.status != SessionStatus.RUNNING
            ):
                return False
            record.front = None
            record.back = None
            record.result = result
            record.error_code = None
            record.status = {
                ProcessingStatus.SUCCESS: SessionStatus.SUCCESS,
                ProcessingStatus.PARTIAL: SessionStatus.PARTIAL,
                ProcessingStatus.FAILED: SessionStatus.FAILED,
            }[result.status]
            self._refresh_expiry(record)
            return True

    def fail(self, session_id: str, job_id: str, code: str) -> bool:
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.job_id != job_id
                or record.status not in {SessionStatus.QUEUED, SessionStatus.RUNNING}
            ):
                return False
            record.front = None
            record.back = None
            record.result = None
            record.error_code = code
            record.status = SessionStatus.FAILED
            self._refresh_expiry(record)
            return True

    def timeout(self, session_id: str, job_id: str) -> bool:
        """Fail an overdue running job and invalidate any later completion."""
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.job_id != job_id
                or record.status != SessionStatus.RUNNING
            ):
                return False
            record.front = None
            record.back = None
            record.result = None
            record.error_code = "JOB_TIMEOUT"
            record.status = SessionStatus.FAILED
            self._refresh_expiry(record)
            return True

    def result(self, session_id: str) -> DocumentExtractionResult:
        with self._lock:
            record = self._get_record(session_id)
            if record.status not in self._TERMINAL:
                raise SessionConflict("The extraction result is not ready")
            if record.result is None:
                raise SessionStoreError(record.error_code or "JOB_FAILED")
            return record.result

    def delete(self, session_id: str) -> None:
        with self._lock:
            record = self._records.pop(session_id, None)
            if record is not None:
                _clear_sensitive_state(record)

    def contains_images(self, session_id: str) -> bool:
        """Testing and diagnostics helper; never exposes image bytes."""
        with self._lock:
            record = self._records.get(session_id)
            return bool(record and (record.front is not None or record.back is not None))

    def cleanup(self) -> int:
        """Remove expired terminal sessions without exposing retained state."""
        with self._lock:
            return self._purge_expired(_now())

    def _get_record(self, session_id: str) -> _SessionRecord:
        self._purge_expired(_now())
        try:
            return self._records[session_id]
        except KeyError as exc:
            raise SessionNotFound("Session was not found") from exc

    def _purge_expired(self, now: datetime) -> int:
        expired = [
            session_id
            for session_id, record in self._records.items()
            if record.status not in self._ACTIVE and record.expires_at <= now
        ]
        for session_id in expired:
            record = self._records.pop(session_id)
            record.status = SessionStatus.EXPIRED
            _clear_sensitive_state(record)
        return len(expired)

    def _refresh_expiry(self, record: _SessionRecord) -> None:
        record.expires_at = _now() + self._ttl


def _snapshot(record: _SessionRecord) -> SessionSnapshot:
    sides: list[DocumentSide] = []
    if record.front is not None:
        sides.append(DocumentSide.FRONT)
    if record.back is not None:
        sides.append(DocumentSide.BACK)
    return SessionSnapshot(
        session_id=record.session_id,
        status=record.status,
        created_at=record.created_at,
        expires_at=record.expires_at,
        job_id=record.job_id,
        uploaded_sides=tuple(sides),
        result_available=record.result is not None,
        error_code=record.error_code,
    )


def _clear_sensitive_state(record: _SessionRecord) -> None:
    record.front = None
    record.back = None
    record.result = None


def _now() -> datetime:
    return datetime.now(UTC)
