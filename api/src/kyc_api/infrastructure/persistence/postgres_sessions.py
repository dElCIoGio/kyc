from __future__ import annotations

from dataclasses import dataclass
from contextvars import ContextVar
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from kyc_engine import CaptureAssessment, DocumentExtractionResult, LivenessResult
from kyc_engine.contracts import Image
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore

from ...application.events.lifecycle import WebhookEvent
from ...application.sessions.store import (
    DocumentCaptureTransition,
    FaceMatchResultSnapshot,
    SessionCapacityExceeded,
    SessionConflict,
    SessionExpired,
    SessionNotFound,
    SessionSnapshot,
    SessionStore,
    SessionStoreError,
    _DocumentState,
    _FaceMatchState,
    _LivenessState,
    _NifVerificationState,
    _SessionRecord,
    _accept_document_capture,
    _begin_document_processing,
    _can_finish_document_job,
    _claim_nif_verification,
    _clear_portrait_artifact_id,
    _clear_sensitive_state,
    _complete_document_processing,
    _complete_face_match,
    _complete_liveness,
    _complete_nif_verification,
    _face_match_portrait_artifact_id,
    _fail_document_processing,
    _fail_face_match,
    _fail_liveness,
    _fail_nif_dispatch_capacity,
    _now,
    _queue_document_processing,
    _refresh_expiry,
    _refresh_verification_completion,
    _release_face_match_biometric_artifacts,
    _release_pending_live_face,
    _require_accepted_capture,
    _require_finite_similarity,
    _require_terminal_nif_status,
    _require_trusted_face_failure_code,
    _snapshot,
    _start_face_match,
    _start_liveness,
)
from ...domain.verification import (
    CaptureStatus,
    DocumentSide,
    DocumentStatus,
    FaceMatchStatus,
    LivenessStatus,
    NifVerificationStatus,
    VerificationStatus,
)
from .codec import decode_document_result, encode_document_result
from .models import BrowserCredential, SessionTombstone, VerificationSession, WebhookEventRow


_CAPACITY_LOCK_ID = 0x4B59435F434150  # stable "KYC_CAP" advisory lock key
_CLEANUP_LOCK_ID = 0x4B59435F434C4E  # stable "KYC_CLN" advisory lock key
_TOMBSTONE_LOCK_ID = 0x4B59435F544D42  # stable "KYC_TMB" advisory lock key
_RECOVERY_LOCK_ID = 0x4B59435F524543  # stable "KYC_REC" advisory lock key
_ACTIVE_ARTIFACT_EFFECTS: ContextVar["_ArtifactTransaction | None"] = ContextVar(
    "postgres_artifact_effects", default=None
)

_DOCUMENT_PROCESSING = frozenset({DocumentStatus.QUEUED, DocumentStatus.PROCESSING})
_DOCUMENT_RESULTS = frozenset(
    {DocumentStatus.PASSED, DocumentStatus.PARTIAL, DocumentStatus.FAILED}
)


@dataclass
class _RuntimeState:
    front: bytes | None = None
    back: bytes | None = None
    front_capture: CaptureAssessment | None = None
    back_capture: CaptureAssessment | None = None
    document_result: DocumentExtractionResult | None = None
    liveness_result: LivenessResult | None = None
    live_face_artifact_id: str | None = None
    live_face_eligible: bool = False
    face_similarity: float | None = None


class _ArtifactTransaction:
    """Stage destructive artifact cleanup and compensate pre-commit claims.

    Portrait ownership is intentionally process-local.  Claims are made before
    the SQL commit because they decide whether a transition may succeed; if the
    database transaction aborts, those claims are restored to pending.  Releases
    are harmless to defer and are applied only after the durable commit.
    """

    def __init__(self, store: InMemoryPortraitArtifactStore | None) -> None:
        self._store = store
        self._claims: list[tuple[tuple[str, ...], str]] = []
        self._releases: list[tuple[str, tuple[str, ...], str | None]] = []
        self._after_commit: list[Callable[[], None]] = []

    def claim(self, artifact_ids: tuple[str, ...], session_id: str) -> bool:
        if self._store is None or not self._store.claim(artifact_ids, session_id):
            return False
        self._claims.append((artifact_ids, session_id))
        return True

    def release_pending(self, artifact_ids: tuple[str, ...]) -> None:
        self._releases.append(("pending", artifact_ids, None))

    def release_owned(self, artifact_ids: tuple[str, ...], session_id: str) -> None:
        self._releases.append(("owned", artifact_ids, session_id))

    def after_commit(self, action: Callable[[], None]) -> None:
        self._after_commit.append(action)

    def commit(self) -> None:
        """Run post-commit cleanup without changing an already durable result.

        These releases are idempotent best-effort privacy cleanup.  They are
        deliberately isolated: an unexpected runtime cleanup error cannot make
        a caller observe a failed transaction after PostgreSQL has committed.
        """
        if self._store is not None:
            for kind, artifact_ids, session_id in self._releases:
                try:
                    if kind == "pending":
                        self._store.release_pending(artifact_ids)
                    else:
                        assert session_id is not None
                        self._store.release_owned(artifact_ids, session_id)
                except Exception:
                    # Runtime artifacts are intentionally non-durable.  The
                    # database outcome is authoritative once committed.
                    continue
        for action in self._after_commit:
            try:
                action()
            except Exception:
                continue

    def rollback(self) -> None:
        if self._store is None:
            return
        for artifact_ids, session_id in reversed(self._claims):
            self._store._restore_owned_to_pending(artifact_ids, session_id)


class PostgresSessionStore(SessionStore):
    """Durable store that applies the existing in-memory transition rules under row locks."""

    atomic_webhooks = True

    def __init__(
        self,
        *,
        engine: Engine,
        ttl_seconds: int,
        max_sessions: int,
        portrait_artifacts: InMemoryPortraitArtifactStore | None = None,
        liveness_required: bool = True,
        face_match_enabled: bool = False,
        nif_verification_enabled: bool = False,
        webhook_enabled: bool = False,
        webhook_retention_seconds: int = 24 * 60 * 60,
    ) -> None:
        if ttl_seconds <= 0 or max_sessions <= 0 or webhook_retention_seconds <= 0:
            raise ValueError("PostgreSQL store limits must be positive")
        self._engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_sessions = max_sessions
        self.portrait_artifacts = portrait_artifacts
        self._liveness_required = liveness_required
        self._face_match_enabled = face_match_enabled
        self._nif_verification_enabled = nif_verification_enabled
        self._webhook_enabled = webhook_enabled
        self._webhook_retention = timedelta(seconds=webhook_retention_seconds)
        self._runtime: dict[str, _RuntimeState] = {}
        self._runtime_lock = RLock()

    @property
    def engine(self) -> Engine:
        return self._engine

    def configure_webhooks(self, *, enabled: bool, retention_seconds: int) -> None:
        if retention_seconds <= 0:
            raise ValueError("webhook retention must be positive")
        self._webhook_enabled = enabled
        self._webhook_retention = timedelta(seconds=retention_seconds)

    def configure_requirements(
        self,
        *,
        liveness_required: bool,
        face_match_required: bool,
        nif_verification_enabled: bool = False,
    ) -> None:
        with self._sessions() as database:
            if database.scalar(select(func.count()).select_from(VerificationSession)):
                return
        self._liveness_required = liveness_required
        self._face_match_enabled = face_match_required
        self._nif_verification_enabled = nif_verification_enabled

    def create(self) -> SessionSnapshot:
        now = _now()
        effects = _ArtifactTransaction(self.portrait_artifacts)
        try:
            with self._runtime_lock, self._sessions.begin() as database:
                database.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _CAPACITY_LOCK_ID})
                self._expire_due_locked(database, now, skip_locked=False, effects=effects)
                count = database.scalar(select(func.count()).select_from(VerificationSession)) or 0
                if count >= self._max_sessions:
                    raise SessionCapacityExceeded("Session capacity has been reached")
                record = _SessionRecord(
                    session_id=uuid4().hex,
                    status=VerificationStatus.IN_PROGRESS,
                    created_at=now,
                    expires_at=now + self._ttl,
                    document=_DocumentState(),
                    liveness=_LivenessState(),
                    face_match=_FaceMatchState(),
                    nif_verification=_NifVerificationState(),
                    liveness_required=self._liveness_required,
                    face_match_required=self._face_match_enabled,
                    nif_verification_enabled=self._nif_verification_enabled,
                )
                row = VerificationSession(session_id=record.session_id)
                database.add(row)
                self._write_record(row, record, version=1)
                snapshot = _snapshot(record)
                self._insert_events(database, snapshot, ("verification.session.created",))
        except Exception:
            effects.rollback()
            raise
        effects.commit()
        with self._runtime_lock:
            self._runtime[record.session_id] = _RuntimeState()
        return snapshot

    def get(self, session_id: str) -> SessionSnapshot:
        record = self._read_record(session_id, expire=True)
        return _snapshot(record)

    def accept_document_capture(self, session_id: str, side: DocumentSide, content: bytes, assessment: CaptureAssessment) -> DocumentCaptureTransition:
        _require_accepted_capture(assessment)
        return self._mutate(
            session_id,
            lambda record: _accept_document_capture(record, side, content, assessment, self.portrait_artifacts, self._ttl),
            expire=True,
            reason_kind="accept_document_capture",
        )

    def queue_document_processing(self, session_id: str) -> SessionSnapshot:
        def _queue(record: _SessionRecord) -> SessionSnapshot:
            _queue_document_processing(record, self._ttl)
            return _snapshot(record)
        return self._mutate(
            session_id,
            _queue,
            expire=True,
            reason_kind="queue_document_processing",
        )

    def start_document_processing(self, session_id: str, job_id: str) -> tuple[bytes, bytes, SessionSnapshot]:
        def _begin(record: _SessionRecord) -> tuple[bytes, bytes, SessionSnapshot]:
            front, back = _begin_document_processing(record, job_id, self._ttl)
            return front, back, _snapshot(record)
        return self._mutate(
            session_id,
            _begin,
            expire=True,
            reason_kind="start_document_processing",
        )

    def complete_document_processing(self, session_id: str, job_id: str, result: DocumentExtractionResult) -> SessionSnapshot | None:
        def _complete(record: _SessionRecord) -> SessionSnapshot | None:
            if not _can_finish_document_job(record, job_id):
                return None
            _complete_document_processing(record, result, self._active_artifacts(), self._ttl)
            return _snapshot(record)
        return self._mutate(session_id, _complete, missing_none=True, reason_kind="complete_document_processing")

    def fail_document_processing(self, session_id: str, job_id: str, code: str) -> SessionSnapshot | None:
        def _fail(record: _SessionRecord) -> SessionSnapshot | None:
            if record.document.job_id != job_id or record.document.status not in _DOCUMENT_PROCESSING:
                return None
            _fail_document_processing(record, code, self._ttl)
            return _snapshot(record)
        return self._mutate(session_id, _fail, missing_none=True, reason_kind="fail_document_processing")

    def timeout_document_processing(self, session_id: str, job_id: str) -> SessionSnapshot | None:
        """Fail an overdue running job and invalidate any later completion."""
        def _timeout(record: _SessionRecord) -> SessionSnapshot | None:
            if not _can_finish_document_job(record, job_id):
                return None
            _fail_document_processing(record, "JOB_TIMEOUT", self._ttl)
            return _snapshot(record)
        return self._mutate(session_id, _timeout, missing_none=True, reason_kind="timeout_document_processing")

    def start_liveness(self, session_id: str) -> SessionSnapshot:
        def _start(record: _SessionRecord) -> SessionSnapshot:
            _start_liveness(record, self._ttl)
            return _snapshot(record)
        return self._mutate(
            session_id,
            _start,
            expire=True,
            reason_kind="start_liveness",
        )

    def complete_liveness(self, session_id: str, result: LivenessResult, live_face_artifact_id: str | None = None, live_face_eligible: bool = False) -> SessionSnapshot | None:
        def _complete(record: _SessionRecord) -> SessionSnapshot | None:
            if record.status != VerificationStatus.IN_PROGRESS or record.liveness.status != LivenessStatus.PROCESSING:
                return None
            _complete_liveness(record, result, live_face_artifact_id, live_face_eligible, self._active_artifacts(), self._ttl)
            return _snapshot(record)

        def _release_pending() -> None:
            _release_pending_live_face(live_face_artifact_id, self.portrait_artifacts)
        return self._mutate(
            session_id, _complete, missing_none=True, reason_kind="complete_liveness", on_none=_release_pending
        )

    def claim_nif_verification(self, session_id: str) -> tuple[str, str | None, SessionSnapshot] | None:
        def _claim(record: _SessionRecord) -> tuple[str, str | None, SessionSnapshot] | None:
            nif_state = record.nif_verification
            if nif_state.attempted or nif_state.status == NifVerificationStatus.PROCESSING:
                return None
            nif, claimed_name = _claim_nif_verification(record, self._ttl)
            # A tuple with an empty NIF tells orchestration that this was skipped.
            return nif, claimed_name, _snapshot(record)
        return self._mutate(session_id, _claim, missing_none=True, reason_kind="claim_nif_verification")

    def fail_nif_dispatch_capacity(self, session_id: str) -> SessionSnapshot | None:
        def _fail(record: _SessionRecord) -> SessionSnapshot | None:
            if record.status != VerificationStatus.IN_PROGRESS or record.nif_verification.status != NifVerificationStatus.PROCESSING:
                return None
            _fail_nif_dispatch_capacity(record, self._ttl)
            return _snapshot(record)
        return self._mutate(session_id, _fail, missing_none=True, reason_kind="fail_nif_dispatch_capacity")

    def complete_nif_verification(self, session_id: str, *, status: NifVerificationStatus, source: str | None, name_match: bool | None, error_code: str | None = None) -> SessionSnapshot | None:
        _require_terminal_nif_status(status)

        def _complete(record: _SessionRecord) -> SessionSnapshot | None:
            if record.status != VerificationStatus.IN_PROGRESS or record.nif_verification.status != NifVerificationStatus.PROCESSING:
                return None
            _complete_nif_verification(record, status, source, name_match, error_code, self._ttl)
            return _snapshot(record)
        return self._mutate(session_id, _complete, missing_none=True, reason_kind="complete_nif_verification")

    def start_face_match(self, session_id: str) -> SessionSnapshot:
        def _start(record: _SessionRecord) -> SessionSnapshot:
            _start_face_match(record, self._ttl)
            return _snapshot(record)
        return self._mutate(
            session_id,
            _start,
            expire=True,
            reason_kind="start_face_match",
        )

    def complete_face_match(self, session_id: str, similarity: float) -> SessionSnapshot | None:
        _require_finite_similarity(similarity)

        def _complete(record: _SessionRecord) -> SessionSnapshot | None:
            if record.status != VerificationStatus.IN_PROGRESS or record.face_match.status != FaceMatchStatus.PROCESSING:
                return None
            _complete_face_match(record, similarity, self._ttl)
            return _snapshot(record)
        return self._mutate(session_id, _complete, missing_none=True, reason_kind="complete_face_match")

    def fail_face_match(self, session_id: str, code: str) -> SessionSnapshot | None:
        _require_trusted_face_failure_code(code)

        def _fail(record: _SessionRecord) -> SessionSnapshot | None:
            if record.status != VerificationStatus.IN_PROGRESS or record.face_match.status != FaceMatchStatus.PROCESSING:
                return None
            _fail_face_match(record, code, self._ttl)
            return _snapshot(record)
        return self._mutate(session_id, _fail, missing_none=True, reason_kind="fail_face_match")

    def refresh_verification_completion(self, session_id: str) -> SessionSnapshot | None:
        def _complete(record: _SessionRecord) -> SessionSnapshot | None:
            if not _refresh_verification_completion(record):
                return None
            record.event_sequence += 1
            _refresh_expiry(record, self._ttl)
            return _snapshot(record)
        return self._mutate(session_id, _complete, missing_none=True, reason_kind="refresh_verification_completion")

    def face_match_result(self, session_id: str) -> FaceMatchResultSnapshot:
        record = self._read_record(session_id, expire=True)
        return FaceMatchResultSnapshot(
            status=record.face_match.status,
            similarity=record.face_match.similarity if record.face_match.status == FaceMatchStatus.COMPLETED else None,
        )

    def release_face_match_biometric_artifacts(self, session_id: str) -> None:
        def _release(record: _SessionRecord) -> None:
            if record.face_match.status not in {
                FaceMatchStatus.COMPLETED,
                FaceMatchStatus.FAILED,
            }:
                raise SessionConflict("Face-match artifacts cannot be released while active")
            _release_face_match_biometric_artifacts(record, self._active_artifacts())
        self._mutate(session_id, _release, expire=True)

    def fail_liveness(self, session_id: str, code: str) -> SessionSnapshot | None:
        def _fail(record: _SessionRecord) -> SessionSnapshot | None:
            if record.status != VerificationStatus.IN_PROGRESS or record.liveness.status != LivenessStatus.PROCESSING:
                return None
            _fail_liveness(record, code, self._active_artifacts(), self._ttl)
            return _snapshot(record)
        return self._mutate(session_id, _fail, missing_none=True, reason_kind="fail_liveness")

    def result(self, session_id: str) -> DocumentExtractionResult:
        record = self._read_record(session_id, expire=True)
        document = record.document
        if document.status not in _DOCUMENT_RESULTS:
            raise SessionConflict("The document extraction result is not ready")
        if document.result is None:
            raise SessionStoreError(document.error_code or "JOB_FAILED")
        return document.result

    def delete(self, session_id: str) -> None:
        effects = _ArtifactTransaction(self.portrait_artifacts)
        try:
            with self._runtime_lock, self._sessions.begin() as database:
                database.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:value, 0))"), {"value": session_id})
                row = database.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id).with_for_update())
                if row is not None:
                    record = self._record_from_row(row)
                    _clear_sensitive_state(record, effects)
                    database.delete(row)
                database.execute(delete(SessionTombstone).where(SessionTombstone.session_id == session_id))
                database.execute(delete(BrowserCredential).where(BrowserCredential.session_id == session_id))
        except Exception:
            effects.rollback()
            raise
        effects.commit()
        with self._runtime_lock:
            self._runtime.pop(session_id, None)

    def resolve_portrait_artifact(self, session_id: str):
        record = self._read_record(session_id, expire=True)
        result = record.document.result
        portrait = result.front.portrait if result is not None and result.front is not None else None
        if portrait is None or portrait.artifact_id is None or self.portrait_artifacts is None:
            return None
        return self.portrait_artifacts.get(portrait.artifact_id, session_id=session_id)

    def resolve_face_match_portrait(self, session_id: str) -> Image | None:
        """Return the document portrait only when it is matcher-eligible."""
        record = self._read_record(session_id, expire=True)
        result = record.document.result
        portrait = result.front.portrait if result is not None and result.front is not None else None
        if (
            record.status != VerificationStatus.IN_PROGRESS
            or portrait is None
            or not portrait.eligible_for_face_match
            or portrait.artifact_id is None
            or self.portrait_artifacts is None
        ):
            return None
        return self.portrait_artifacts.get(portrait.artifact_id, session_id=session_id)

    def resolve_live_face_artifact(self, session_id: str) -> Image | None:
        """Internal-only future matcher boundary for the retained live face."""
        record = self._read_record(session_id, expire=True)
        liveness = record.liveness
        if (
            liveness.status != LivenessStatus.PASSED
            or liveness.result is None
            or not liveness.result.passed
            or not liveness.live_face_eligible
            or liveness.live_face_artifact_id is None
            or self.portrait_artifacts is None
        ):
            return None
        return self.portrait_artifacts.get(
            liveness.live_face_artifact_id, session_id=session_id
        )

    def contains_images(self, session_id: str) -> bool:
        with self._runtime_lock:
            runtime = self._runtime.get(session_id)
            return bool(runtime and (runtime.front is not None or runtime.back is not None))

    def cleanup(self) -> int:
        now = _now()
        effects = _ArtifactTransaction(self.portrait_artifacts)
        try:
            with self._runtime_lock, self._sessions.begin() as database:
                database.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _CLEANUP_LOCK_ID})
                database.execute(delete(SessionTombstone).where(SessionTombstone.retained_until <= now))
                count = self._expire_due_locked(database, now, skip_locked=True, effects=effects)
        except Exception:
            effects.rollback()
            raise
        effects.commit()
        return count

    def expiry_tombstone_deadline(self, session_id: str) -> datetime | None:
        now = _now()
        effects = _ArtifactTransaction(self.portrait_artifacts)
        try:
            with self._runtime_lock, self._sessions.begin() as database:
                database.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _TOMBSTONE_LOCK_ID})
                database.execute(delete(SessionTombstone).where(SessionTombstone.retained_until <= now))
                row = database.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id).with_for_update())
                if row is not None and row.expires_at <= now:
                    self._expire_row(database, row, now, effects=effects)
                deadline = database.scalar(select(SessionTombstone.retained_until).where(SessionTombstone.session_id == session_id))
        except Exception:
            effects.rollback()
            raise
        effects.commit()
        return deadline

    def recover(self) -> int:
        """Conservatively settle states that depended on process-local work."""
        recovered = 0
        effects = _ArtifactTransaction(self.portrait_artifacts)
        try:
            with self._runtime_lock, self._sessions.begin() as database:
                database.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _RECOVERY_LOCK_ID})
                self._expire_due_locked(database, _now(), skip_locked=False, effects=effects)
                rows = list(database.scalars(select(VerificationSession).where(VerificationSession.verification_status == VerificationStatus.IN_PROGRESS.value).with_for_update()))
                for row in rows:
                    record = self._record_from_row(row, include_runtime=False)
                    changed = False
                    if record.nif_verification.status == NifVerificationStatus.PROCESSING:
                        state = record.nif_verification
                        state.status = NifVerificationStatus.UNAVAILABLE
                        state.error_code = "NIF_VERIFICATION_INTERRUPTED"
                        state.settled = True
                        _refresh_verification_completion(record)
                        record.event_sequence += 1
                        nif_reasons = ["nif.completed"]
                        if record.status == VerificationStatus.COMPLETED:
                            nif_reasons.append("verification.completed")
                        elif record.status == VerificationStatus.FAILED:
                            nif_reasons.append("verification.failed")
                        self._insert_events(database, _snapshot(record), tuple(nif_reasons))
                        changed = True
                    document_interrupted = (
                        row.document_front_capture_status == CaptureStatus.ACCEPTED.value
                        or row.document_back_capture_status == CaptureStatus.ACCEPTED.value
                        or record.document.status in {DocumentStatus.READY, DocumentStatus.QUEUED, DocumentStatus.PROCESSING}
                    ) and record.document.status not in {DocumentStatus.PASSED, DocumentStatus.PARTIAL, DocumentStatus.FAILED}
                    face_inputs_lost = (
                        record.face_match_required
                        and record.document.status == DocumentStatus.PASSED
                        and record.face_match.status != FaceMatchStatus.COMPLETED
                    )
                    if document_interrupted:
                        record.document.status = DocumentStatus.FAILED
                        record.document.error_code = "PROCESS_INTERRUPTED_DOCUMENT"
                        record.status = VerificationStatus.FAILED
                        record.event_sequence += 1
                        self._insert_events(database, _snapshot(record), ("document.failed", "verification.failed"))
                        changed = True
                    elif record.liveness.status == LivenessStatus.PROCESSING:
                        record.liveness.status = LivenessStatus.FAILED
                        record.liveness.error_code = "PROCESS_INTERRUPTED_LIVENESS"
                        record.status = VerificationStatus.FAILED
                        record.event_sequence += 1
                        self._insert_events(database, _snapshot(record), ("liveness.failed", "verification.failed"))
                        changed = True
                    elif face_inputs_lost or (
                        record.face_match_required
                        and record.liveness.status == LivenessStatus.PASSED
                        and record.face_match.status != FaceMatchStatus.COMPLETED
                    ):
                        record.face_match.status = FaceMatchStatus.FAILED
                        record.face_match.error_code = "PROCESS_INTERRUPTED_FACE_MATCH"
                        record.status = VerificationStatus.FAILED
                        record.event_sequence += 1
                        self._insert_events(database, _snapshot(record), ("verification.failed",))
                        changed = True
                    if changed:
                        record.expires_at = _now() + self._ttl
                        self._write_record(row, record, version=row.version + 1)
                        recovered += 1
        except Exception:
            effects.rollback()
            raise
        effects.commit()
        return recovered

    def _mutate(
        self,
        session_id: str,
        transition: Callable[[_SessionRecord], Any],
        *,
        expire: bool = False,
        missing_none: bool = False,
        reason_kind: str | None = None,
        on_none: Callable[[], None] | None = None,
    ) -> Any:
        effects = _ArtifactTransaction(self.portrait_artifacts)
        token = _ACTIVE_ARTIFACT_EFFECTS.set(effects)
        try:
            return self._mutate_with_effects(
                session_id, transition, expire=expire, missing_none=missing_none,
                reason_kind=reason_kind, on_none=on_none, effects=effects,
            )
        except Exception:
            effects.rollback()
            raise
        finally:
            _ACTIVE_ARTIFACT_EFFECTS.reset(token)

    def _mutate_with_effects(
        self,
        session_id: str,
        transition: Callable[[_SessionRecord], Any],
        *,
        expire: bool,
        missing_none: bool,
        reason_kind: str | None,
        on_none: Callable[[], None] | None,
        effects: _ArtifactTransaction,
    ) -> Any:
        with self._runtime_lock:
            expired = False
            record = None
            result = None
            with self._sessions.begin() as database:
                row = database.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id).with_for_update())
                if row is None:
                    if on_none is not None:
                        on_none()
                    if missing_none:
                        return None
                    self._raise_missing(database, session_id)
                assert row is not None
                if expire and row.expires_at <= _now():
                    self._expire_row(database, row, _now(), effects=effects)
                    expired = True
                else:
                    record = self._record_from_row(row)
                    result = transition(record)
                    if result is not None:
                        self._write_record(row, record, version=row.version + 1)
                        self._insert_events(
                            database,
                            self._event_snapshot(result),
                            self._event_reasons(reason_kind, result),
                        )
                    elif on_none is not None:
                        on_none()
            if expired:
                raise SessionExpired("Session has expired")
            assert record is not None
            effects.commit()
            self._store_runtime(record)
            return result

    def _active_artifacts(self) -> _ArtifactTransaction | InMemoryPortraitArtifactStore | None:
        """Use transaction-scoped artifact effects while a PostgreSQL mutation runs."""
        return _ACTIVE_ARTIFACT_EFFECTS.get() or self.portrait_artifacts

    def _read_record(self, session_id: str, *, expire: bool) -> _SessionRecord:
        expired = False
        record = None
        effects = _ArtifactTransaction(self.portrait_artifacts)
        try:
            with self._runtime_lock, self._sessions.begin() as database:
                row = database.scalar(select(VerificationSession).where(VerificationSession.session_id == session_id).with_for_update())
                if row is None:
                    self._raise_missing(database, session_id)
                assert row is not None
                if expire and row.expires_at <= _now():
                    self._expire_row(database, row, _now(), effects=effects)
                    expired = True
                else:
                    record = self._record_from_row(row)
        except Exception:
            effects.rollback()
            raise
        effects.commit()
        if expired:
            raise SessionExpired("Session has expired")
        assert record is not None
        return record

    def _record_from_row(self, row: VerificationSession, *, include_runtime: bool = True) -> _SessionRecord:
        runtime = self._runtime.get(row.session_id) if include_runtime else None
        persisted_result = decode_document_result(row.document_result_json) if row.document_result_json is not None else None
        return _SessionRecord(
            session_id=row.session_id,
            status=VerificationStatus(row.verification_status),
            created_at=row.created_at,
            expires_at=row.expires_at,
            document=_DocumentState(
                status=DocumentStatus(row.document_status),
                front=runtime.front if runtime else None,
                back=runtime.back if runtime else None,
                # Acceptance is durable lifecycle state; the assessment object
                # itself is only ever process-local runtime state.
                front_capture_accepted=row.document_front_capture_status == CaptureStatus.ACCEPTED.value,
                back_capture_accepted=row.document_back_capture_status == CaptureStatus.ACCEPTED.value,
                front_capture=runtime.front_capture if runtime else None,
                back_capture=runtime.back_capture if runtime else None,
                job_id=row.document_job_id,
                result=runtime.document_result if runtime and runtime.document_result is not None else persisted_result,
                error_code=row.document_error_code,
            ),
            liveness=_LivenessState(
                status=LivenessStatus(row.liveness_status),
                result=runtime.liveness_result if runtime else None,
                error_code=row.liveness_error_code,
                live_face_artifact_id=runtime.live_face_artifact_id if runtime else None,
                live_face_eligible=runtime.live_face_eligible if runtime else False,
            ),
            face_match=_FaceMatchState(
                status=FaceMatchStatus(row.face_match_status),
                similarity=runtime.face_similarity if runtime else None,
                error_code=row.face_match_error_code,
            ),
            nif_verification=_NifVerificationState(
                status=NifVerificationStatus(row.nif_status),
                source=row.nif_source,
                name_match=row.nif_name_match,
                not_run_reason=row.nif_not_run_reason,
                error_code=row.nif_error_code,
                settled=row.nif_settled,
                attempted=row.nif_attempted,
            ),
            liveness_required=row.liveness_required,
            face_match_required=row.face_match_required,
            nif_verification_enabled=row.nif_verification_enabled,
            event_sequence=row.event_sequence,
        )

    def _write_record(self, row: VerificationSession, record: _SessionRecord, *, version: int) -> None:
        row.verification_status = record.status.value
        row.created_at = record.created_at
        row.updated_at = _now()
        row.expires_at = record.expires_at
        row.event_sequence = record.event_sequence
        row.version = version
        row.liveness_required = record.liveness_required
        row.face_match_required = record.face_match_required
        row.nif_verification_enabled = record.nif_verification_enabled
        row.document_status = record.document.status.value
        row.document_front_capture_status = CaptureStatus.ACCEPTED.value if record.document.front_capture_accepted else CaptureStatus.MISSING.value
        row.document_back_capture_status = CaptureStatus.ACCEPTED.value if record.document.back_capture_accepted else CaptureStatus.MISSING.value
        row.document_job_id = record.document.job_id
        row.document_result_json = encode_document_result(record.document.result) if record.document.result is not None else None
        row.document_error_code = record.document.error_code
        row.liveness_status = record.liveness.status.value
        row.liveness_error_code = record.liveness.error_code
        row.face_match_status = record.face_match.status.value
        row.face_match_error_code = record.face_match.error_code
        row.nif_status = record.nif_verification.status.value
        row.nif_source = record.nif_verification.source
        row.nif_name_match = record.nif_verification.name_match
        row.nif_not_run_reason = record.nif_verification.not_run_reason
        row.nif_error_code = record.nif_verification.error_code
        row.nif_settled = record.nif_verification.settled
        row.nif_attempted = record.nif_verification.attempted

    def _store_runtime(self, record: _SessionRecord) -> None:
        self._runtime[record.session_id] = _RuntimeState(
            front=record.document.front,
            back=record.document.back,
            front_capture=record.document.front_capture,
            back_capture=record.document.back_capture,
            document_result=record.document.result,
            liveness_result=record.liveness.result,
            live_face_artifact_id=record.liveness.live_face_artifact_id,
            live_face_eligible=record.liveness.live_face_eligible,
            face_similarity=record.face_match.similarity,
        )

    def _raise_missing(self, database: Session, session_id: str) -> None:
        retained = database.scalar(select(SessionTombstone.retained_until).where(SessionTombstone.session_id == session_id))
        if retained is not None and retained > _now():
            raise SessionExpired("Session has expired")
        raise SessionNotFound("Session was not found")

    def _expire_due_locked(
        self, database: Session, now: datetime, *, skip_locked: bool,
        effects: _ArtifactTransaction | None = None,
    ) -> int:
        rows = list(database.scalars(select(VerificationSession).where(VerificationSession.expires_at <= now).with_for_update(skip_locked=skip_locked)))
        for row in rows:
            self._expire_row(database, row, now, effects=effects)
        database.execute(delete(SessionTombstone).where(SessionTombstone.retained_until <= now))
        return len(rows)

    def _expire_row(
        self, database: Session, row: VerificationSession, now: datetime,
        *, effects: _ArtifactTransaction | None = None,
    ) -> None:
        runtime = self._runtime.get(row.session_id)
        if runtime is not None:
            record = self._record_from_row(row)
            _clear_sensitive_state(record, effects or self.portrait_artifacts)
            if effects is None:
                self._runtime.pop(row.session_id, None)
            else:
                effects.after_commit(
                    lambda session_id=row.session_id: self._runtime.pop(session_id, None)
                )
        database.execute(
            insert(SessionTombstone)
            .values(session_id=row.session_id, retained_until=now + self._ttl)
            .on_conflict_do_update(index_elements=[SessionTombstone.session_id], set_={"retained_until": now + self._ttl})
        )
        database.delete(row)

    def _insert_events(self, database: Session, snapshot: SessionSnapshot | None, reasons: tuple[str, ...]) -> None:
        if not self._webhook_enabled or snapshot is None:
            return
        now = _now()
        for reason in reasons:
            event = WebhookEvent.from_snapshot(snapshot, transition_reason=reason)
            if event is None:
                continue
            database.execute(
                insert(WebhookEventRow)
                .values(
                    event_id=event.event_id,
                    session_id=event.session_id,
                    sequence=event.sequence,
                    event_type=event.event_type,
                    payload=event.payload(),
                    created_at=now,
                    expires_at=now + self._webhook_retention,
                    attempts=0,
                    next_attempt_at=now,
                )
                .on_conflict_do_nothing(constraint="uq_webhook_session_sequence_type")
            )

    @staticmethod
    def _event_snapshot(result: Any) -> SessionSnapshot | None:
        if isinstance(result, SessionSnapshot):
            return result
        if isinstance(result, DocumentCaptureTransition):
            return result.snapshot
        if isinstance(result, tuple) and result and isinstance(result[-1], SessionSnapshot):
            return result[-1]
        return None

    @staticmethod
    def _event_reasons(kind: str | None, result: Any) -> tuple[str, ...]:
        if kind is None:
            return ()
        snapshot = PostgresSessionStore._event_snapshot(result)
        if snapshot is None:
            return ()
        terminal = (
            ("verification.completed",) if snapshot.verification_status == VerificationStatus.COMPLETED
            else (("verification.failed",) if snapshot.verification_status == VerificationStatus.FAILED else ())
        )
        if kind == "complete_document_processing":
            reason = {
                DocumentStatus.PASSED: "document.passed",
                DocumentStatus.PARTIAL: "document.partial",
                DocumentStatus.FAILED: "document.failed",
            }.get(snapshot.document.status)
            return ((reason,) if reason else ()) + terminal
        if kind in {"fail_document_processing", "timeout_document_processing"}:
            return ("document.failed",) + terminal
        if kind == "complete_liveness":
            return (("liveness.passed",) if snapshot.liveness_status == LivenessStatus.PASSED else ("liveness.failed",)) + terminal
        if kind == "fail_liveness":
            return ("liveness.failed",) + terminal
        if kind == "complete_nif_verification":
            return ("nif.completed",) + terminal
        if kind in {"claim_nif_verification", "fail_nif_dispatch_capacity", "complete_face_match", "fail_face_match", "refresh_verification_completion"}:
            return terminal
        return ()
