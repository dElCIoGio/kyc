from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from math import isfinite
from threading import RLock
from uuid import uuid4

from kyc_engine import (
    CaptureAssessment,
    DocumentExtractionResult,
    LivenessResult,
    ProcessingStatus,
)
from kyc_engine.contracts import Image
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore

from .models import (
    CaptureStatus,
    DocumentSide,
    DocumentStatus,
    FaceMatchStatus,
    LivenessStatus,
    NifVerificationStatus,
    VerificationStatus,
)


class SessionStoreError(RuntimeError):
    code = "SESSION_ERROR"


class SessionNotFound(SessionStoreError):
    code = "SESSION_NOT_FOUND"


class SessionExpired(SessionStoreError):
    code = "SESSION_EXPIRED"


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
    nif_verification_status: NifVerificationStatus = NifVerificationStatus.NOT_RUN
    nif_verification_source: str | None = None
    nif_name_match: bool | None = None
    liveness_required: bool = True
    face_match_required: bool = False
    event_sequence: int = 0


@dataclass(frozen=True)
class FaceMatchResultSnapshot:
    """Safe result metadata for the authenticated result representation."""

    status: FaceMatchStatus
    similarity: float | None


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
    result: LivenessResult | None = None
    error_code: str | None = None
    live_face_artifact_id: str | None = None
    live_face_eligible: bool = False


@dataclass
class _FaceMatchState:
    status: FaceMatchStatus = FaceMatchStatus.BLOCKED
    similarity: float | None = None
    error_code: str | None = None


@dataclass
class _NifVerificationState:
    status: NifVerificationStatus = NifVerificationStatus.NOT_RUN
    source: str | None = None
    name_match: bool | None = None
    # Reasons are deliberately internal: they explain skipped work without
    # leaking document data into the public session contract.
    not_run_reason: str = "disabled"
    error_code: str | None = None
    settled: bool = True
    attempted: bool = False


@dataclass
class _SessionRecord:
    session_id: str
    status: VerificationStatus
    created_at: datetime
    expires_at: datetime
    document: _DocumentState
    liveness: _LivenessState
    face_match: _FaceMatchState
    nif_verification: _NifVerificationState
    liveness_required: bool
    face_match_required: bool
    nif_verification_enabled: bool
    event_sequence: int = 0


class SessionStore:
    """Thread-safe, memory-only storage for complete verification sessions."""

    _DOCUMENT_PROCESSING = {DocumentStatus.QUEUED, DocumentStatus.PROCESSING}
    _DOCUMENT_RESULTS = {DocumentStatus.PASSED, DocumentStatus.PARTIAL, DocumentStatus.FAILED}
    _VERIFICATION_TERMINAL = {
        VerificationStatus.COMPLETED,
        VerificationStatus.FAILED,
        VerificationStatus.VERIFIED,
        VerificationStatus.REJECTED,
        VerificationStatus.EXPIRED,
    }

    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_sessions: int,
        portrait_artifacts: InMemoryPortraitArtifactStore | None = None,
        liveness_required: bool = True,
        face_match_enabled: bool = False,
        nif_verification_enabled: bool = False,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_sessions = max_sessions
        self._records: dict[str, _SessionRecord] = {}
        # Only opaque identifiers and their purge deadlines are retained after
        # expiry so callers receive 410 rather than an ambiguous 404.
        self._expired: dict[str, datetime] = {}
        self._lock = RLock()
        self.portrait_artifacts = portrait_artifacts
        self._liveness_required = liveness_required
        self._face_match_enabled = face_match_enabled
        self._nif_verification_enabled = nif_verification_enabled

    def configure_requirements(
        self, *, liveness_required: bool, face_match_required: bool,
        nif_verification_enabled: bool = False,
    ) -> None:
        """Bind process configuration before the store serves any sessions."""
        with self._lock:
            if self._records:
                # Test and composition callers may provide an already-populated
                # store. Preserve the requirements bound to those sessions.
                return
            self._liveness_required = liveness_required
            self._face_match_enabled = face_match_required
            self._nif_verification_enabled = nif_verification_enabled

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
                nif_verification=_NifVerificationState(),
                liveness_required=self._liveness_required,
                face_match_required=self._face_match_enabled,
                nif_verification_enabled=self._nif_verification_enabled,
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
            artifact_ids = _portrait_artifact_ids(result)
            if artifact_ids and result.status == ProcessingStatus.FAILED:
                if self.portrait_artifacts is not None:
                    self.portrait_artifacts.release_pending(artifact_ids)
                artifact_ids = ()
            if artifact_ids and (
                self.portrait_artifacts is None
                or not self.portrait_artifacts.claim(artifact_ids, session_id)
            ):
                if self.portrait_artifacts is not None:
                    self.portrait_artifacts.release_pending(artifact_ids)
                _clear_document_sources(document)
                document.result = None
                document.error_code = "PORTRAIT_ARTIFACT_CLAIM_FAILED"
                document.status = DocumentStatus.FAILED
                self._refresh_verification_failure(record)
                record.event_sequence += 1
                self._refresh_expiry(record)
                return _snapshot(record)
            _clear_document_sources(document)
            document.result = result
            document.error_code = None
            document.status = {
                ProcessingStatus.SUCCESS: DocumentStatus.PASSED,
                ProcessingStatus.PARTIAL: DocumentStatus.PARTIAL,
                ProcessingStatus.FAILED: DocumentStatus.FAILED,
            }[result.status]
            # Defer aggregate completion until the post-extraction NIF
            # orchestrator has either claimed applicable work or explicitly
            # settled a safe skipped state.
            if record.nif_verification_enabled and result.status == ProcessingStatus.SUCCESS:
                record.nif_verification.settled = False
                record.nif_verification.not_run_reason = ""
            self._refresh_face_match_readiness(record)
            self._refresh_verification_completion(record)
            self._refresh_verification_failure(record)
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
            self._refresh_verification_failure(record)
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
            self._refresh_verification_failure(record)
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def start_liveness(self, session_id: str) -> SessionSnapshot:
        with self._lock:
            record = self._get_record(session_id)
            if record.status != VerificationStatus.IN_PROGRESS:
                raise SessionConflict("Verification is no longer active")
            if record.liveness.status != LivenessStatus.READY:
                raise SessionConflict("Liveness is not ready to start")
            record.liveness.status = LivenessStatus.PROCESSING
            record.liveness.error_code = None
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def complete_liveness(
        self,
        session_id: str,
        result: LivenessResult,
        live_face_artifact_id: str | None = None,
        live_face_eligible: bool = False,
    ) -> SessionSnapshot | None:
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.status != VerificationStatus.IN_PROGRESS
                or record.liveness.status != LivenessStatus.PROCESSING
            ):
                self._release_pending_live_face(live_face_artifact_id)
                return None
            claimed_artifact_id = None
            if live_face_artifact_id is not None and result.passed and live_face_eligible:
                if self.portrait_artifacts is not None and self.portrait_artifacts.claim(
                    (live_face_artifact_id,), session_id
                ):
                    claimed_artifact_id = live_face_artifact_id
                else:
                    self._release_pending_live_face(live_face_artifact_id)
            elif live_face_artifact_id is not None:
                self._release_pending_live_face(live_face_artifact_id)
            record.liveness.result = result
            record.liveness.live_face_artifact_id = claimed_artifact_id
            record.liveness.live_face_eligible = claimed_artifact_id is not None
            record.liveness.status = (
                LivenessStatus.PASSED if result.passed else LivenessStatus.FAILED
            )
            record.liveness.error_code = (
                None if result.passed else "PASSIVE_LIVENESS_FAILED"
            )
            self._refresh_face_match_readiness(record)
            self._refresh_verification_completion(record)
            self._refresh_verification_failure(record)
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def claim_nif_verification(
        self, session_id: str
    ) -> tuple[str, str | None, SessionSnapshot] | None:
        """Atomically claim applicable registry work after document extraction.

        The extracted identifier and name are used only as short-lived local
        variables for dispatch; neither is copied into NIF session state.
        """
        with self._lock:
            record = self._records.get(session_id)
            if record is None or record.status != VerificationStatus.IN_PROGRESS:
                return None
            nif_state = record.nif_verification
            if nif_state.attempted or nif_state.status == NifVerificationStatus.PROCESSING:
                return None
            candidate = _nif_candidate(record.document.result)
            if not record.nif_verification_enabled:
                return self._settle_nif_not_run(record, "disabled")
            if candidate is None:
                reason = (
                    "identifier_unavailable"
                    if _nif_profile_present(record.document.result)
                    else "not_applicable"
                )
                return self._settle_nif_not_run(record, reason)

            nif, claimed_name = candidate
            nif_state.status = NifVerificationStatus.PROCESSING
            nif_state.source = "minfin"
            nif_state.name_match = None
            nif_state.not_run_reason = ""
            nif_state.error_code = None
            nif_state.settled = False
            nif_state.attempted = True
            record.event_sequence += 1
            self._refresh_expiry(record)
            return nif, claimed_name, _snapshot(record)

    def fail_nif_dispatch_capacity(self, session_id: str) -> SessionSnapshot | None:
        """Settle a claimed but unstarted attempt without a provider source."""
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.status != VerificationStatus.IN_PROGRESS
                or record.nif_verification.status != NifVerificationStatus.PROCESSING
            ):
                return None
            state = record.nif_verification
            state.status = NifVerificationStatus.FAILED
            state.source = None
            state.name_match = None
            state.error_code = "NIF_VERIFIER_CAPACITY_EXCEEDED"
            state.settled = True
            self._refresh_verification_completion(record)
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def complete_nif_verification(
        self,
        session_id: str,
        *,
        status: NifVerificationStatus,
        source: str | None,
        name_match: bool | None,
        error_code: str | None = None,
    ) -> SessionSnapshot | None:
        if status not in {
            NifVerificationStatus.VERIFIED,
            NifVerificationStatus.NOT_FOUND,
            NifVerificationStatus.UNAVAILABLE,
            NifVerificationStatus.FAILED,
        }:
            raise ValueError("NIF completion status is not terminal")
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.status != VerificationStatus.IN_PROGRESS
                or record.nif_verification.status != NifVerificationStatus.PROCESSING
            ):
                return None
            state = record.nif_verification
            state.status = status
            state.source = source
            state.name_match = name_match if status == NifVerificationStatus.VERIFIED else None
            state.error_code = error_code
            state.settled = True
            if status == NifVerificationStatus.NOT_FOUND:
                record.status = VerificationStatus.FAILED
            else:
                self._refresh_verification_completion(record)
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def _settle_nif_not_run(
        self, record: _SessionRecord, reason: str
    ) -> tuple[str, str | None, SessionSnapshot] | None:
        state = record.nif_verification
        state.status = NifVerificationStatus.NOT_RUN
        state.source = None
        state.name_match = None
        state.not_run_reason = reason
        state.error_code = None
        state.settled = True
        self._refresh_verification_completion(record)
        record.event_sequence += 1
        self._refresh_expiry(record)
        # A tuple with an empty NIF tells orchestration that this was skipped.
        return "", None, _snapshot(record)

    def start_face_match(self, session_id: str) -> SessionSnapshot:
        """Atomically claim a ready face match before model inference begins."""
        with self._lock:
            record = self._get_record(session_id)
            if record.status != VerificationStatus.IN_PROGRESS:
                raise SessionConflict("Verification is no longer active")
            if record.face_match.status != FaceMatchStatus.READY:
                raise SessionConflict("Face matching is not ready to start")
            record.face_match.status = FaceMatchStatus.PROCESSING
            record.face_match.error_code = None
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def complete_face_match(
        self, session_id: str, similarity: float
    ) -> SessionSnapshot | None:
        if not isfinite(similarity):
            raise ValueError("Face-match similarity must be finite")
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.status != VerificationStatus.IN_PROGRESS
                or record.face_match.status != FaceMatchStatus.PROCESSING
            ):
                return None
            record.face_match.status = FaceMatchStatus.COMPLETED
            record.face_match.similarity = float(similarity)
            record.face_match.error_code = None
            self._refresh_verification_completion(record)
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def fail_face_match(self, session_id: str, code: str) -> SessionSnapshot | None:
        if code not in _FACE_MATCH_FAILURE_CODES:
            raise ValueError("Face-match failure code is not trusted")
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.status != VerificationStatus.IN_PROGRESS
                or record.face_match.status != FaceMatchStatus.PROCESSING
            ):
                return None
            record.face_match.status = FaceMatchStatus.FAILED
            record.face_match.similarity = None
            record.face_match.error_code = code
            self._refresh_verification_failure(record)
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def refresh_verification_completion(self, session_id: str) -> SessionSnapshot | None:
        """Commit the final lifecycle transition exactly once when checks complete."""
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return None
            if not self._refresh_verification_completion(record):
                return None
            record.event_sequence += 1
            self._refresh_expiry(record)
            return _snapshot(record)

    def face_match_result(self, session_id: str) -> FaceMatchResultSnapshot:
        with self._lock:
            record = self._get_record(session_id)
            return FaceMatchResultSnapshot(
                status=record.face_match.status,
                similarity=(
                    record.face_match.similarity
                    if record.face_match.status == FaceMatchStatus.COMPLETED
                    else None
                ),
            )

    def release_face_match_biometric_artifacts(self, session_id: str) -> None:
        """Release terminal face-match inputs without discarding document metadata."""
        with self._lock:
            record = self._get_record(session_id)
            if record.face_match.status not in {
                FaceMatchStatus.COMPLETED,
                FaceMatchStatus.FAILED,
            }:
                raise SessionConflict("Face-match artifacts cannot be released while active")
            portrait_id = _face_match_portrait_artifact_id(record.document.result)
            live_id = record.liveness.live_face_artifact_id
            artifact_ids = tuple(artifact_id for artifact_id in (portrait_id, live_id) if artifact_id is not None)
            if artifact_ids and self.portrait_artifacts is not None:
                self.portrait_artifacts.release_owned(artifact_ids, record.session_id)
            if portrait_id is not None:
                record.document.result = _clear_portrait_artifact_id(record.document.result)
            record.liveness.live_face_artifact_id = None
            record.liveness.live_face_eligible = False

    def fail_liveness(self, session_id: str, code: str) -> SessionSnapshot | None:
        with self._lock:
            record = self._records.get(session_id)
            if (
                record is None
                or record.status != VerificationStatus.IN_PROGRESS
                or record.liveness.status != LivenessStatus.PROCESSING
            ):
                return None
            self._release_owned_live_face(record)
            record.liveness.result = None
            record.liveness.error_code = code
            record.liveness.status = LivenessStatus.FAILED
            self._refresh_verification_failure(record)
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
                self._clear_sensitive_state(record)
            self._expired.pop(session_id, None)

    def resolve_portrait_artifact(self, session_id: str):
        """Internal-only future matcher boundary; no API route calls this method."""
        with self._lock:
            record = self._get_record(session_id)
            result = record.document.result
            portrait = result.front.portrait if result is not None and result.front is not None else None
            if portrait is None or portrait.artifact_id is None or self.portrait_artifacts is None:
                return None
            return self.portrait_artifacts.get(portrait.artifact_id, session_id=session_id)

    def resolve_face_match_portrait(self, session_id: str) -> Image | None:
        """Return the document portrait only when it is matcher-eligible."""
        with self._lock:
            record = self._get_record(session_id)
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
        with self._lock:
            record = self._get_record(session_id)
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
        """Testing and diagnostics helper; never exposes image bytes."""
        with self._lock:
            record = self._records.get(session_id)
            return bool(record and (record.document.front is not None or record.document.back is not None))

    def cleanup(self) -> int:
        with self._lock:
            return self._purge_expired(_now())

    def expiry_tombstone_deadline(self, session_id: str) -> datetime | None:
        """Return only the non-sensitive expiry tombstone deadline, if retained."""
        with self._lock:
            self._purge_expired(_now())
            return self._expired.get(session_id)

    def _get_record(self, session_id: str) -> _SessionRecord:
        self._purge_expired(_now())
        try:
            return self._records[session_id]
        except KeyError as exc:
            if session_id in self._expired:
                raise SessionExpired("Session has expired") from exc
            raise SessionNotFound("Session was not found") from exc

    def _can_finish_document_job(self, record: _SessionRecord | None, job_id: str) -> bool:
        return bool(
            record is not None
            and record.status == VerificationStatus.IN_PROGRESS
            and record.document.job_id == job_id
            and record.document.status == DocumentStatus.PROCESSING
        )

    def _refresh_face_match_readiness(self, record: _SessionRecord) -> bool:
        """Advance only BLOCKED sessions once every configured input is usable."""
        if (
            not record.face_match_required
            or record.face_match.status != FaceMatchStatus.BLOCKED
            or record.status != VerificationStatus.IN_PROGRESS
            or record.document.status != DocumentStatus.PASSED
            or record.liveness.status != LivenessStatus.PASSED
            or _face_match_portrait_artifact_id(record.document.result) is None
            or not record.liveness.live_face_eligible
            or record.liveness.live_face_artifact_id is None
        ):
            return False
        record.face_match.status = FaceMatchStatus.READY
        return True

    def _refresh_verification_completion(self, record: _SessionRecord) -> bool:
        if record.status != VerificationStatus.IN_PROGRESS:
            return False
        if record.document.status != DocumentStatus.PASSED:
            return False
        if self._nif_blocks_completion(record):
            return False
        if record.liveness_required and record.liveness.status != LivenessStatus.PASSED:
            return False
        if (
            record.face_match_required
            and record.face_match.status != FaceMatchStatus.COMPLETED
        ):
            return False
        record.status = VerificationStatus.COMPLETED
        return True

    @staticmethod
    def _nif_blocks_completion(record: _SessionRecord) -> bool:
        if not record.nif_verification_enabled:
            return False
        # The post-extraction resolver must record whether the check is
        # disabled, inapplicable, identifier-unavailable, or claimed. This
        # brief pending state prevents a successful document from completing
        # before that decision is made.
        return not record.nif_verification.settled

    def _refresh_verification_failure(self, record: _SessionRecord) -> bool:
        if record.status != VerificationStatus.IN_PROGRESS:
            return False
        if record.document.status in {DocumentStatus.PARTIAL, DocumentStatus.FAILED}:
            record.status = VerificationStatus.FAILED
            return True
        if record.liveness_required and record.liveness.status == LivenessStatus.FAILED:
            record.status = VerificationStatus.FAILED
            return True
        if record.face_match_required and record.face_match.status == FaceMatchStatus.FAILED:
            record.status = VerificationStatus.FAILED
            return True
        return False

    def _purge_expired(self, now: datetime) -> int:
        self._expired = {
            session_id: retained_until
            for session_id, retained_until in self._expired.items()
            if retained_until > now
        }
        expired = [
            session_id
            for session_id, record in self._records.items()
            if record.expires_at <= now
        ]
        for session_id in expired:
            record = self._records.pop(session_id)
            record.status = VerificationStatus.EXPIRED
            self._clear_sensitive_state(record)
            self._expired[session_id] = now + self._ttl
        return len(expired)

    def _refresh_expiry(self, record: _SessionRecord) -> None:
        record.expires_at = _now() + self._ttl

    def _clear_sensitive_state(self, record: _SessionRecord) -> None:
        if self.portrait_artifacts is not None:
            self.portrait_artifacts.release_owned(
                _portrait_artifact_ids(record.document.result), record.session_id
            )
            self._release_owned_live_face(record)
        _clear_sensitive_state(record)

    def _release_pending_live_face(self, artifact_id: str | None) -> None:
        if artifact_id is not None and self.portrait_artifacts is not None:
            self.portrait_artifacts.release_pending((artifact_id,))

    def _release_owned_live_face(self, record: _SessionRecord) -> None:
        if record.liveness.live_face_artifact_id is not None and self.portrait_artifacts is not None:
            self.portrait_artifacts.release_owned(
                (record.liveness.live_face_artifact_id,), record.session_id
            )
        record.liveness.live_face_artifact_id = None
        record.liveness.live_face_eligible = False
        record.face_match.similarity = None
        record.face_match.error_code = None


_FACE_MATCH_FAILURE_CODES = frozenset(
    {
        "FACE_COMPARISON_FAILED",
        "FACE_COMPARISON_REFERENCE_UNAVAILABLE",
        "FACE_COMPARISON_PROBE_UNAVAILABLE",
        "FACE_COMPARISON_REFERENCE_ZERO_FACES",
        "FACE_COMPARISON_PROBE_ZERO_FACES",
        "FACE_COMPARISON_REFERENCE_MULTIPLE_FACES",
        "FACE_COMPARISON_PROBE_MULTIPLE_FACES",
        "FACE_COMPARISON_REFERENCE_RECOGNITION_FAILED",
        "FACE_COMPARISON_PROBE_RECOGNITION_FAILED",
        "FACE_COMPARISON_INVALID_EMBEDDING",
        "FACE_COMPARISON_EMPTY_EMBEDDING",
        "FACE_COMPARISON_ZERO_NORM_EMBEDDING",
        "FACE_COMPARISON_NONFINITE_EMBEDDING",
        "FACE_COMPARISON_EMBEDDING_DIMENSION_MISMATCH",
        "FACE_COMPARISON_NONFINITE_SIMILARITY",
    }
)


def _face_match_portrait_artifact_id(result: DocumentExtractionResult | None) -> str | None:
    portrait = result.front.portrait if result is not None and result.front is not None else None
    if portrait is None or not portrait.eligible_for_face_match:
        return None
    return portrait.artifact_id


def _clear_portrait_artifact_id(
    result: DocumentExtractionResult | None,
) -> DocumentExtractionResult | None:
    if result is None or result.front is None or result.front.portrait is None:
        return result
    portrait = result.front.portrait
    return replace(
        result,
        front=replace(result.front, portrait=replace(portrait, artifact_id=None)),
    )


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
        nif_verification_status=record.nif_verification.status,
        nif_verification_source=record.nif_verification.source,
        nif_name_match=record.nif_verification.name_match,
        liveness_required=record.liveness_required,
        face_match_required=record.face_match_required,
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
    record.liveness.result = None
    record.liveness.error_code = None
    record.liveness.live_face_artifact_id = None
    record.liveness.live_face_eligible = False


def _nif_profile_present(result: DocumentExtractionResult | None) -> bool:
    return result is not None and result.front is not None and result.front.profile_id == "ao_id_card/front/v1"


def _nif_candidate(result: DocumentExtractionResult | None) -> tuple[str, str | None] | None:
    """Return the mapped valid NIF and optional valid document name only."""
    if not _nif_profile_present(result):
        return None
    assert result is not None and result.front is not None
    identifier = result.front.fields.get("id_number")
    if (
        identifier is None
        or identifier.status.value != "valid"
        or not identifier.normalized_value
    ):
        return None
    name = result.front.fields.get("full_name")
    claimed_name = (
        name.normalized_value
        if name is not None and name.status.value == "valid" and name.normalized_value
        else None
    )
    return identifier.normalized_value, claimed_name


def _portrait_artifact_ids(result: DocumentExtractionResult | None) -> tuple[str, ...]:
    if result is None:
        return ()
    return tuple(
        side_result.portrait.artifact_id
        for side_result in (result.front, result.back)
        if side_result is not None
        and side_result.portrait is not None
        and side_result.portrait.artifact_id is not None
    )


def _now() -> datetime:
    return datetime.now(UTC)
