"""Safe lifecycle event values shared with webhook delivery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from ...domain.verification import (
    DocumentStatus,
    FaceMatchStatus,
    LivenessStatus,
    NextAction,
    NifVerificationStatus,
    SessionStatus,
    VerificationStatus,
)
from ..sessions.store import SessionSnapshot


_DOCUMENT_ACTIVE = {
    DocumentStatus.READY,
    DocumentStatus.QUEUED,
    DocumentStatus.PROCESSING,
}


@dataclass(frozen=True)
class WebhookEvent:
    event_id: str
    session_id: str
    sequence: int
    event_type: str
    session_status: str
    next_action: str | None
    result_available: bool
    created_at: str
    nif_verification: dict[str, object] | None = None

    @classmethod
    def from_snapshot(
        cls, snapshot: SessionSnapshot, *, transition_reason: str
    ) -> "WebhookEvent | None":
        event_type = _event_type(transition_reason)
        if event_type is None:
            return None
        status = _session_status(snapshot)
        return cls(
            event_id=uuid4().hex,
            session_id=snapshot.session_id,
            sequence=snapshot.event_sequence,
            event_type=event_type,
            session_status=status.value,
            next_action=(
                action.value
                if (action := _next_action(snapshot, status)) is not None
                else None
            ),
            result_available=snapshot.document.result_available,
            created_at=datetime.now(UTC).isoformat(),
            nif_verification=(
                {
                    "status": snapshot.nif_verification_status.value,
                    "source": snapshot.nif_verification_source,
                    "name_match": snapshot.nif_name_match,
                }
                if event_type == "verification.nif.completed"
                else None
            ),
        )

    def payload(self) -> bytes:
        data: dict[str, object]
        if self.nif_verification is not None:
            data = {
                "session_id": self.session_id,
                "sequence": self.sequence,
                "nif_verification": self.nif_verification,
            }
        else:
            data = {
                "session_id": self.session_id,
                "sequence": self.sequence,
                "status": self.session_status,
                "next_action": self.next_action,
                "result_available": self.result_available,
            }
        return json.dumps(
            {
                "schema_version": "1",
                "id": self.event_id,
                "type": self.event_type,
                "created_at": self.created_at,
                "data": data,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()


def _session_status(snapshot: SessionSnapshot) -> SessionStatus:
    if snapshot.verification_status == VerificationStatus.FAILED:
        return SessionStatus.FAILED
    if snapshot.nif_verification_status == NifVerificationStatus.NOT_FOUND:
        return SessionStatus.FAILED
    if snapshot.nif_verification_status == NifVerificationStatus.PROCESSING:
        return SessionStatus.IN_PROGRESS
    if snapshot.verification_status == VerificationStatus.COMPLETED:
        return SessionStatus.COMPLETED
    if (
        snapshot.document.status in {DocumentStatus.PARTIAL, DocumentStatus.FAILED}
        or (snapshot.liveness_required and snapshot.liveness_status == LivenessStatus.FAILED)
        or (snapshot.face_match_required and snapshot.face_match_status == FaceMatchStatus.FAILED)
    ):
        return SessionStatus.FAILED
    if (
        snapshot.document.status == DocumentStatus.PASSED
        and (not snapshot.liveness_required or snapshot.liveness_status == LivenessStatus.PASSED)
        and (not snapshot.face_match_required or snapshot.face_match_status == FaceMatchStatus.COMPLETED)
    ):
        return SessionStatus.COMPLETED
    return SessionStatus.IN_PROGRESS


def _next_action(snapshot: SessionSnapshot, status: SessionStatus) -> NextAction | None:
    if status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
        return None
    if snapshot.document.front_capture.value == "missing":
        return NextAction.SUBMIT_DOCUMENT_FRONT
    if snapshot.document.back_capture.value == "missing":
        return NextAction.SUBMIT_DOCUMENT_BACK
    if snapshot.document.status in _DOCUMENT_ACTIVE:
        return NextAction.WAIT
    if snapshot.liveness_required and snapshot.liveness_status in {
        LivenessStatus.BLOCKED,
        LivenessStatus.READY,
    }:
        return NextAction.SUBMIT_LIVENESS
    return NextAction.WAIT


def _event_type(transition_reason: str) -> str | None:
    return {
        "verification.session.created": "verification.session.created",
        "document.passed": "verification.document.completed",
        "document.partial": "verification.document.failed",
        "document.failed": "verification.document.failed",
        "liveness.passed": "verification.liveness.passed",
        "liveness.failed": "verification.liveness.failed",
        "verification.failed": "verification.processing.failed",
        "verification.completed": "verification.processing.completed",
        "nif.completed": "verification.nif.completed",
    }.get(transition_reason)
