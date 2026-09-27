"""Public API projections for internal verification state and extraction results."""

from __future__ import annotations

from kyc_engine import DocumentExtractionResult, ProcessingStatus

from .models import (
    DocumentStateResponse,
    DocumentStatus,
    FaceComparisonStateResponse,
    FaceMatchStatus,
    LivenessStateResponse,
    LivenessStatus,
    NextAction,
    PublicDocumentStatus,
    PublicFaceComparisonStatus,
    PublicLivenessStatus,
    ResultDocumentResponse,
    ResultFieldResponse,
    ResultIssueResponse,
    SessionResponse,
    SessionStatus,
    VerificationResultResponse,
)
from .sessions import SessionSnapshot


_DOCUMENT_ACTIVE = {
    DocumentStatus.READY,
    DocumentStatus.QUEUED,
    DocumentStatus.PROCESSING,
}
_DOCUMENT_SUCCEEDED = {DocumentStatus.PASSED, DocumentStatus.PARTIAL}


def session_response(snapshot: SessionSnapshot) -> SessionResponse:
    """Map engine lifecycle state to the stable external session contract."""
    current_status = _session_status(snapshot)
    return SessionResponse(
        session_id=snapshot.session_id,
        status=current_status,
        created_at=snapshot.created_at,
        expires_at=snapshot.expires_at,
        next_action=_next_action(snapshot, current_status),
        document=DocumentStateResponse(
            status=_document_status(snapshot.document.status),
            front_capture=snapshot.document.front_capture,
            back_capture=snapshot.document.back_capture,
            result_available=snapshot.document.result_available,
        ),
        liveness=_liveness_state(snapshot),
        face_comparison=_face_comparison_state(snapshot),
    )


def result_response(
    snapshot: SessionSnapshot, result: DocumentExtractionResult
) -> VerificationResultResponse:
    """Return canonical document values without engine, biometric, or QR internals."""
    fields: dict[str, ResultFieldResponse] = {}
    issues: list[ResultIssueResponse] = []
    document_type: str | None = None
    for side in (result.front, result.back):
        if side is None:
            continue
        document_type = document_type or side.document_type
        for name, field in side.fields.items():
            # The profile controls field names; preserve a deterministic first side
            # rather than exposing OCR candidates or reconciliation alternatives.
            fields.setdefault(
                name,
                ResultFieldResponse(
                    status=field.status.value,
                    value=field.normalized_value,
                ),
            )
        issues.extend(
            ResultIssueResponse(
                code=issue.code,
                message=issue.message,
                field_name=issue.field_name,
            )
            for issue in side.issues
        )
    issues.extend(
        ResultIssueResponse(
            code=issue.code,
            message=issue.message,
            field_name=issue.field_name,
        )
        for issue in result.issues
    )
    return VerificationResultResponse(
        session_id=snapshot.session_id,
        document=ResultDocumentResponse(
            status={
                ProcessingStatus.SUCCESS: "completed",
                ProcessingStatus.PARTIAL: "partial",
                ProcessingStatus.FAILED: "failed",
            }[result.status],
            document_type=document_type,
            fields=fields,
            issues=tuple(issues),
        ),
        liveness=_liveness_state(snapshot),
        face_comparison=_face_comparison_state(snapshot),
    )


def _session_status(snapshot: SessionSnapshot) -> SessionStatus:
    document_failed = snapshot.document.status == DocumentStatus.FAILED
    liveness_failed = snapshot.liveness_required and snapshot.liveness_status == LivenessStatus.FAILED
    comparison_failed = snapshot.face_match_required and snapshot.face_match_status == FaceMatchStatus.FAILED
    if document_failed or liveness_failed or comparison_failed:
        return SessionStatus.FAILED

    document_succeeded = snapshot.document.status in _DOCUMENT_SUCCEEDED
    liveness_succeeded = not snapshot.liveness_required or snapshot.liveness_status == LivenessStatus.PASSED
    comparison_succeeded = not snapshot.face_match_required or snapshot.face_match_status == FaceMatchStatus.COMPLETED
    if document_succeeded and liveness_succeeded and comparison_succeeded:
        return SessionStatus.COMPLETED
    return SessionStatus.IN_PROGRESS


def _next_action(
    snapshot: SessionSnapshot, current_status: SessionStatus
) -> NextAction | None:
    if current_status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
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


def _document_status(status: DocumentStatus) -> PublicDocumentStatus:
    if status == DocumentStatus.AWAITING_CAPTURE:
        return PublicDocumentStatus.AWAITING_CAPTURE
    if status in _DOCUMENT_ACTIVE:
        return PublicDocumentStatus.PROCESSING
    if status == DocumentStatus.FAILED:
        return PublicDocumentStatus.FAILED
    return PublicDocumentStatus.COMPLETED


def _liveness_state(snapshot: SessionSnapshot) -> LivenessStateResponse:
    if not snapshot.liveness_required:
        return LivenessStateResponse(status=PublicLivenessStatus.NOT_AVAILABLE)
    return LivenessStateResponse(
        status={
            LivenessStatus.BLOCKED: PublicLivenessStatus.NOT_STARTED,
            LivenessStatus.READY: PublicLivenessStatus.NOT_STARTED,
            LivenessStatus.PROCESSING: PublicLivenessStatus.PROCESSING,
            LivenessStatus.PASSED: PublicLivenessStatus.PASSED,
            LivenessStatus.FAILED: PublicLivenessStatus.FAILED,
        }[snapshot.liveness_status]
    )


def _face_comparison_state(snapshot: SessionSnapshot) -> FaceComparisonStateResponse:
    if not snapshot.face_match_required:
        return FaceComparisonStateResponse(status=PublicFaceComparisonStatus.NOT_AVAILABLE)
    return FaceComparisonStateResponse(
        status={
            FaceMatchStatus.BLOCKED: PublicFaceComparisonStatus.NOT_STARTED,
            FaceMatchStatus.READY: PublicFaceComparisonStatus.NOT_STARTED,
            FaceMatchStatus.PROCESSING: PublicFaceComparisonStatus.PROCESSING,
            FaceMatchStatus.COMPLETED: PublicFaceComparisonStatus.COMPLETED,
            FaceMatchStatus.FAILED: PublicFaceComparisonStatus.FAILED,
        }[snapshot.face_match_status]
    )
