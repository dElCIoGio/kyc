from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Index, Integer, LargeBinary, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class VerificationSession(Base):
    __tablename__ = "verification_sessions"

    session_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    verification_status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)

    liveness_required: Mapped[bool] = mapped_column(Boolean, nullable=False)
    face_match_required: Mapped[bool] = mapped_column(Boolean, nullable=False)
    nif_verification_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)

    document_status: Mapped[str] = mapped_column(String(32), nullable=False)
    document_front_capture_status: Mapped[str] = mapped_column(String(32), nullable=False)
    document_back_capture_status: Mapped[str] = mapped_column(String(32), nullable=False)
    document_job_id: Mapped[str | None] = mapped_column(String(32))
    document_result_json: Mapped[dict | None] = mapped_column(JSONB)
    document_error_code: Mapped[str | None] = mapped_column(String(128))

    liveness_status: Mapped[str] = mapped_column(String(32), nullable=False)
    liveness_error_code: Mapped[str | None] = mapped_column(String(128))
    face_match_status: Mapped[str] = mapped_column(String(32), nullable=False)
    face_match_error_code: Mapped[str | None] = mapped_column(String(128))

    nif_status: Mapped[str] = mapped_column(String(32), nullable=False)
    nif_source: Mapped[str | None] = mapped_column(String(64))
    nif_name_match: Mapped[bool | None] = mapped_column(Boolean)
    nif_not_run_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    nif_error_code: Mapped[str | None] = mapped_column(String(128))
    nif_settled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    nif_attempted: Mapped[bool] = mapped_column(Boolean, nullable=False)

    __table_args__ = (
        Index("ix_verification_sessions_expires_at", "expires_at"),
        Index("ix_verification_sessions_status", "verification_status"),
        Index("ix_verification_sessions_document_job_id", "document_job_id"),
    )


class SessionTombstone(Base):
    __tablename__ = "session_tombstones"

    session_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    retained_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


class BrowserCredential(Base):
    __tablename__ = "browser_credentials"

    digest: Mapped[bytes] = mapped_column(LargeBinary(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    saw_session_expiry: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class WebhookEventRow(Base):
    __tablename__ = "webhook_events"

    event_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(32), nullable=False)
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    event_type: Mapped[str] = mapped_column(String(96), nullable=False)
    payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dead_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("session_id", "sequence", "event_type", name="uq_webhook_session_sequence_type"),
        Index("ix_webhook_events_due", "next_attempt_at", "session_id", "sequence", "event_type"),
    )
