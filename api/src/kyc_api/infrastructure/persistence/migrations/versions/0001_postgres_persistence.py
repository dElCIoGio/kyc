"""Create durable verification persistence tables."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_postgres_persistence"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "verification_sessions",
        sa.Column("session_id", sa.String(32), primary_key=True),
        sa.Column("verification_status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_sequence", sa.BigInteger(), nullable=False),
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column("liveness_required", sa.Boolean(), nullable=False),
        sa.Column("face_match_required", sa.Boolean(), nullable=False),
        sa.Column("nif_verification_enabled", sa.Boolean(), nullable=False),
        sa.Column("document_status", sa.String(32), nullable=False),
        sa.Column("document_front_capture_status", sa.String(32), nullable=False),
        sa.Column("document_back_capture_status", sa.String(32), nullable=False),
        sa.Column("document_job_id", sa.String(32)),
        sa.Column("document_result_json", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("document_error_code", sa.String(128)),
        sa.Column("liveness_status", sa.String(32), nullable=False),
        sa.Column("liveness_error_code", sa.String(128)),
        sa.Column("face_match_status", sa.String(32), nullable=False),
        sa.Column("face_match_error_code", sa.String(128)),
        sa.Column("nif_status", sa.String(32), nullable=False),
        sa.Column("nif_source", sa.String(64)),
        sa.Column("nif_name_match", sa.Boolean()),
        sa.Column("nif_not_run_reason", sa.String(64), nullable=False),
        sa.Column("nif_error_code", sa.String(128)),
        sa.Column("nif_settled", sa.Boolean(), nullable=False),
        sa.Column("nif_attempted", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_verification_sessions_expires_at", "verification_sessions", ["expires_at"])
    op.create_index("ix_verification_sessions_status", "verification_sessions", ["verification_status"])
    op.create_index("ix_verification_sessions_document_job_id", "verification_sessions", ["document_job_id"])

    op.create_table(
        "session_tombstones",
        sa.Column("session_id", sa.String(32), primary_key=True),
        sa.Column("retained_until", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_session_tombstones_retained_until", "session_tombstones", ["retained_until"])

    op.create_table(
        "browser_credentials",
        sa.Column("digest", sa.LargeBinary(32), primary_key=True),
        sa.Column("session_id", sa.String(32), nullable=False, unique=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("saw_session_expiry", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_browser_credentials_expires_at", "browser_credentials", ["expires_at"])

    op.create_table(
        "webhook_events",
        sa.Column("event_id", sa.String(32), primary_key=True),
        sa.Column("session_id", sa.String(32), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("event_type", sa.String(96), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True)),
        sa.Column("dead_at", sa.DateTime(timezone=True)),
        sa.Column("lease_owner", sa.String(64)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("session_id", "sequence", "event_type", name="uq_webhook_session_sequence_type"),
    )
    op.create_index("ix_webhook_events_due", "webhook_events", ["next_attempt_at", "session_id", "sequence", "event_type"])


def downgrade() -> None:
    op.drop_index("ix_webhook_events_due", table_name="webhook_events")
    op.drop_table("webhook_events")
    op.drop_index("ix_browser_credentials_expires_at", table_name="browser_credentials")
    op.drop_table("browser_credentials")
    op.drop_index("ix_session_tombstones_retained_until", table_name="session_tombstones")
    op.drop_table("session_tombstones")
    op.drop_index("ix_verification_sessions_document_job_id", table_name="verification_sessions")
    op.drop_index("ix_verification_sessions_status", table_name="verification_sessions")
    op.drop_index("ix_verification_sessions_expires_at", table_name="verification_sessions")
    op.drop_table("verification_sessions")
