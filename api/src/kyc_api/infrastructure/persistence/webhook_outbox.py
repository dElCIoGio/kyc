from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from sqlalchemy import Engine, and_, delete, exists, or_, select, update
from sqlalchemy.orm import aliased, sessionmaker

from ...infrastructure.webhooks.outbox import _StoredEvent
from ...application.security.browser_credentials import _now
from .models import WebhookEventRow


class PostgresWebhookOutbox:
    """Lease-based, at-least-once PostgreSQL webhook outbox."""

    def __init__(self, engine: Engine, *, retention_seconds: int, lease_seconds: float = 30.0) -> None:
        if retention_seconds <= 0 or lease_seconds <= 0:
            raise ValueError("outbox retention and lease must be positive")
        self._database = sessionmaker(engine, expire_on_commit=False)
        self._retention = timedelta(seconds=retention_seconds)
        self._lease = timedelta(seconds=lease_seconds)
        self._owner = uuid4().hex

    def due(self) -> _StoredEvent | None:
        now = _now()
        earlier = aliased(WebhookEventRow)
        with self._database.begin() as database:
            database.execute(
                delete(WebhookEventRow).where(
                    WebhookEventRow.expires_at <= now,
                    or_(WebhookEventRow.delivered_at.is_not(None), WebhookEventRow.dead_at.is_not(None)),
                )
            )
            database.execute(
                update(WebhookEventRow)
                .where(WebhookEventRow.delivered_at.is_(None), WebhookEventRow.dead_at.is_(None), WebhookEventRow.expires_at <= now)
                .values(dead_at=now, lease_owner=None, lease_expires_at=None)
            )
            candidate = database.scalar(
                select(WebhookEventRow)
                .where(
                    WebhookEventRow.delivered_at.is_(None),
                    WebhookEventRow.dead_at.is_(None),
                    WebhookEventRow.next_attempt_at <= now,
                    or_(WebhookEventRow.lease_expires_at.is_(None), WebhookEventRow.lease_expires_at <= now),
                    ~exists().where(
                        earlier.session_id == WebhookEventRow.session_id,
                        earlier.delivered_at.is_(None),
                        earlier.dead_at.is_(None),
                        or_(
                            earlier.sequence < WebhookEventRow.sequence,
                            and_(earlier.sequence == WebhookEventRow.sequence, earlier.event_type < WebhookEventRow.event_type),
                        ),
                    ),
                )
                .order_by(WebhookEventRow.created_at, WebhookEventRow.session_id, WebhookEventRow.sequence, WebhookEventRow.event_type)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if candidate is None:
                return None
            candidate.lease_owner = self._owner
            candidate.lease_expires_at = now + self._lease
            return _StoredEvent(candidate.event_id, candidate.session_id, candidate.sequence, bytes(candidate.payload), candidate.attempts)

    def delivered(self, event_id: str) -> None:
        now = _now()
        with self._database.begin() as database:
            database.execute(
                update(WebhookEventRow)
                .where(WebhookEventRow.event_id == event_id, WebhookEventRow.lease_owner == self._owner)
                .values(delivered_at=now, lease_owner=None, lease_expires_at=None)
            )

    def failed(self, event_id: str, attempts: int, *, retry_after: float | None, retryable: bool) -> None:
        now = _now()
        values: dict[str, object] = {
            "attempts": attempts + 1,
            "lease_owner": None,
            "lease_expires_at": None,
        }
        if retryable:
            delay = min(300.0, retry_after if retry_after is not None else float(2 ** min(attempts, 8)))
            values["next_attempt_at"] = now + timedelta(seconds=max(1.0, delay))
        else:
            values["dead_at"] = now
        with self._database.begin() as database:
            database.execute(
                update(WebhookEventRow)
                .where(WebhookEventRow.event_id == event_id, WebhookEventRow.lease_owner == self._owner)
                .values(**values)
            )
