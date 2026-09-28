from __future__ import annotations

import hashlib
import hmac
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from ...application.events.lifecycle import WebhookEvent
from ...application.sessions.store import SessionSnapshot


logger = logging.getLogger("kyc_api.webhooks")


@dataclass(frozen=True)
class _StoredEvent:
    event_id: str
    session_id: str
    sequence: int
    payload: bytes
    attempts: int


class WebhookOutbox:
    """SQLite transactional-style outbox for non-sensitive lifecycle events."""

    def __init__(self, path: Path, *, retention_seconds: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._retention_seconds = retention_seconds
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(webhook_events)")
            }
            migrate_legacy_schema = bool(columns) and "event_type" not in columns
            if migrate_legacy_schema:
                connection.execute("DROP INDEX IF EXISTS webhook_events_due")
                connection.execute(
                    "ALTER TABLE webhook_events RENAME TO webhook_events_legacy"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS webhook_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL,
                    delivered_at REAL,
                    dead_at REAL,
                    UNIQUE(session_id, sequence, event_type)
                );
                CREATE INDEX IF NOT EXISTS webhook_events_due
                    ON webhook_events(next_attempt_at, session_id, sequence, event_type);
                """
            )
            if migrate_legacy_schema:
                connection.execute(
                    """INSERT INTO webhook_events
                    (event_id, session_id, sequence, event_type, payload, created_at,
                     expires_at, attempts, next_attempt_at, delivered_at, dead_at)
                    SELECT event_id, session_id, sequence, 'legacy', payload, created_at,
                           expires_at, attempts, next_attempt_at, delivered_at, dead_at
                    FROM webhook_events_legacy"""
                )
                connection.execute("DROP TABLE webhook_events_legacy")

    def enqueue(self, event: WebhookEvent) -> None:
        now = time.time()
        with self._lock, self._connection() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO webhook_events
                (event_id, session_id, sequence, event_type, payload, created_at, expires_at, next_attempt_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.session_id,
                    event.sequence,
                    event.event_type,
                    event.payload(),
                    now,
                    now + self._retention_seconds,
                    now,
                ),
            )

    def due(self) -> _StoredEvent | None:
        now = time.time()
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE webhook_events SET dead_at = ? WHERE delivered_at IS NULL AND dead_at IS NULL AND expires_at <= ?",
                (now, now),
            )
            row = connection.execute(
                """SELECT event_id, session_id, sequence, payload, attempts FROM webhook_events current
                WHERE current.delivered_at IS NULL AND current.dead_at IS NULL
                  AND current.next_attempt_at <= ?
                  AND NOT EXISTS (
                    SELECT 1 FROM webhook_events earlier
                    WHERE earlier.session_id = current.session_id
                      AND (
                        earlier.sequence < current.sequence
                        OR (
                          earlier.sequence = current.sequence
                          AND earlier.event_type < current.event_type
                        )
                      )
                      AND earlier.delivered_at IS NULL AND earlier.dead_at IS NULL
                  )
                ORDER BY current.created_at, current.session_id, current.sequence,
                         current.event_type LIMIT 1""",
                (now,),
            ).fetchone()
        return _StoredEvent(*row) if row is not None else None

    def delivered(self, event_id: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute("UPDATE webhook_events SET delivered_at = ? WHERE event_id = ?", (time.time(), event_id))

    def failed(self, event_id: str, attempts: int, *, retry_after: float | None, retryable: bool) -> None:
        now = time.time()
        if not retryable:
            with self._lock, self._connection() as connection:
                connection.execute("UPDATE webhook_events SET attempts = ?, dead_at = ? WHERE event_id = ?", (attempts + 1, now, event_id))
            return
        delay = min(300.0, retry_after if retry_after is not None else float(2 ** min(attempts, 8)))
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE webhook_events SET attempts = ?, next_attempt_at = ? WHERE event_id = ?",
                (attempts + 1, now + max(1.0, delay), event_id),
            )


class WebhookDispatcher:
    def __init__(self, *, outbox: WebhookOutbox, url: str, secret: str, timeout_seconds: float) -> None:
        self._outbox = outbox
        self._url = url
        self._secret = secret.encode()
        self._timeout_seconds = timeout_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="kyc-webhook-dispatch", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, snapshot: SessionSnapshot, transition_reason: str) -> None:
        event = WebhookEvent.from_snapshot(snapshot, transition_reason=transition_reason)
        if event is None:
            return
        self._outbox.enqueue(event)
        self._wake.set()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=self._timeout_seconds + 1)

    def _run(self) -> None:
        while not self._stop.is_set():
            event = self._outbox.due()
            if event is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            try:
                delivered, retry_after, retryable = self._deliver(event)
                if delivered:
                    self._outbox.delivered(event.event_id)
                else:
                    self._outbox.failed(event.event_id, event.attempts, retry_after=retry_after, retryable=retryable)
            except Exception as exc:
                logger.warning("webhook delivery failed", extra={"event": "webhook_delivery_failed", "exception_type": type(exc).__name__})
                self._outbox.failed(event.event_id, event.attempts, retry_after=None, retryable=True)

    def _deliver(self, event: _StoredEvent) -> tuple[bool, float | None, bool]:
        timestamp = str(int(time.time()))
        signature = hmac.new(self._secret, timestamp.encode() + b"." + event.payload, hashlib.sha256).hexdigest()
        request = urllib.request.Request(
            self._url,
            data=event.payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-KYC-Webhook-ID": event.event_id,
                "X-KYC-Webhook-Timestamp": timestamp,
                "X-KYC-Webhook-Signature": f"v1={signature}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                return 200 <= response.status < 300, None, response.status >= 500
        except urllib.error.HTTPError as exc:
            retry_after = None
            try:
                retry_after = min(300.0, float(exc.headers.get("Retry-After", "")))
            except ValueError:
                pass
            return False, retry_after, exc.code == 408 or exc.code == 429 or exc.code >= 500
        except urllib.error.URLError:
            return False, None, True
