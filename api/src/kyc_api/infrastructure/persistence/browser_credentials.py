from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
from secrets import token_urlsafe

from sqlalchemy import Engine, delete, select, text
from sqlalchemy.orm import sessionmaker

from ...application.security.browser_credentials import BrowserCredentialError
from ...application.sessions.store import SessionExpired, SessionStore
from ...application.security.browser_credentials import _now
from .models import BrowserCredential, SessionTombstone


class PostgresBrowserCredentialStore:
    def __init__(self, *, engine: Engine, sessions: SessionStore, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("browser credential TTL must be positive")
        self._sessions_store = sessions
        self._database = sessionmaker(engine, expire_on_commit=False)
        self._ttl = timedelta(seconds=ttl_seconds)

    def issue(self, session_id: str) -> tuple[str, datetime]:
        now = _now()
        expires_at = now + self._ttl
        raw = f"bt_{token_urlsafe(32)}"
        digest = _digest(raw)
        # Match PostgresSessionStore.delete's lock order: process-local runtime
        # lock, per-session advisory lock, then session row.  Holding the same
        # re-entrant runtime lock also closes the delete/rotate gap between the
        # live-session check and credential replacement.
        runtime_lock = getattr(self._sessions_store, "_runtime_lock", None)
        if runtime_lock is None:
            return self._issue_locked(session_id, raw, digest, now, expires_at)
        with runtime_lock:
            return self._issue_locked(session_id, raw, digest, now, expires_at)

    def _issue_locked(
        self,
        session_id: str,
        raw: str,
        digest: bytes,
        now: datetime,
        expires_at: datetime,
    ) -> tuple[str, datetime]:
        with self._database.begin() as database:
            database.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:value, 0))"),
                {"value": session_id},
            )
            self._sessions_store.get(session_id)
            database.execute(
                delete(BrowserCredential).where(
                    BrowserCredential.session_id == session_id
                )
            )
            database.add(
                BrowserCredential(
                    digest=digest,
                    session_id=session_id,
                    issued_at=now,
                    expires_at=expires_at,
                    saw_session_expiry=False,
                )
            )
        return raw, expires_at

    def authorize(self, token: str, session_id: str, *, safe_read: bool) -> str:
        digest = _digest(token)
        now = _now()
        expired = False
        with self._database.begin() as database:
            record = database.scalar(select(BrowserCredential).where(BrowserCredential.digest == digest).with_for_update())
            if record is None or record.session_id != session_id:
                raise BrowserCredentialError("browser credential is invalid")
            if record.expires_at <= now:
                database.delete(record)
                expired = True
        if expired:
            raise BrowserCredentialError("browser credential has expired")
        try:
            self._sessions_store.get(session_id)
        except SessionExpired:
            with self._database.begin() as database:
                database.execute(
                    BrowserCredential.__table__.update()
                    .where(BrowserCredential.digest == digest)
                    .values(saw_session_expiry=True)
                )
            if safe_read:
                return digest.hex()
            raise
        return digest.hex()

    def revoke_session(self, session_id: str) -> None:
        with self._database.begin() as database:
            database.execute(delete(BrowserCredential).where(BrowserCredential.session_id == session_id))

    def cleanup(self) -> None:
        now = _now()
        with self._database.begin() as database:
            database.execute(delete(BrowserCredential).where(BrowserCredential.expires_at <= now))
            records = list(database.scalars(select(BrowserCredential).with_for_update(skip_locked=True)))
            for record in records:
                tombstone = database.scalar(select(SessionTombstone.retained_until).where(SessionTombstone.session_id == record.session_id))
                if tombstone is not None and tombstone > now:
                    record.saw_session_expiry = True
                elif record.saw_session_expiry:
                    database.delete(record)


def _digest(token: str) -> bytes:
    return sha256(token.encode("utf-8")).digest()
