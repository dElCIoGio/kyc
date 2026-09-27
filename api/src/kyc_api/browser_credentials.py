"""Short-lived, session-bound browser credentials for the hosted verifier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from secrets import token_urlsafe
from threading import RLock

from .sessions import SessionExpired, SessionStore


class BrowserCredentialError(RuntimeError):
    """Safe browser-credential authentication failure."""


@dataclass
class _CredentialRecord:
    digest: bytes
    session_id: str
    expires_at: datetime
    saw_session_expiry: bool = False


class BrowserCredentialStore:
    """Memory-only opaque browser credentials, one active credential per session."""

    def __init__(self, *, sessions: SessionStore, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("browser credential TTL must be positive")
        self._sessions = sessions
        self._ttl = timedelta(seconds=ttl_seconds)
        self._by_digest: dict[bytes, _CredentialRecord] = {}
        self._by_session: dict[str, bytes] = {}
        self._lock = RLock()

    def issue(self, session_id: str) -> tuple[str, datetime]:
        """Rotate and return a credential for a currently live session."""
        snapshot = self._sessions.get(session_id)
        now = _now()
        expires_at = min(snapshot.expires_at, now + self._ttl)
        raw = f"bt_{token_urlsafe(32)}"
        digest = _digest(raw)
        with self._lock:
            self._remove_session_locked(session_id)
            self._by_digest[digest] = _CredentialRecord(
                digest=digest, session_id=session_id, expires_at=expires_at
            )
            self._by_session[session_id] = digest
        return raw, expires_at

    def authorize(self, token: str, session_id: str, *, safe_read: bool) -> str:
        """Authorize a token for its sole bound session.

        An expired session intentionally remains authenticatable for a safe GET so
        SessionStore can return the stable 410 SESSION_EXPIRED contract.
        """
        digest = _digest(token)
        now = _now()
        with self._lock:
            self._cleanup_locked(now)
            record = self._by_digest.get(digest)
            if record is None or record.session_id != session_id:
                raise BrowserCredentialError("browser credential is invalid")
            if record.expires_at <= now:
                self._remove_digest_locked(digest)
                raise BrowserCredentialError("browser credential has expired")
        try:
            self._sessions.get(session_id)
        except SessionExpired:
            with self._lock:
                stored = self._by_digest.get(digest)
                if stored is not None:
                    stored.saw_session_expiry = True
            if safe_read:
                return digest.hex()
            raise
        return digest.hex()

    def revoke_session(self, session_id: str) -> None:
        """Immediately remove a credential after explicit session deletion."""
        with self._lock:
            self._remove_session_locked(session_id)

    def cleanup(self) -> None:
        with self._lock:
            self._cleanup_locked(_now())

    def _cleanup_locked(self, now: datetime) -> None:
        for digest, record in tuple(self._by_digest.items()):
            if record.expires_at <= now:
                self._remove_digest_locked(digest)
                continue
            tombstone_deadline = self._sessions.expiry_tombstone_deadline(record.session_id)
            if tombstone_deadline is not None:
                record.saw_session_expiry = True
            if record.saw_session_expiry and tombstone_deadline is None:
                self._remove_digest_locked(digest)

    def _remove_session_locked(self, session_id: str) -> None:
        digest = self._by_session.get(session_id)
        if digest is not None:
            self._remove_digest_locked(digest)

    def _remove_digest_locked(self, digest: bytes) -> None:
        record = self._by_digest.pop(digest, None)
        if record is not None and self._by_session.get(record.session_id) == digest:
            self._by_session.pop(record.session_id, None)


def _digest(token: str) -> bytes:
    return sha256(token.encode("utf-8")).digest()


def _now() -> datetime:
    return datetime.now(UTC)
