from __future__ import annotations

from dataclasses import dataclass
from threading import RLock, Timer
from uuid import uuid4

from .contracts import Image, readonly_image


class PortraitArtifactStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Artifact:
    image: Image
    owner_session_id: str | None


class InMemoryPortraitArtifactStore:
    """Thread-safe, process-memory storage for short-lived portrait crops.

    A crop is pending until one active verification session atomically claims it.
    Pending artifacts must be released by the caller whenever a job cannot be
    completed; claimed artifacts are released by their owning session lifecycle.
    """

    def __init__(self, *, pending_ttl_seconds: float = 60.0) -> None:
        if pending_ttl_seconds <= 0:
            raise ValueError("pending_ttl_seconds must be positive")
        self._artifacts: dict[str, _Artifact] = {}
        self._pending_timers: dict[str, Timer] = {}
        self._lock = RLock()
        self._pending_ttl_seconds = pending_ttl_seconds

    def put(self, image: Image) -> str:
        try:
            stored = readonly_image(image, copy=True)
        except Exception as exc:
            raise PortraitArtifactStoreError("Portrait crop could not be retained") from exc
        artifact_id = uuid4().hex
        with self._lock:
            self._artifacts[artifact_id] = _Artifact(stored, owner_session_id=None)
            timer = Timer(self._pending_ttl_seconds, lambda: self.release_pending((artifact_id,)))
            timer.daemon = True
            self._pending_timers[artifact_id] = timer
            timer.start()
        return artifact_id

    def claim(self, artifact_ids: tuple[str, ...], session_id: str) -> bool:
        """Atomically transfer pending artifacts to exactly one session."""
        unique_ids = tuple(dict.fromkeys(artifact_ids))
        if len(unique_ids) != len(artifact_ids) or not session_id:
            return False
        with self._lock:
            if any(
                artifact_id not in self._artifacts
                or self._artifacts[artifact_id].owner_session_id is not None
                for artifact_id in unique_ids
            ):
                return False
            for artifact_id in unique_ids:
                artifact = self._artifacts[artifact_id]
                self._artifacts[artifact_id] = _Artifact(artifact.image, session_id)
                self._cancel_pending_timer(artifact_id)
            return True

    def release_pending(self, artifact_ids: tuple[str, ...]) -> None:
        with self._lock:
            for artifact_id in set(artifact_ids):
                artifact = self._artifacts.get(artifact_id)
                if artifact is not None and artifact.owner_session_id is None:
                    self._artifacts.pop(artifact_id, None)
                    self._cancel_pending_timer(artifact_id)

    def release_owned(self, artifact_ids: tuple[str, ...], session_id: str) -> None:
        with self._lock:
            for artifact_id in set(artifact_ids):
                artifact = self._artifacts.get(artifact_id)
                if artifact is not None and artifact.owner_session_id == session_id:
                    self._artifacts.pop(artifact_id, None)
                    self._cancel_pending_timer(artifact_id)

    def _restore_owned_to_pending(self, artifact_ids: tuple[str, ...], session_id: str) -> None:
        """Undo a just-claimed ownership transfer without exposing crop data.

        This narrowly scoped operation is for a durable transaction that failed
        after ``claim``.  It only affects artifacts still owned by the same
        session, so it cannot be used to take or reassign another session's
        crop.
        """
        with self._lock:
            for artifact_id in set(artifact_ids):
                artifact = self._artifacts.get(artifact_id)
                if artifact is not None and artifact.owner_session_id == session_id:
                    self._artifacts[artifact_id] = _Artifact(artifact.image, None)
                    timer = Timer(
                        self._pending_ttl_seconds,
                        lambda artifact_id=artifact_id: self.release_pending((artifact_id,)),
                    )
                    timer.daemon = True
                    self._pending_timers[artifact_id] = timer
                    timer.start()

    def get(self, artifact_id: str, *, session_id: str) -> Image | None:
        """Return a readonly crop only to its owning in-process session."""
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None or artifact.owner_session_id != session_id:
                return None
            return artifact.image

    def exists(self, artifact_id: str) -> bool:
        """Testing/diagnostic helper that never exposes pixels."""
        with self._lock:
            return artifact_id in self._artifacts

    def _cancel_pending_timer(self, artifact_id: str) -> None:
        timer = self._pending_timers.pop(artifact_id, None)
        if timer is not None:
            timer.cancel()
