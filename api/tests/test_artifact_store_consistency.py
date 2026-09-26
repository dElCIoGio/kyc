from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore
from kyc_api.main import create_app
from kyc_api.sessions import SessionStore

from helpers import FakeCoordinator, settings


class ArtifactStoreConsistencyTests(unittest.TestCase):
    def test_same_explicit_coordinator_and_session_store_is_preserved(self) -> None:
        artifacts = InMemoryPortraitArtifactStore()
        coordinator = FakeCoordinator()
        coordinator.portrait_artifacts = artifacts
        session_store = SessionStore(
            ttl_seconds=60, max_sessions=2, portrait_artifacts=artifacts
        )

        with TestClient(
            create_app(
                settings=settings(),
                coordinator=coordinator,
                session_store=session_store,
                portrait_artifacts=artifacts,
            )
        ) as client:
            self.assertIs(artifacts, coordinator.portrait_artifacts)
            self.assertIs(artifacts, session_store.portrait_artifacts)
            self.assertIs(session_store, client.app.state.sessions)

    def test_conflicting_injected_artifact_stores_fail_startup(self) -> None:
        cases = (
            ("explicit and coordinator", "first", None, "second"),
            ("explicit and session store", None, "second", "first"),
            ("coordinator and session store", "first", "second", None),
        )
        for label, coordinator_source, session_source, explicit_source in cases:
            with self.subTest(label=label):
                first = InMemoryPortraitArtifactStore()
                second = InMemoryPortraitArtifactStore()
                sources = {"first": first, "second": second}
                coordinator = FakeCoordinator()
                if coordinator_source is not None:
                    coordinator.portrait_artifacts = sources[coordinator_source]
                session_store = SessionStore(
                    ttl_seconds=60,
                    max_sessions=2,
                    portrait_artifacts=(
                        sources[session_source] if session_source is not None else None
                    ),
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Coordinator, SessionStore, and explicit portrait artifact store "
                    "must share the same instance",
                ):
                    with TestClient(
                        create_app(
                            settings=settings(),
                            coordinator=coordinator,
                            session_store=session_store,
                            portrait_artifacts=(
                                sources[explicit_source]
                                if explicit_source is not None
                                else None
                            ),
                        )
                    ):
                        pass


if __name__ == "__main__":
    unittest.main()
