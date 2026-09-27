from __future__ import annotations

from datetime import timedelta
import unittest

import numpy as np
from fastapi.testclient import TestClient

from kyc_engine import LiveFaceSelection, LivenessEvaluation, LivenessResult
from kyc_engine.intake import ImageIntake
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore
from kyc_api.main import create_app
from kyc_api.sessions import SessionStore, _now

from helpers import AUTH_HEADERS, PNG_BYTES, FakeCoordinator, accept_document, settings


class LiveFaceArtifactSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = InMemoryPortraitArtifactStore()
        self.store = SessionStore(ttl_seconds=60, max_sessions=4, portrait_artifacts=self.artifacts)

    def _start(self) -> str:
        session_id = self.store.create().session_id
        accept_document(self.store, session_id)
        self.store.start_liveness(session_id)
        return session_id

    def _pending(self) -> str:
        return self.artifacts.put(np.full((80, 80, 3), 128, dtype=np.uint8))

    @staticmethod
    def _passed() -> LivenessResult:
        return LivenessResult(True, 0.9, 3, 3)

    def test_claimed_live_face_is_resolved_only_for_a_passed_session_then_deleted(self) -> None:
        session_id = self._start()
        artifact_id = self._pending()
        self.assertIsNotNone(
            self.store.complete_liveness(session_id, self._passed(), artifact_id, True)
        )
        live_face = self.store.resolve_live_face_artifact(session_id)
        self.assertIsNotNone(live_face)
        assert live_face is not None
        self.assertFalse(live_face.flags.writeable)
        self.store.delete(session_id)
        self.assertFalse(self.artifacts.exists(artifact_id))

    def test_failed_and_late_or_duplicate_completion_release_pending_artifacts(self) -> None:
        failed_session = self._start()
        failed = self._pending()
        self.assertIsNotNone(
            self.store.complete_liveness(
                failed_session, LivenessResult(False, 0.1, 3, 0), failed
            )
        )
        self.assertFalse(self.artifacts.exists(failed))
        self.assertIsNone(self.store.resolve_live_face_artifact(failed_session))

        unqualified_session = self._start()
        unqualified = self._pending()
        self.assertIsNotNone(
            self.store.complete_liveness(unqualified_session, self._passed(), unqualified)
        )
        self.assertIsNone(self.store.resolve_live_face_artifact(unqualified_session))
        self.assertFalse(self.artifacts.exists(unqualified))

        session_id = self._start()
        first = self._pending()
        self.assertIsNotNone(self.store.complete_liveness(session_id, self._passed(), first, True))
        duplicate = self._pending()
        self.assertIsNone(self.store.complete_liveness(session_id, self._passed(), duplicate))
        self.assertFalse(self.artifacts.exists(duplicate))

        late_session = self._start()
        late = self._pending()
        self.store.delete(late_session)
        self.assertIsNone(self.store.complete_liveness(late_session, self._passed(), late))
        self.assertFalse(self.artifacts.exists(late))

        expired_session = self._start()
        expired = self._pending()
        self.store._records[expired_session].expires_at = _now() - timedelta(seconds=1)
        self.assertEqual(1, self.store.cleanup())
        self.assertIsNone(self.store.complete_liveness(expired_session, self._passed(), expired))
        self.assertFalse(self.artifacts.exists(expired))

    def test_expiry_releases_a_claimed_live_face(self) -> None:
        session_id = self._start()
        artifact_id = self._pending()
        self.store.complete_liveness(session_id, self._passed(), artifact_id, True)
        self.store._records[session_id].expires_at = _now() - timedelta(seconds=1)
        self.assertEqual(1, self.store.cleanup())
        self.assertFalse(self.artifacts.exists(artifact_id))


class _LiveFaceEvaluator:
    frame_count = 3

    def evaluate_with_live_face(self, _frames) -> LivenessEvaluation:
        return LivenessEvaluation(
            LivenessResult(True, 0.9, 3, 3),
            LiveFaceSelection(
                candidate_frame_count=3,
                eligible_frame_count=1,
                selected_frame_index=1,
                face_detected=True,
                face_count=1,
                quality_score=0.75,
                selection_outcome="selected",
                selected_frame=np.full((240, 320, 3), 128, dtype=np.uint8),
            ),
        )


class LiveFaceArtifactApiTests(unittest.TestCase):
    def test_live_face_artifact_is_never_serialized_by_the_liveness_api(self) -> None:
        with TestClient(
            create_app(
                settings=settings(),
                coordinator=FakeCoordinator(),
                liveness_evaluator=_LiveFaceEvaluator(),
                liveness_intake=ImageIntake(),
            )
        ) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            for side in ("front", "back"):
                self.assertEqual(
                    200,
                    client.post(
                        f"/v1/sessions/{session_id}/images/{side}",
                        headers=AUTH_HEADERS,
                        files={"image": (f"{side}.png", PNG_BYTES, "image/png")},
                    ).status_code,
                )
            response = client.post(
                f"/v1/sessions/{session_id}/liveness",
                headers=AUTH_HEADERS,
                files=[("frames", (f"frame-{index}.png", PNG_BYTES, "image/png")) for index in range(3)],
            )
            self.assertEqual(200, response.status_code)
            payload = response.json()
            for forbidden in ("artifact_id", "image", "pixels", "crop", "bounding_box", "path", "embedding"):
                self.assertNotIn(forbidden, payload["session"]["liveness"])
            retained = client.app.state.sessions.resolve_live_face_artifact(session_id)
            self.assertIsNotNone(retained)
            assert retained is not None
            self.assertEqual((240, 320, 3), retained.shape)


if __name__ == "__main__":
    unittest.main()
