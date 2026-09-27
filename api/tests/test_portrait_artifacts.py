from __future__ import annotations

from datetime import timedelta
import unittest

import numpy as np

from kyc_engine import (
    DocumentExtractionResult,
    KycExtractionResult,
    PortraitExtractionResult,
    PortraitStatus,
    ProcessingStatus,
)
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore

from helpers import PNG_BYTES, accept_document
from kyc_api.models import VerificationStatus
from kyc_api.sessions import SessionStore, _now


def _result(artifact_id: str, *, status: ProcessingStatus = ProcessingStatus.SUCCESS, eligible: bool = True) -> DocumentExtractionResult:
    portrait = PortraitExtractionResult(
        status=PortraitStatus.AVAILABLE if eligible else PortraitStatus.FACE_NOT_FOUND,
        requested_region=None,
        clamped_region=None,
        face_detected=eligible,
        face_count=1 if eligible else 0,
        face=None,
        eligible_for_face_match=eligible,
        artifact_id=artifact_id,
    )
    front = KycExtractionResult(
        schema_version="1.2",
        processing_id="safe-processing-id",
        status=status,
        document_type="ao_id_card",
        side="front",
        profile_id="ao_id_card/front/v1",
        detection=None,
        fields={},
        issues=(),
        timings_ms={},
        portrait=portrait,
    )
    return DocumentExtractionResult(
        schema_version="1.1",
        status=status,
        front=front,
        back=None,
        issues=(),
        timings_ms={},
    )


class PortraitArtifactSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = InMemoryPortraitArtifactStore()
        self.store = SessionStore(ttl_seconds=60, max_sessions=4, portrait_artifacts=self.artifacts)

    def _start(self) -> tuple[str, str]:
        session_id = self.store.create().session_id
        accept_document(self.store, session_id)
        job_id = self.store.queue_document_processing(session_id).document.job_id
        assert job_id is not None
        self.store.start_document_processing(session_id, job_id)
        return session_id, job_id

    def _pending(self) -> str:
        return self.artifacts.put(np.zeros((160, 160, 3), dtype=np.uint8))

    def test_claimed_artifact_is_retrievable_even_when_not_face_eligible_then_deleted(self) -> None:
        session_id, job_id = self._start()
        artifact_id = self._pending()
        completed = self.store.complete_document_processing(
            session_id, job_id, _result(artifact_id, eligible=False)
        )
        self.assertIsNotNone(completed)
        self.assertIsNotNone(self.store.resolve_portrait_artifact(session_id))
        self.assertIsNone(self.store.resolve_face_match_portrait(session_id))
        self.store.delete(session_id)
        self.assertFalse(self.artifacts.exists(artifact_id))

    def test_matcher_resolver_requires_eligibility_and_an_active_session(self) -> None:
        session_id, job_id = self._start()
        artifact_id = self._pending()
        self.assertIsNotNone(
            self.store.complete_document_processing(session_id, job_id, _result(artifact_id))
        )
        portrait = self.store.resolve_face_match_portrait(session_id)
        self.assertIsNotNone(portrait)
        assert portrait is not None
        self.assertFalse(portrait.flags.writeable)

        self.store._records[session_id].status = VerificationStatus.REJECTED
        self.assertIsNone(self.store.resolve_face_match_portrait(session_id))

    def test_expiry_releases_claimed_artifact(self) -> None:
        session_id, job_id = self._start()
        artifact_id = self._pending()
        self.store.complete_document_processing(session_id, job_id, _result(artifact_id))
        self.store._records[session_id].expires_at = _now() - timedelta(seconds=1)
        self.assertEqual(1, self.store.cleanup())
        self.assertFalse(self.artifacts.exists(artifact_id))

    def test_timeout_and_discarded_late_completion_release_pending_artifact(self) -> None:
        session_id, job_id = self._start()
        artifact_id = self._pending()
        self.assertIsNotNone(self.store.timeout_document_processing(session_id, job_id))
        self.assertIsNone(self.store.complete_document_processing(session_id, job_id, _result(artifact_id)))
        self.artifacts.release_pending((artifact_id,))
        self.assertFalse(self.artifacts.exists(artifact_id))

    def test_delete_before_completion_and_duplicate_completion_cannot_strand_artifacts(self) -> None:
        session_id, job_id = self._start()
        pending = self._pending()
        self.store.delete(session_id)
        self.assertIsNone(self.store.complete_document_processing(session_id, job_id, _result(pending)))
        self.artifacts.release_pending((pending,))
        self.assertFalse(self.artifacts.exists(pending))

        session_id, job_id = self._start()
        first = self._pending()
        self.assertIsNotNone(self.store.complete_document_processing(session_id, job_id, _result(first)))
        duplicate = self._pending()
        self.assertIsNone(self.store.complete_document_processing(session_id, job_id, _result(duplicate)))
        self.artifacts.release_pending((duplicate,))
        self.assertTrue(self.artifacts.exists(first))
        self.assertFalse(self.artifacts.exists(duplicate))
        self.store.delete(session_id)
        self.assertFalse(self.artifacts.exists(first))

    def test_failed_pipeline_result_releases_pending_artifact_instead_of_claiming(self) -> None:
        session_id, job_id = self._start()
        artifact_id = self._pending()
        self.assertIsNotNone(
            self.store.complete_document_processing(
                session_id, job_id, _result(artifact_id, status=ProcessingStatus.FAILED)
            )
        )
        self.assertFalse(self.artifacts.exists(artifact_id))

    def test_claim_failure_marks_document_failed_and_releases_artifact(self) -> None:
        session_id, job_id = self._start()
        artifact_id = self._pending()
        self.artifacts.release_pending((artifact_id,))
        completed = self.store.complete_document_processing(session_id, job_id, _result(artifact_id))
        assert completed is not None
        self.assertEqual("failed", completed.document.status.value)
        self.assertEqual("PORTRAIT_ARTIFACT_CLAIM_FAILED", completed.document.error_code)
        self.assertFalse(self.artifacts.exists(artifact_id))


if __name__ == "__main__":
    unittest.main()
