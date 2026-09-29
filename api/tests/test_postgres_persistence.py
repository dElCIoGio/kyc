from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
import os
from threading import Barrier
import unittest

import numpy as np
from sqlalchemy import delete, func, select, text, update
from fastapi.testclient import TestClient

from kyc_api.main import create_app
from kyc_api.application.events.lifecycle import WebhookEvent
from kyc_api.application.jobs.manager import JobManager
from kyc_api.application.sessions.store import SessionCapacityExceeded, SessionConflict, SessionExpired, SessionNotFound, SessionStore, _now
from kyc_api.application.security.browser_credentials import BrowserCredentialError
from kyc_api.infrastructure.persistence.browser_credentials import PostgresBrowserCredentialStore
from kyc_api.infrastructure.persistence.database import create_database_engine, verify_database
from kyc_api.infrastructure.persistence.models import (
    BrowserCredential,
    SessionTombstone,
    VerificationSession,
    WebhookEventRow,
)
from kyc_api.infrastructure.persistence.postgres_sessions import _ArtifactTransaction, PostgresSessionStore
from kyc_api.infrastructure.persistence.webhook_outbox import PostgresWebhookOutbox
from kyc_api.models import DocumentSide, DocumentStatus, FaceMatchStatus, NifVerificationStatus
from kyc_engine import LivenessResult, PortraitExtractionResult, PortraitStatus
from kyc_engine.contracts import ExtractedField, FieldStatus
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore

from helpers import AUTH_HEADERS, PNG_BYTES, FakeCoordinator, accepted_capture_assessment, extraction_result, settings


_URL = os.environ.get("KYC_TEST_DATABASE_URL")


class _InlineExecutor:
    def submit(self, callback, *args):
        callback(*args)

    def shutdown(self, **_kwargs) -> None:
        pass


@unittest.skipUnless(_URL, "KYC_TEST_DATABASE_URL is required for PostgreSQL integration tests")
class PostgresPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert _URL is not None
        cls.engine = create_database_engine(_URL)
        verify_database(cls.engine, _URL)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.dispose()

    def setUp(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(WebhookEventRow))
            connection.execute(delete(BrowserCredential))
            connection.execute(delete(VerificationSession))
            connection.execute(delete(SessionTombstone))

    def store(self, **overrides) -> PostgresSessionStore:
        values = {
            "engine": self.engine,
            "ttl_seconds": 60,
            "max_sessions": 8,
            "liveness_required": False,
        }
        values.update(overrides)
        return PostgresSessionStore(**values)

    @staticmethod
    def _portrait_result(artifact_id: str):
        source = extraction_result()
        assert source.front is not None
        portrait = PortraitExtractionResult(
            status=PortraitStatus.AVAILABLE,
            requested_region=None,
            clamped_region=None,
            face_detected=True,
            face_count=1,
            face=None,
            eligible_for_face_match=True,
            artifact_id=artifact_id,
        )
        return replace(source, front=replace(source.front, portrait=portrait))

    def _running_document(self, store: PostgresSessionStore) -> tuple[str, str]:
        session_id = store.create().session_id
        for side in (DocumentSide.FRONT, DocumentSide.BACK):
            store.accept_document_capture(session_id, side, PNG_BYTES, accepted_capture_assessment())
        queued = store.queue_document_processing(session_id)
        assert queued.document.job_id is not None
        store.start_document_processing(session_id, queued.document.job_id)
        return session_id, queued.document.job_id

    def _postgres_events(self, session_id: str) -> list[tuple[str, int]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(WebhookEventRow.event_type, WebhookEventRow.sequence)
                .where(WebhookEventRow.session_id == session_id)
                .order_by(WebhookEventRow.sequence, WebhookEventRow.event_type)
            )
            return [(row.event_type, row.sequence) for row in rows]

    @staticmethod
    def _memory_publisher(events: list[tuple[str, int]]):
        def publish(snapshot, reason: str) -> None:
            event = WebhookEvent.from_snapshot(snapshot, transition_reason=reason)
            if event is not None:
                events.append((event.event_type, event.sequence))
        return publish

    @staticmethod
    def _run_document_workflow(store, *, coordinator: FakeCoordinator, publisher=None):
        session_id = store.create().session_id
        if publisher is not None:
            publisher(store.get(session_id), "verification.session.created")
        for side in (DocumentSide.FRONT, DocumentSide.BACK):
            store.accept_document_capture(session_id, side, PNG_BYTES, accepted_capture_assessment())
        jobs = JobManager(
            coordinator, store, workers=1, capacity=1, executor=_InlineExecutor(), webhook_publisher=publisher
        )
        jobs.submit_document_processing(session_id)
        return session_id

    def test_document_lifecycle_webhook_events_match_memory_publishers(self) -> None:
        """The actual memory JobManager publisher is the event compatibility oracle."""
        from kyc_engine import ProcessingStatus

        for status in (ProcessingStatus.SUCCESS, ProcessingStatus.PARTIAL, ProcessingStatus.FAILED):
            memory_events: list[tuple[str, int]] = []
            memory = SessionStore(ttl_seconds=60, max_sessions=8, liveness_required=False)
            self._run_document_workflow(
                memory, coordinator=FakeCoordinator(status=status), publisher=self._memory_publisher(memory_events)
            )
            postgres = self.store(webhook_enabled=True)
            session_id = self._run_document_workflow(postgres, coordinator=FakeCoordinator(status=status))
            self.assertEqual(memory_events, self._postgres_events(session_id), status.value)

        memory_events = []
        memory = SessionStore(ttl_seconds=60, max_sessions=8, liveness_required=False)
        self._run_document_workflow(
            memory, coordinator=FakeCoordinator(fail=True), publisher=self._memory_publisher(memory_events)
        )
        postgres = self.store(webhook_enabled=True)
        session_id = self._run_document_workflow(postgres, coordinator=FakeCoordinator(fail=True))
        self.assertEqual(memory_events, self._postgres_events(session_id), "explicit failure")

    def test_document_artifact_claim_is_compensated_when_outbox_write_fails(self) -> None:
        artifacts = InMemoryPortraitArtifactStore()
        store = self.store(webhook_enabled=True, portrait_artifacts=artifacts)
        session_id, job_id = self._running_document(store)
        artifact_id = artifacts.put(np.zeros((64, 64, 3), dtype=np.uint8))
        original = store._insert_events
        store._insert_events = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("outbox failed"))  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "outbox failed"):
            store.complete_document_processing(session_id, job_id, self._portrait_result(artifact_id))
        store._insert_events = original  # type: ignore[method-assign]
        self.assertEqual(DocumentStatus.PROCESSING, store.get(session_id).document.status)
        self.assertTrue(artifacts.exists(artifact_id))
        self.assertIsNone(artifacts.get(artifact_id, session_id=session_id))
        self.assertIsNotNone(store.complete_document_processing(session_id, job_id, self._portrait_result(artifact_id)))
        self.assertIsNotNone(artifacts.get(artifact_id, session_id=session_id))

    def test_liveness_artifact_claim_is_compensated_when_outbox_write_fails(self) -> None:
        artifacts = InMemoryPortraitArtifactStore()
        store = self.store(webhook_enabled=True, liveness_required=True, portrait_artifacts=artifacts)
        session_id, job_id = self._running_document(store)
        store.complete_document_processing(session_id, job_id, extraction_result())
        store.start_liveness(session_id)
        artifact_id = artifacts.put(np.zeros((64, 64, 3), dtype=np.uint8))
        original = store._insert_events
        store._insert_events = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("outbox failed"))  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "outbox failed"):
            store.complete_liveness(session_id, LivenessResult(True, 0.9, 3, 3), artifact_id, True)
        store._insert_events = original  # type: ignore[method-assign]
        self.assertEqual("processing", store.get(session_id).liveness_status.value)
        self.assertTrue(artifacts.exists(artifact_id))
        self.assertIsNone(artifacts.get(artifact_id, session_id=session_id))
        self.assertIsNotNone(store.complete_liveness(session_id, LivenessResult(True, 0.9, 3, 3), artifact_id, True))
        self.assertIsNotNone(artifacts.get(artifact_id, session_id=session_id))

    def test_runtime_only_face_artifact_cleanup_does_not_change_version_or_sequence(self) -> None:
        store = self.store(face_match_enabled=True)
        session_id = store.create().session_id
        with self.engine.begin() as connection:
            connection.execute(
                update(VerificationSession)
                .where(VerificationSession.session_id == session_id)
                .values(face_match_status=FaceMatchStatus.COMPLETED.value)
            )
        with self.engine.connect() as connection:
            before = connection.execute(
                select(VerificationSession.version, VerificationSession.event_sequence)
                .where(VerificationSession.session_id == session_id)
            ).one()
        store.release_face_match_biometric_artifacts(session_id)
        with self.engine.connect() as connection:
            after = connection.execute(
                select(VerificationSession.version, VerificationSession.event_sequence)
                .where(VerificationSession.session_id == session_id)
            ).one()
        self.assertEqual(before, after)

    def test_failed_transaction_does_not_release_owned_live_face(self) -> None:
        artifacts = InMemoryPortraitArtifactStore()
        store = self.store(webhook_enabled=True, liveness_required=True, portrait_artifacts=artifacts)
        session_id, job_id = self._running_document(store)
        store.complete_document_processing(session_id, job_id, extraction_result())
        store.start_liveness(session_id)
        artifact_id = artifacts.put(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertTrue(artifacts.claim((artifact_id,), session_id))
        # This simulates an owned ephemeral face retained by a running worker;
        # it is deliberately not written to the durable row.
        store._runtime[session_id].live_face_artifact_id = artifact_id
        store._runtime[session_id].live_face_eligible = True
        original = store._insert_events
        store._insert_events = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("outbox failed"))  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "outbox failed"):
            store.fail_liveness(session_id, "LIVENESS_EVALUATION_FAILED")
        store._insert_events = original  # type: ignore[method-assign]
        self.assertIsNotNone(artifacts.get(artifact_id, session_id=session_id))
        self.assertEqual("processing", store.get(session_id).liveness_status.value)
        self.assertIsNotNone(store.fail_liveness(session_id, "LIVENESS_EVALUATION_FAILED"))
        self.assertFalse(artifacts.exists(artifact_id))

    def test_document_workflow_and_result_survive_new_store(self) -> None:
        store = self.store()
        session_id = store.create().session_id
        assessment = accepted_capture_assessment()
        first = store.accept_document_capture(session_id, DocumentSide.FRONT, PNG_BYTES, assessment)
        second = store.accept_document_capture(session_id, DocumentSide.BACK, PNG_BYTES, assessment)
        self.assertEqual(1, first.snapshot.event_sequence)
        self.assertEqual(3, second.snapshot.event_sequence)
        queued = store.queue_document_processing(session_id)
        job_id = queued.document.job_id
        assert job_id is not None
        front, back, running = store.start_document_processing(session_id, job_id)
        self.assertEqual(PNG_BYTES, front)
        self.assertEqual(PNG_BYTES, back)
        self.assertEqual(5, running.event_sequence)
        completed = store.complete_document_processing(session_id, job_id, extraction_result())
        assert completed is not None
        self.assertEqual("completed", completed.verification_status.value)

        restarted = self.store()
        self.assertEqual(completed, restarted.get(session_id))
        self.assertEqual(extraction_result().to_dict(), restarted.result(session_id).to_dict())

    def test_memory_and_postgres_follow_the_same_core_contract(self) -> None:
        memory = SessionStore(ttl_seconds=60, max_sessions=8, liveness_required=False)
        postgres = self.store()

        def workflow(store):
            snapshots = []
            session_id = store.create().session_id
            snapshots.append(store.get(session_id))
            snapshots.append(store.accept_document_capture(session_id, DocumentSide.FRONT, PNG_BYTES, accepted_capture_assessment()).snapshot)
            snapshots.append(store.accept_document_capture(session_id, DocumentSide.BACK, PNG_BYTES, accepted_capture_assessment()).snapshot)
            queued = store.queue_document_processing(session_id)
            snapshots.append(queued)
            assert queued.document.job_id is not None
            snapshots.append(store.start_document_processing(session_id, queued.document.job_id)[-1])
            completed = store.complete_document_processing(session_id, queued.document.job_id, extraction_result())
            assert completed is not None
            snapshots.append(completed)
            self.assertIsNone(store.complete_document_processing(session_id, queued.document.job_id, extraction_result()))
            return session_id, [
                (
                    item.verification_status,
                    item.document.status,
                    item.document.front_capture,
                    item.document.back_capture,
                    item.liveness_status,
                    item.face_match_status,
                    item.event_sequence,
                )
                for item in snapshots
            ]

        memory_id, memory_states = workflow(memory)
        postgres_id, postgres_states = workflow(postgres)
        self.assertEqual(memory_states, postgres_states)
        memory.delete(memory_id)
        postgres.delete(postgres_id)
        for store, session_id in ((memory, memory_id), (postgres, postgres_id)):
            with self.assertRaises(SessionNotFound):
                store.get(session_id)

    def test_liveness_nif_timeout_and_stale_completion_paths(self) -> None:
        live = self.store(liveness_required=True)
        live_id = live.create().session_id
        for side in (DocumentSide.FRONT, DocumentSide.BACK):
            live.accept_document_capture(live_id, side, PNG_BYTES, accepted_capture_assessment())
        queued = live.queue_document_processing(live_id)
        assert queued.document.job_id is not None
        live.start_document_processing(live_id, queued.document.job_id)
        live.complete_document_processing(live_id, queued.document.job_id, extraction_result())
        live.start_liveness(live_id)
        completed = live.complete_liveness(live_id, LivenessResult(True, 0.9, 3, 3))
        assert completed is not None
        self.assertEqual("completed", completed.verification_status.value)

        timed = self.store()
        timed_id = timed.create().session_id
        for side in (DocumentSide.FRONT, DocumentSide.BACK):
            timed.accept_document_capture(timed_id, side, PNG_BYTES, accepted_capture_assessment())
        queued = timed.queue_document_processing(timed_id)
        assert queued.document.job_id is not None
        timed.start_document_processing(timed_id, queued.document.job_id)
        self.assertIsNotNone(timed.timeout_document_processing(timed_id, queued.document.job_id))
        self.assertIsNone(timed.complete_document_processing(timed_id, queued.document.job_id, extraction_result()))

        nif = self.store(nif_verification_enabled=True)
        nif_id = nif.create().session_id
        for side in (DocumentSide.FRONT, DocumentSide.BACK):
            nif.accept_document_capture(nif_id, side, PNG_BYTES, accepted_capture_assessment())
        queued = nif.queue_document_processing(nif_id)
        assert queued.document.job_id is not None
        nif.start_document_processing(nif_id, queued.document.job_id)
        nif.complete_document_processing(nif_id, queued.document.job_id, extraction_result())
        skipped = nif.claim_nif_verification(nif_id)
        assert skipped is not None
        self.assertEqual("", skipped[0])
        self.assertEqual("completed", skipped[-1].verification_status.value)

    def test_expiry_tombstone_then_not_found_and_explicit_delete(self) -> None:
        store = self.store(ttl_seconds=2)
        session_id = store.create().session_id
        with self.engine.begin() as connection:
            connection.execute(
                update(VerificationSession)
                .where(VerificationSession.session_id == session_id)
                .values(expires_at=_now() - timedelta(seconds=1))
            )
        self.assertEqual(1, store.cleanup())
        with self.assertRaises(SessionExpired):
            store.get(session_id)
        with self.engine.begin() as connection:
            connection.execute(
                update(SessionTombstone)
                .where(SessionTombstone.session_id == session_id)
                .values(retained_until=_now() - timedelta(seconds=1))
            )
        store.cleanup()
        with self.assertRaises(SessionNotFound):
            store.get(session_id)

        another = store.create().session_id
        credentials = PostgresBrowserCredentialStore(engine=self.engine, sessions=store, ttl_seconds=60)
        credentials.issue(another)
        store.delete(another)
        with self.engine.connect() as connection:
            self.assertEqual(0, connection.scalar(select(func.count()).select_from(BrowserCredential)))

    def test_expiry_cleanup_racing_capture_cannot_resurrect_a_session(self) -> None:
        first = self.store(webhook_enabled=True)
        session_id = first.create().session_id
        credentials = PostgresBrowserCredentialStore(engine=self.engine, sessions=first, ttl_seconds=60)
        token, _ = credentials.issue(session_id)
        with self.engine.begin() as connection:
            connection.execute(
                update(VerificationSession)
                .where(VerificationSession.session_id == session_id)
                .values(expires_at=_now() - timedelta(seconds=1))
            )
        second = self.store(webhook_enabled=True)
        barrier = Barrier(2)

        def expire() -> int:
            barrier.wait()
            return first.cleanup()

        def mutate() -> object:
            barrier.wait()
            try:
                return second.accept_document_capture(
                    session_id, DocumentSide.FRONT, PNG_BYTES, accepted_capture_assessment()
                )
            except SessionExpired:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            expired, mutation = list(executor.map(lambda action: action(), (expire, mutate)))
        self.assertIn(expired, {0, 1})  # Either caller may perform the expiry under the row lock.
        self.assertIsNone(mutation)
        with self.engine.connect() as connection:
            self.assertEqual(0, connection.scalar(select(func.count()).select_from(VerificationSession).where(VerificationSession.session_id == session_id)))
            self.assertEqual(1, connection.scalar(select(func.count()).select_from(SessionTombstone).where(SessionTombstone.session_id == session_id)))
            self.assertEqual(0, connection.scalar(select(func.count()).select_from(WebhookEventRow).where(WebhookEventRow.session_id == session_id)))
        with self.assertRaises(SessionExpired):
            second.get(session_id)
        self.assertEqual(64, len(credentials.authorize(token, session_id, safe_read=True)))
        with self.engine.begin() as connection:
            connection.execute(
                update(SessionTombstone)
                .where(SessionTombstone.session_id == session_id)
                .values(retained_until=_now() - timedelta(seconds=1))
            )
        first.cleanup()
        credentials.cleanup()
        with self.assertRaises(SessionNotFound):
            second.get(session_id)
        with self.engine.connect() as connection:
            self.assertEqual(0, connection.scalar(select(func.count()).select_from(BrowserCredential).where(BrowserCredential.session_id == session_id)))

    def test_failure_inserts_two_same_sequence_events(self) -> None:
        store = self.store(webhook_enabled=True)
        session_id = store.create().session_id
        for side in (DocumentSide.FRONT, DocumentSide.BACK):
            store.accept_document_capture(session_id, side, PNG_BYTES, accepted_capture_assessment())
        queued = store.queue_document_processing(session_id)
        assert queued.document.job_id is not None
        store.start_document_processing(session_id, queued.document.job_id)
        failed = store.fail_document_processing(session_id, queued.document.job_id, "PROCESSING_FAILED")
        assert failed is not None
        with self.engine.connect() as connection:
            rows = list(connection.execute(
                select(WebhookEventRow.sequence, WebhookEventRow.event_type)
                .where(WebhookEventRow.session_id == session_id, WebhookEventRow.sequence == failed.event_sequence)
            ))
        self.assertEqual(2, len(rows))
        self.assertEqual({failed.event_sequence}, {row.sequence for row in rows})

    def test_recovery_matrix_for_resumable_and_interrupted_work(self) -> None:
        def processed(store, result=None):
            session_id = store.create().session_id
            for side in (DocumentSide.FRONT, DocumentSide.BACK):
                store.accept_document_capture(session_id, side, PNG_BYTES, accepted_capture_assessment())
            queued = store.queue_document_processing(session_id)
            assert queued.document.job_id is not None
            store.start_document_processing(session_id, queued.document.job_id)
            store.complete_document_processing(session_id, queued.document.job_id, result or extraction_result())
            return session_id

        untouched_store = self.store(liveness_required=True)
        untouched = untouched_store.create().session_id
        resumable = processed(untouched_store)
        restarted = self.store(liveness_required=True)
        self.assertEqual(0, restarted.recover())
        self.assertEqual("awaiting_capture", restarted.get(untouched).document.status.value)
        self.assertEqual("in_progress", restarted.get(resumable).verification_status.value)

        processing_store = self.store(liveness_required=True)
        processing = processing_store.create().session_id
        for side in (DocumentSide.FRONT, DocumentSide.BACK):
            processing_store.accept_document_capture(processing, side, PNG_BYTES, accepted_capture_assessment())
        processing_store.queue_document_processing(processing)
        self.assertEqual(1, self.store(liveness_required=True).recover())
        self.assertEqual("failed", self.store(liveness_required=True).get(processing).verification_status.value)

        live_store = self.store(liveness_required=True)
        live_processing = processed(live_store)
        live_store.start_liveness(live_processing)
        self.assertEqual(1, self.store(liveness_required=True).recover())
        self.assertEqual("failed", self.store(liveness_required=True).get(live_processing).liveness_status.value)

        face_store = self.store(liveness_required=True, face_match_enabled=True)
        face_pending = processed(face_store)
        face_store.start_liveness(face_pending)
        face_store.complete_liveness(face_pending, LivenessResult(True, 0.9, 3, 3))
        self.assertEqual(1, self.store(liveness_required=True, face_match_enabled=True).recover())
        self.assertEqual("failed", self.store(liveness_required=True, face_match_enabled=True).get(face_pending).face_match_status.value)

    def test_nif_recovery_is_unavailable_idempotent_and_emits_once(self) -> None:
        source = extraction_result()
        assert source.front is not None
        fields = {
            "id_number": ExtractedField("id_number", FieldStatus.VALID, "123", "123", 0.99, None),
            "full_name": ExtractedField("full_name", FieldStatus.VALID, "Test Name", "Test Name", 0.99, None),
        }
        applicable = replace(source, front=replace(source.front, fields=fields))
        store = self.store(liveness_required=True, nif_verification_enabled=True, webhook_enabled=True)
        session_id = store.create().session_id
        for side in (DocumentSide.FRONT, DocumentSide.BACK):
            store.accept_document_capture(session_id, side, PNG_BYTES, accepted_capture_assessment())
        queued = store.queue_document_processing(session_id)
        assert queued.document.job_id is not None
        store.start_document_processing(session_id, queued.document.job_id)
        store.complete_document_processing(session_id, queued.document.job_id, applicable)
        claim = store.claim_nif_verification(session_id)
        assert claim is not None
        self.assertEqual("123", claim[0])

        restarted = self.store(liveness_required=True, nif_verification_enabled=True, webhook_enabled=True)
        self.assertEqual(1, restarted.recover())
        snapshot = restarted.get(session_id)
        self.assertEqual(NifVerificationStatus.UNAVAILABLE, snapshot.nif_verification_status)
        self.assertEqual(0, restarted.recover())
        with self.engine.connect() as connection:
            count = connection.scalar(
                select(func.count()).select_from(WebhookEventRow).where(
                    WebhookEventRow.session_id == session_id,
                    WebhookEventRow.event_type == "verification.nif.completed",
                )
            )
        self.assertEqual(1, count)

    def test_face_claim_is_atomic_and_similarity_is_ephemeral(self) -> None:
        first = self.store(face_match_enabled=True)
        session_id = first.create().session_id
        with self.engine.begin() as connection:
            connection.execute(
                update(VerificationSession)
                .where(VerificationSession.session_id == session_id)
                .values(face_match_status=FaceMatchStatus.READY.value)
            )
        second = self.store(face_match_enabled=True)
        barrier = Barrier(2)

        def claim(store):
            barrier.wait()
            try:
                return store.start_face_match(session_id)
            except SessionConflict:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(claim, (first, second)))
        self.assertEqual(1, sum(item is not None for item in claims))
        winner = first if claims[0] is not None else second
        winner.complete_face_match(session_id, 0.75)
        self.assertEqual(0.75, winner.face_match_result(session_id).similarity)
        restarted = self.store(face_match_enabled=True)
        self.assertEqual(FaceMatchStatus.COMPLETED, restarted.face_match_result(session_id).status)
        self.assertIsNone(restarted.face_match_result(session_id).similarity)

    def test_concurrent_nif_claim_and_duplicate_completion_are_single_mutations(self) -> None:
        source = extraction_result()
        assert source.front is not None
        applicable = replace(source, front=replace(source.front, fields={
            "id_number": ExtractedField("id_number", FieldStatus.VALID, "123", "123", 0.99, None),
        }))
        first = self.store(nif_verification_enabled=True, webhook_enabled=True)
        session_id, job_id = self._running_document(first)
        first.complete_document_processing(session_id, job_id, applicable)
        second = self.store(nif_verification_enabled=True, webhook_enabled=True)
        with self.engine.connect() as connection:
            before = connection.execute(select(VerificationSession.version, VerificationSession.event_sequence).where(VerificationSession.session_id == session_id)).one()
        barrier = Barrier(2)

        def claim(store):
            barrier.wait()
            return store.claim_nif_verification(session_id)

        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(claim, (first, second)))
        self.assertEqual(1, sum(claim is not None for claim in claims))
        with self.engine.connect() as connection:
            claimed = connection.execute(select(VerificationSession.version, VerificationSession.event_sequence, VerificationSession.nif_status).where(VerificationSession.session_id == session_id)).one()
        self.assertEqual(before.version + 1, claimed.version)
        self.assertEqual(before.event_sequence + 1, claimed.event_sequence)
        self.assertEqual(NifVerificationStatus.PROCESSING.value, claimed.nif_status)

        barrier = Barrier(2)

        def complete(store):
            barrier.wait()
            return store.complete_nif_verification(
                session_id, status=NifVerificationStatus.VERIFIED, source="minfin", name_match=True
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            completions = list(executor.map(complete, (first, second)))
        self.assertEqual(1, sum(completion is not None for completion in completions))
        with self.engine.connect() as connection:
            completed = connection.execute(select(VerificationSession.version, VerificationSession.event_sequence).where(VerificationSession.session_id == session_id)).one()
            events = connection.scalar(select(func.count()).select_from(WebhookEventRow).where(
                WebhookEventRow.session_id == session_id,
                WebhookEventRow.event_type == "verification.nif.completed",
            ))
        self.assertEqual(claimed.version + 1, completed.version)
        self.assertEqual(claimed.event_sequence + 1, completed.event_sequence)
        self.assertEqual(1, events)

    def test_duplicate_face_completion_and_document_timeout_races_have_one_winner(self) -> None:
        artifacts = InMemoryPortraitArtifactStore()
        first = self.store(
            liveness_required=True,
            face_match_enabled=True,
            webhook_enabled=True,
            portrait_artifacts=artifacts,
        )
        session_id, job_id = self._running_document(first)
        portrait_id = artifacts.put(np.zeros((64, 64, 3), dtype=np.uint8))
        first.complete_document_processing(session_id, job_id, self._portrait_result(portrait_id))
        first.start_liveness(session_id)
        live_face_id = artifacts.put(np.ones((64, 64, 3), dtype=np.uint8))
        first.complete_liveness(
            session_id, LivenessResult(True, 0.9, 3, 3), live_face_id, True
        )
        first.start_face_match(session_id)
        second = self.store(
            liveness_required=True,
            face_match_enabled=True,
            webhook_enabled=True,
            portrait_artifacts=artifacts,
        )
        barrier = Barrier(2)

        def complete_face(store):
            barrier.wait()
            return store.complete_face_match(session_id, 0.75)

        with ThreadPoolExecutor(max_workers=2) as executor:
            completed = list(executor.map(complete_face, (first, second)))
        self.assertEqual(1, sum(item is not None for item in completed))
        self.assertEqual(FaceMatchStatus.COMPLETED, first.get(session_id).face_match_status)

        racing = self.store(webhook_enabled=True)
        race_id, race_job = self._running_document(racing)
        rival = self.store(webhook_enabled=True)
        barrier = Barrier(2)

        def finish():
            barrier.wait()
            return racing.complete_document_processing(race_id, race_job, extraction_result())

        def timeout():
            barrier.wait()
            return rival.timeout_document_processing(race_id, race_job)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda operation: operation(), (finish, timeout)))
        self.assertEqual(1, sum(item is not None for item in results))
        final = racing.get(race_id)
        self.assertIn(final.document.status, {DocumentStatus.PASSED, DocumentStatus.FAILED})
        self.assertEqual(race_job, final.document.job_id)

    def test_concurrent_browser_rotation_leaves_one_active_digest(self) -> None:
        sessions = self.store()
        session_id = sessions.create().session_id
        first = PostgresBrowserCredentialStore(engine=self.engine, sessions=sessions, ttl_seconds=60)
        second = PostgresBrowserCredentialStore(engine=self.engine, sessions=sessions, ttl_seconds=60)
        barrier = Barrier(2)

        def issue(credentials):
            barrier.wait()
            return credentials.issue(session_id)[0]

        with ThreadPoolExecutor(max_workers=2) as executor:
            tokens = list(executor.map(issue, (first, second)))
        accepted = 0
        for token in tokens:
            try:
                first.authorize(token, session_id, safe_read=True)
                accepted += 1
            except BrowserCredentialError:
                pass
        self.assertEqual(1, accepted)
        with self.engine.connect() as connection:
            self.assertEqual(1, connection.scalar(select(func.count()).select_from(BrowserCredential)))

    def test_browser_token_preserves_safe_read_410_then_cleans_up(self) -> None:
        sessions = self.store(ttl_seconds=2)
        session_id = sessions.create().session_id
        credentials = PostgresBrowserCredentialStore(engine=self.engine, sessions=sessions, ttl_seconds=60)
        token, _ = credentials.issue(session_id)
        with self.engine.begin() as connection:
            connection.execute(
                update(VerificationSession)
                .where(VerificationSession.session_id == session_id)
                .values(expires_at=_now() - timedelta(seconds=1))
            )
        sessions.cleanup()
        self.assertEqual(64, len(credentials.authorize(token, session_id, safe_read=True)))
        with self.assertRaises(SessionExpired):
            credentials.authorize(token, session_id, safe_read=False)
        with self.engine.begin() as connection:
            connection.execute(
                update(SessionTombstone)
                .where(SessionTombstone.session_id == session_id)
                .values(retained_until=_now() - timedelta(seconds=1))
            )
        sessions.cleanup()
        credentials.cleanup()
        with self.engine.connect() as connection:
            self.assertEqual(0, connection.scalar(select(func.count()).select_from(BrowserCredential)))

    def test_capacity_is_serialized_across_store_instances(self) -> None:
        first = self.store(max_sessions=1)
        second = self.store(max_sessions=1)
        barrier = Barrier(2)

        def create(store):
            barrier.wait()
            try:
                return store.create().session_id
            except SessionCapacityExceeded:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(create, (first, second)))
        self.assertEqual(1, sum(value is not None for value in results))
        with self.engine.connect() as connection:
            self.assertEqual(1, connection.scalar(select(func.count()).select_from(VerificationSession)))

    def test_recovery_fails_accepted_capture_once(self) -> None:
        store = self.store(liveness_required=True)
        session_id = store.create().session_id
        store.accept_document_capture(session_id, DocumentSide.FRONT, PNG_BYTES, accepted_capture_assessment())

        restarted = self.store(liveness_required=True)
        self.assertEqual(1, restarted.recover())
        snapshot = restarted.get(session_id)
        self.assertEqual("failed", snapshot.verification_status.value)
        self.assertEqual("failed", snapshot.document.status.value)
        self.assertEqual(0, restarted.recover())

    def test_state_and_webhook_commit_and_rollback_together(self) -> None:
        store = self.store(webhook_enabled=True)
        snapshot = store.create()
        with self.engine.connect() as connection:
            sequence = connection.scalar(select(WebhookEventRow.sequence).where(WebhookEventRow.session_id == snapshot.session_id))
            self.assertEqual(snapshot.event_sequence, sequence)

        self.setUp()
        failing = self.store(webhook_enabled=True)
        original = failing._insert_events

        def explode(*_args, **_kwargs):
            raise RuntimeError("outbox insertion failed")

        failing._insert_events = explode  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            failing.create()
        failing._insert_events = original  # type: ignore[method-assign]
        with self.engine.connect() as connection:
            self.assertEqual(0, connection.scalar(select(func.count()).select_from(VerificationSession)))
            self.assertEqual(0, connection.scalar(select(func.count()).select_from(WebhookEventRow)))

        self.setUp()
        mutating = self.store(webhook_enabled=True)
        session_id = mutating.create().session_id
        for side in (DocumentSide.FRONT, DocumentSide.BACK):
            mutating.accept_document_capture(
                session_id, side, PNG_BYTES, accepted_capture_assessment()
            )
        queued = mutating.queue_document_processing(session_id)
        assert queued.document.job_id is not None
        running = mutating.start_document_processing(
            session_id, queued.document.job_id
        )[-1]
        prior_sequence = running.event_sequence
        original = mutating._insert_events
        mutating._insert_events = explode  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            mutating.fail_document_processing(
                session_id, queued.document.job_id, "PROCESSING_FAILED"
            )
        mutating._insert_events = original  # type: ignore[method-assign]
        unchanged = mutating.get(session_id)
        self.assertEqual("processing", unchanged.document.status.value)
        self.assertEqual(prior_sequence, unchanged.event_sequence)
        with self.engine.connect() as connection:
            self.assertEqual(
                0,
                connection.scalar(
                    select(func.count()).select_from(WebhookEventRow).where(
                        WebhookEventRow.session_id == session_id,
                        WebhookEventRow.sequence > prior_sequence,
                    )
                ),
            )

    def test_browser_credential_survives_store_recreation_without_raw_token(self) -> None:
        sessions = self.store()
        session_id = sessions.create().session_id
        credentials = PostgresBrowserCredentialStore(engine=self.engine, sessions=sessions, ttl_seconds=120)
        token, _ = credentials.issue(session_id)

        restarted_sessions = self.store()
        restarted = PostgresBrowserCredentialStore(engine=self.engine, sessions=restarted_sessions, ttl_seconds=120)
        self.assertEqual(64, len(restarted.authorize(token, session_id, safe_read=True)))
        with self.engine.connect() as connection:
            stored = connection.scalar(select(BrowserCredential.digest).where(BrowserCredential.session_id == session_id))
        self.assertNotEqual(token.encode(), stored)

    def test_concurrent_outbox_claim_has_one_owner_and_lease_can_be_reclaimed(self) -> None:
        store = self.store(webhook_enabled=True)
        store.create()
        first = PostgresWebhookOutbox(self.engine, retention_seconds=60, lease_seconds=60)
        second = PostgresWebhookOutbox(self.engine, retention_seconds=60, lease_seconds=60)
        barrier = Barrier(2)

        def claim(outbox):
            barrier.wait()
            return outbox.due()

        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(claim, (first, second)))
        self.assertEqual(1, sum(item is not None for item in claims))

        with self.engine.begin() as connection:
            connection.execute(update(WebhookEventRow).values(lease_expires_at=func.now() - text("INTERVAL '1 second'")))
        self.assertIsNotNone(second.due())

    def test_composed_postgres_app_uses_only_atomic_webhook_path(self) -> None:
        class InjectedDispatcher:
            def __init__(self) -> None:
                self.enqueued = 0

            def enqueue(self, *_args) -> None:
                self.enqueued += 1

            def shutdown(self) -> None:
                pass

        dispatcher = InjectedDispatcher()
        configured = settings(
            session_backend="postgres",
            database_url=_URL,
            webhook_url="https://example.invalid/webhooks",
            webhook_secret="a" * 32,
        )
        with TestClient(
            create_app(
                settings=configured,
                coordinator=FakeCoordinator(),
                webhook_dispatcher=dispatcher,  # type: ignore[arg-type]
            )
        ) as client:
            response = client.post("/v1/sessions", headers=AUTH_HEADERS)
            self.assertEqual(201, response.status_code)
            session_id = response.json()["session_id"]

        self.assertEqual(0, dispatcher.enqueued)
        with self.engine.connect() as connection:
            self.assertEqual(
                1,
                connection.scalar(
                    select(func.count()).select_from(WebhookEventRow).where(
                        WebhookEventRow.session_id == session_id
                    )
                ),
            )


class ArtifactTransactionTests(unittest.TestCase):
    def test_post_commit_cleanup_is_best_effort_and_continues(self) -> None:
        class CleanupStore:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[str, ...]]] = []

            def release_pending(self, artifact_ids: tuple[str, ...]) -> None:
                self.calls.append(("pending", artifact_ids))
                raise RuntimeError("cleanup failure")

            def release_owned(self, artifact_ids: tuple[str, ...], _session_id: str) -> None:
                self.calls.append(("owned", artifact_ids))

        store = CleanupStore()
        effects = _ArtifactTransaction(store)  # type: ignore[arg-type]
        effects.release_pending(("first",))
        effects.release_owned(("second",), "session")
        effects.commit()
        self.assertEqual([("pending", ("first",)), ("owned", ("second",))], store.calls)


@unittest.skipUnless(_URL, "KYC_TEST_DATABASE_URL is required for PostgreSQL integration tests")
class MigrationLifecycleTests(unittest.TestCase):
    """Alembic migrations apply idempotently on a scratch database."""

    def test_upgrade_downgrade_upgrade_cycle_is_idempotent(self) -> None:
        import urllib.parse

        from alembic import command

        from kyc_api.infrastructure.persistence.database import expected_revision, migration_config

        assert _URL is not None
        split = urllib.parse.urlsplit(_URL)
        assert split.path.rstrip("/")  # require explicit database
        maintenance_url = urllib.parse.urlunsplit(
            (split.scheme, split.netloc, "/postgres", split.query, split.fragment)
        )
        scratch_url = urllib.parse.urlunsplit(
            (split.scheme, split.netloc, "/kyc_migration_test", split.query, split.fragment)
        )

        def admin(database_url: str):
            return create_database_engine(database_url)

        def create_scratch() -> None:
            engine = admin(maintenance_url)
            try:
                with engine.connect() as connection:
                    connection.connection.connection.autocommit = True
                    connection.execute(text("DROP DATABASE IF EXISTS kyc_migration_test"))
                    connection.execute(text("CREATE DATABASE kyc_migration_test ENCODING 'UTF8' TEMPLATE template0"))
            finally:
                engine.dispose()

        create_scratch()
        try:
            engine = create_database_engine(scratch_url)
            try:
                revision = expected_revision(scratch_url)
                config = migration_config(scratch_url)
                command.upgrade(config, revision)
                verify_database(engine, scratch_url)
                command.downgrade(config, "base")
                with engine.connect() as connection:
                    self.assertFalse(connection.execute(text("SELECT version_num FROM alembic_version")).scalar())
                command.upgrade(config, revision)
                verify_database(engine, scratch_url)
                self.assertEqual(revision, expected_revision(scratch_url))
            finally:
                engine.dispose()
        finally:
            engine = admin(maintenance_url)
            try:
                with engine.connect() as connection:
                    connection.connection.connection.autocommit = True
                    connection.execute(text("DROP DATABASE IF EXISTS kyc_migration_test"))
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
