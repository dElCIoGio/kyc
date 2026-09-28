from __future__ import annotations

import asyncio
import time
import unittest
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from kyc_engine import DocumentExtractionResult, KycExtractionResult, ProcessingStatus
from kyc_engine.contracts import ExtractedField, FieldStatus
from kyc_api.main import create_app
from kyc_api.models import NifVerificationStatus
from kyc_api.nif import (
    MinfinNifVerifier,
    NifVerificationDispatcher,
    NifVerificationResult,
    NifVerifierCapacityExceeded,
)
from kyc_api.models import CaptureStatus, DocumentStatus, FaceMatchStatus, LivenessStatus, VerificationStatus
from kyc_api.sessions import DocumentSnapshot, SessionSnapshot
from kyc_api.webhooks import WebhookEvent

from helpers import AUTH_HEADERS, FakeCoordinator, settings


class _Verifier:
    def __init__(self, result: NifVerificationResult, delay: float = 0.0) -> None:
        self.result = result
        self.delay = delay
        self.calls: list[tuple[str, str | None]] = []
        self.closed = False

    async def verify(self, nif: str, claimed_name: str | None = None) -> NifVerificationResult:
        self.calls.append((nif, claimed_name))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.result

    async def close(self) -> None:
        self.closed = True


class _RawChecker:
    def __init__(self, status: str, *, nif: str | None = None, name: str | None = None) -> None:
        self._result = type(
            "RawResult", (), {"status": type("Status", (), {"value": status})(), "nif": nif, "name": name}
        )()
        self.closed = False

    async def verify(self, _nif: str):
        return self._result

    async def close(self) -> None:
        self.closed = True


def _document(*, identifier: str | None = "007096754LA043", name: str | None = "Ana Silva") -> DocumentExtractionResult:
    fields = {}
    if identifier is not None:
        fields["id_number"] = ExtractedField(
            name="id_number", raw_value=identifier, normalized_value=identifier,
            status=FieldStatus.VALID, confidence=1.0, selected_candidate=None, warnings=(),
        )
    if name is not None:
        fields["full_name"] = ExtractedField(
            name="full_name", raw_value=name, normalized_value=name,
            status=FieldStatus.VALID, confidence=1.0, selected_candidate=None, warnings=(),
        )
    front = KycExtractionResult(
        schema_version="1.1", processing_id="safe", status=ProcessingStatus.SUCCESS,
        document_type="ao_id_card", side="front", profile_id="ao_id_card/front/v1",
        detection=None, fields=fields, issues=(), timings_ms={},
    )
    return DocumentExtractionResult("1.0", ProcessingStatus.SUCCESS, front, None, (), {})


class NifApiTests(unittest.TestCase):
    def test_standalone_uses_safe_shared_semantics(self) -> None:
        verifier = _Verifier(NifVerificationResult(NifVerificationStatus.VERIFIED, "minfin", True))
        with TestClient(
            create_app(
                settings=settings(nif_verification_enabled=True),
                coordinator=FakeCoordinator(),
                nif_verifier=verifier,
            )
        ) as client:
            response = client.post(
                "/v1/verifications/nif", headers=AUTH_HEADERS,
                json={"nif": "007 096754-la043", "claimed_name": "Ana Silva"},
            )
            self.assertEqual(200, response.status_code)
            self.assertEqual(
                {"status": "verified", "source": "minfin", "name_match": True}, response.json()
            )
            self.assertEqual([("007096754LA043", "Ana Silva")], verifier.calls)
            self.assertEqual(401, client.post("/v1/verifications/nif", json={"nif": "x"}).status_code)
        self.assertTrue(verifier.closed)

    def test_minfin_adapter_keeps_identifier_and_name_signals_separate(self) -> None:
        async def run():
            adapter = MinfinNifVerifier(timeout_seconds=1)
            adapter._checker = _RawChecker("verified", nif="007096754LA043", name="ÁNA-SILVA")
            matching = await adapter.verify("007096754LA043", "Ana Silva")
            adapter._checker = _RawChecker("verified", nif="007096754LA043", name="Other Person")
            different = await adapter.verify("007096754LA043", "Ana Silva")
            adapter._checker = _RawChecker("verified", nif="wrong", name="Ana Silva")
            malformed = await adapter.verify("007096754LA043", "Ana Silva")
            adapter._checker = _RawChecker("not_found")
            absent = await adapter.verify("007096754LA043")
            return matching, different, malformed, absent

        matching, different, malformed, absent = asyncio.run(run())
        self.assertEqual((NifVerificationStatus.VERIFIED, True), (matching.status, matching.name_match))
        self.assertEqual((NifVerificationStatus.VERIFIED, False), (different.status, different.name_match))
        self.assertEqual(NifVerificationStatus.FAILED, malformed.status)
        self.assertEqual(NifVerificationStatus.NOT_FOUND, absent.status)

    def test_profile_scoped_session_trigger_and_not_found_failure(self) -> None:
        verifier = _Verifier(NifVerificationResult(NifVerificationStatus.NOT_FOUND, "minfin"))
        coordinator = FakeCoordinator()
        coordinator.output = _document()
        with TestClient(
            create_app(
                settings=settings(nif_verification_enabled=True), coordinator=coordinator,
                nif_verifier=verifier,
            )
        ) as client:
            session_id = client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
            store = client.app.state.sessions
            from helpers import accepted_capture_assessment, PNG_BYTES
            from kyc_api.models import DocumentSide
            store.accept_document_capture(session_id, DocumentSide.FRONT, PNG_BYTES, accepted_capture_assessment())
            store.accept_document_capture(session_id, DocumentSide.BACK, PNG_BYTES, accepted_capture_assessment())
            client.app.state.verifications.start_document_processing(session_id)
            deadline = time.time() + 2
            while time.time() < deadline and store.get(session_id).nif_verification_status == NifVerificationStatus.NOT_RUN:
                time.sleep(0.01)
            deadline = time.time() + 2
            while time.time() < deadline and store.get(session_id).nif_verification_status == NifVerificationStatus.PROCESSING:
                time.sleep(0.01)
            snapshot = store.get(session_id)
            self.assertEqual(NifVerificationStatus.NOT_FOUND, snapshot.nif_verification_status)
            self.assertEqual("minfin", snapshot.nif_verification_source)
            self.assertEqual("failed", snapshot.verification_status.value)


class NifDispatcherTests(unittest.TestCase):
    def test_capacity_is_bounded_and_worker_closes_verifier(self) -> None:
        verifier = _Verifier(NifVerificationResult(NifVerificationStatus.VERIFIED, "minfin"), delay=0.1)
        dispatcher = NifVerificationDispatcher(
            verifier_factory=lambda: verifier, session_capacity=1, standalone_capacity=1
        )
        try:
            first = dispatcher.submit_standalone("one", None)
            # The queue is bounded independently of the in-flight worker.
            with self.assertRaises(NifVerifierCapacityExceeded):
                dispatcher.submit_standalone("two", None)
            self.assertEqual(NifVerificationStatus.VERIFIED, first.result(timeout=1).status)
        finally:
            dispatcher.shutdown()
        self.assertTrue(verifier.closed)

    def test_nif_webhook_is_minimal_and_skipped_state_has_no_event(self) -> None:
        snapshot = SessionSnapshot(
            session_id="safe-session", verification_status=VerificationStatus.IN_PROGRESS,
            created_at=datetime.now(UTC), expires_at=datetime.now(UTC) + timedelta(minutes=1),
            document=DocumentSnapshot(DocumentStatus.PASSED, CaptureStatus.ACCEPTED, CaptureStatus.ACCEPTED, None, True, None),
            liveness_status=LivenessStatus.READY, face_match_status=FaceMatchStatus.BLOCKED,
            nif_verification_status=NifVerificationStatus.VERIFIED,
            nif_verification_source="minfin", nif_name_match=True,
        )
        event = WebhookEvent.from_snapshot(snapshot, transition_reason="nif.completed")
        assert event is not None
        self.assertEqual(
            {"session_id", "sequence", "nif_verification"}, set(__import__("json").loads(event.payload())["data"])
        )
        self.assertIsNone(WebhookEvent.from_snapshot(snapshot, transition_reason="nif.skipped"))
