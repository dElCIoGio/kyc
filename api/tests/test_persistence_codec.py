from __future__ import annotations

from dataclasses import replace
import json
import unittest
from pathlib import Path
from pydantic import ValidationError

from kyc_engine import PortraitExtractionResult, PortraitStatus
from kyc_engine.contracts import (
    BoundingBox,
    ExtractedField,
    FaceCandidate,
    FieldStatus,
    IssueSeverity,
    OCRCandidate,
    PipelineIssue,
    Point,
    PortraitDefinition,
)

from kyc_api.infrastructure.persistence.codec import (
    DocumentResultCodecError,
    decode_document_result,
    encode_document_result,
)
from kyc_api.infrastructure.persistence.database import create_database_engine
from kyc_api.settings import ApiSettings

from helpers import extraction_result


class DocumentResultCodecTests(unittest.TestCase):
    def test_round_trip_preserves_safe_result_and_removes_artifact_id(self) -> None:
        source = extraction_result()
        assert source.front is not None
        portrait = PortraitExtractionResult(
            status=PortraitStatus.AVAILABLE,
            requested_region=PortraitDefinition(1, 2, 30, 40),
            clamped_region=BoundingBox(1, 2, 30, 40),
            face_detected=True,
            face_count=1,
            face=FaceCandidate(BoundingBox(3, 4, 10, 12), 0.9, (Point(4.0, 5.0),)),
            eligible_for_face_match=True,
            warnings=("SAFE_WARNING",),
            artifact_id="must-not-be-persisted",
        )
        source = replace(source, front=replace(source.front, portrait=portrait))

        encoded = encode_document_result(source)
        serialized = json.dumps(encoded)
        restored = decode_document_result(encoded)

        self.assertNotIn("must-not-be-persisted", serialized)
        self.assertEqual(source.status, restored.status)
        self.assertEqual(source.front.profile_id, restored.front.profile_id)  # type: ignore[union-attr]
        self.assertEqual(portrait.status, restored.front.portrait.status)  # type: ignore[union-attr]
        self.assertIsNone(restored.front.portrait.artifact_id)  # type: ignore[union-attr]

    def test_malformed_and_unsupported_payloads_fail_safely(self) -> None:
        for payload in ({}, {"codec_version": 999, "result": {}}, {"codec_version": 1, "result": []}):
            with self.subTest(payload=payload), self.assertRaises(DocumentResultCodecError):
                decode_document_result(payload)

    def test_round_trip_preserves_fields_candidates_issues_and_timings(self) -> None:
        source = extraction_result()
        assert source.front is not None
        candidate = OCRCandidate(
            field_name="id_number",
            raw_value=" 123 ",
            confidence=0.91,
            generator="gray",
            variant_name="base",
            bounding_box=BoundingBox(1, 2, 30, 10),
            ocr_engine="test",
            ocr_model_version="1",
        )
        fields = {
            "id_number": ExtractedField(
                "id_number", FieldStatus.VALID, " 123 ", "123", 0.91, candidate,
                alternatives=(candidate,), warnings=("NORMALIZED",),
            ),
            "invalid": ExtractedField("invalid", FieldStatus.INVALID, "x", None, 0.4, None),
            "error": ExtractedField("error", FieldStatus.ERROR, None, None, 0.0, None),
        }
        issue = PipelineIssue("validation", "SAFE_CODE", IssueSeverity.WARNING, "safe message", "invalid")
        source = replace(
            source,
            front=replace(source.front, fields=fields, issues=(issue,), timings_ms={"ocr": 2.5}),
            issues=(issue,),
        )

        restored = decode_document_result(encode_document_result(source))
        assert restored.front is not None
        self.assertEqual(fields, dict(restored.front.fields))
        self.assertEqual((issue,), restored.front.issues)
        self.assertEqual({"ocr": 2.5}, dict(restored.front.timings_ms))

    def test_postgres_configuration_requires_database_url(self) -> None:
        common = {"api_key": "test-api-key-123456789", "ocr_model_manifest": Path("unused")}
        self.assertEqual("memory", ApiSettings(**common).session_backend)
        with self.assertRaises(ValidationError):
            ApiSettings(**common, session_backend="postgres", database_url=None)
        configured = ApiSettings(
            **common,
            session_backend="postgres",
            database_url="postgresql+psycopg://user:secret@example/kyc",
        )
        self.assertEqual("postgres", configured.session_backend)

    def test_invalid_database_configuration_does_not_echo_secret(self) -> None:
        secret = "database-password-must-not-leak"
        with self.assertRaises(RuntimeError) as raised:
            create_database_engine(f"postgresql+missing://user:{secret}@localhost/kyc")
        self.assertNotIn(secret, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
