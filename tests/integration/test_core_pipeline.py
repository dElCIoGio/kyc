import json
import unittest

import cv2
import numpy as np

from kyc_engine.contracts import (
    DetectionResult,
    FieldCrop,
    OCRCandidate,
    Point,
    ProcessingStatus,
    Quadrilateral,
)
from kyc_engine.defaults import build_balanced_pipeline
from kyc_engine.detection import DetectionError


class FullImageDetector:
    def detect(self, image: np.ndarray) -> DetectionResult:
        height, width = image.shape[:2]
        return DetectionResult(
            document_type="ao_id_card",
            side="front",
            confidence=0.99,
            corners=Quadrilateral(
                (
                    Point(0, 0),
                    Point(width - 1, 0),
                    Point(width - 1, height - 1),
                    Point(0, height - 1),
                )
            ),
            detector="fake",
            detector_version="1",
        )


class ErrorDetector:
    def __init__(self, code: str) -> None:
        self.code = code

    def detect(self, image: np.ndarray) -> DetectionResult:
        raise DetectionError(self.code, "Document detection did not produce one unambiguous card")


class FakeRecognizer:
    name = "fake"
    model_version = "1"

    def __init__(self, *, omit_id: bool = False) -> None:
        self.omit_id = omit_id

    def recognize_batch(self, crops: tuple[FieldCrop, ...]) -> tuple[OCRCandidate, ...]:
        values = {
            "full_name": "MARIA SILVA",
            "father_name": "JOAO SILVA",
            "mother_name": "ANA SILVA",
            "id_number": "000000000LA000",
        }
        output = []
        for crop in crops:
            value = "" if self.omit_id and crop.field_name == "id_number" else values[crop.field_name]
            output.append(
                OCRCandidate(
                    field_name=crop.field_name,
                    raw_value=value,
                    confidence=0.9 if value else 0.0,
                    generator=crop.generator,
                    variant_name=crop.variant_name,
                    bounding_box=crop.bounding_box,
                    ocr_engine=self.name,
                    ocr_model_version=self.model_version,
                )
            )
        return tuple(output)


def synthetic_card() -> np.ndarray:
    image = np.full((467, 718, 3), 220, dtype=np.uint8)
    cv2.rectangle(image, (5, 202), (420, 245), (30, 30, 30), 2)
    cv2.rectangle(image, (5, 268), (500, 315), (70, 70, 70), 2)
    cv2.rectangle(image, (5, 340), (520, 395), (110, 110, 110), 2)
    cv2.putText(image, "000000000LA000", (8, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 1)
    return image


class CorePipelineTests(unittest.TestCase):
    def test_end_to_end_returns_structured_success(self) -> None:
        pipeline = build_balanced_pipeline(FakeRecognizer(), detector=FullImageDetector())
        result = pipeline.process(synthetic_card())
        self.assertEqual(ProcessingStatus.SUCCESS, result.status)
        self.assertEqual("ao_id_card/front/v1", result.profile_id)
        self.assertEqual("MARIA SILVA", result.fields["full_name"].raw_value)
        self.assertEqual("original", result.fields["full_name"].selected_candidate.variant_name)
        payload = result.to_dict()
        json.dumps(payload)
        self.assertNotIn("image", str(payload).lower())
        self.assertNotIn("path", str(payload).lower())
        self.assertIn("ocr", result.timings_ms)

    def test_missing_required_field_returns_partial_result(self) -> None:
        pipeline = build_balanced_pipeline(
            FakeRecognizer(omit_id=True),
            detector=FullImageDetector(),
        )
        result = pipeline.process(synthetic_card())
        self.assertEqual(ProcessingStatus.PARTIAL, result.status)
        self.assertEqual("missing", result.fields["id_number"].status.value)
        self.assertTrue(any(issue.code == "FIELD_MISSING" for issue in result.issues))

    def test_no_document_returns_failed_result(self) -> None:
        pipeline = build_balanced_pipeline(
            FakeRecognizer(),
            detector=ErrorDetector("DOCUMENT_NOT_DETECTED"),
        )
        result = pipeline.process(synthetic_card())
        self.assertEqual(ProcessingStatus.FAILED, result.status)
        self.assertEqual("DOCUMENT_NOT_DETECTED", result.issues[0].code)
        self.assertEqual({}, dict(result.fields))

    def test_multiple_documents_returns_failed_result(self) -> None:
        pipeline = build_balanced_pipeline(
            FakeRecognizer(),
            detector=ErrorDetector("MULTIPLE_DOCUMENTS"),
        )
        result = pipeline.process(synthetic_card())
        self.assertEqual(ProcessingStatus.FAILED, result.status)
        self.assertEqual("MULTIPLE_DOCUMENTS", result.issues[0].code)


if __name__ == "__main__":
    unittest.main()
