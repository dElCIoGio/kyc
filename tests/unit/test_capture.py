from __future__ import annotations

import io
import unittest

import cv2
import numpy as np
from PIL import Image

from kyc_engine.capture import (
    CaptureAssessmentConfig,
    CaptureAssessmentInputError,
    DocumentCaptureAssessor,
)
from kyc_engine.contracts import CaptureIssueCode
from kyc_engine.intake import ImageIntake, IntakeLimits


def _encode(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise AssertionError("could not encode synthetic image")
    return encoded.tobytes()


def _pattern(low: int = 30, high: int = 225) -> np.ndarray:
    rows, columns = np.indices((240, 320))
    return (((rows // 8 + columns // 8) % 2) * (high - low) + low).astype(np.uint8)


class DocumentCaptureAssessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assessor = DocumentCaptureAssessor(ImageIntake())

    def test_accepts_a_clear_well_exposed_capture(self) -> None:
        assessment = self.assessor.assess(_encode(_pattern()))
        self.assertTrue(assessment.accepted)
        self.assertEqual((), assessment.issues)
        self.assertEqual((320, 240), (assessment.metrics.width, assessment.metrics.height))

    def test_rejects_a_blurry_capture(self) -> None:
        blurred = cv2.GaussianBlur(_pattern(), (51, 51), 0)
        assessment = self.assessor.assess(_encode(blurred))
        self.assertIn(CaptureIssueCode.TOO_BLURRY, {issue.code for issue in assessment.issues})

    def test_rejects_a_dark_capture(self) -> None:
        dark = np.clip(_pattern().astype(np.int16) - 180, 0, 255).astype(np.uint8)
        assessment = self.assessor.assess(_encode(dark))
        self.assertIn(CaptureIssueCode.TOO_DARK, {issue.code for issue in assessment.issues})

    def test_rejects_an_overexposed_capture(self) -> None:
        assessment = self.assessor.assess(_encode(_pattern(230, 255)))
        self.assertIn(CaptureIssueCode.OVEREXPOSED, {issue.code for issue in assessment.issues})

    def test_rejects_a_low_contrast_capture(self) -> None:
        assessment = self.assessor.assess(_encode(_pattern(125, 130)))
        self.assertIn(CaptureIssueCode.LOW_CONTRAST, {issue.code for issue in assessment.issues})

    def test_invalid_encoded_input_is_a_safe_typed_error(self) -> None:
        with self.assertRaises(CaptureAssessmentInputError) as raised:
            self.assessor.assess(b"not-an-image")
        self.assertEqual("DECODE_FAILED", raised.exception.code)

    def test_intake_limits_remain_enforced(self) -> None:
        assessor = DocumentCaptureAssessor(ImageIntake(IntakeLimits(max_pixels=100)))
        with self.assertRaises(CaptureAssessmentInputError) as raised:
            assessor.assess(_encode(_pattern()))
        self.assertEqual("IMAGE_TOO_LARGE", raised.exception.code)

    def test_thresholds_are_validated_and_injectable(self) -> None:
        with self.assertRaises(ValueError):
            CaptureAssessmentConfig(min_brightness=220, max_brightness=220)
        assessor = DocumentCaptureAssessor(
            ImageIntake(), CaptureAssessmentConfig(min_sharpness=1_000_000)
        )
        self.assertFalse(assessor.assess(_encode(_pattern())).accepted)


if __name__ == "__main__":
    unittest.main()
