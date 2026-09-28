from __future__ import annotations

import unittest

import cv2
import numpy as np

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


def _document_scene(
    corners: np.ndarray | None = None,
    *,
    shape: tuple[int, int] = (600, 900),
) -> np.ndarray:
    height, width = shape
    image = np.full((height, width, 3), (48, 70, 88), dtype=np.uint8)
    card = corners if corners is not None else np.asarray(
        ((150, 145), (745, 135), (775, 470), (125, 485)), dtype=np.int32
    )
    card = card.astype(np.int32)
    cv2.fillConvexPoly(image, card, (202, 202, 202), cv2.LINE_AA)
    cv2.polylines(image, (card,), True, (246, 246, 246), 6, cv2.LINE_AA)
    for offset in range(0, 9):
        y = 205 + offset * 22
        cv2.line(image, (230, y), (620, y - 8), (75, 75, 75), 3, cv2.LINE_AA)
    cv2.rectangle(image, (610, 285), (710, 405), (115, 115, 115), -1)
    return image


def _flat(value: int = 128) -> np.ndarray:
    return np.full((240, 320, 3), value, dtype=np.uint8)


class DocumentCaptureAssessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assessor = DocumentCaptureAssessor(ImageIntake())

    def _codes(self, image: np.ndarray) -> list[CaptureIssueCode]:
        return [issue.code for issue in self.assessor.assess(_encode(image)).issues]

    def test_accepts_a_clear_centered_document_and_reports_internal_geometry(self) -> None:
        assessment = self.assessor.assess(_encode(_document_scene()))

        self.assertTrue(assessment.accepted)
        self.assertEqual((), assessment.issues)
        self.assertEqual((900, 600), (assessment.metrics.width, assessment.metrics.height))
        self.assertTrue(assessment.metrics.document_detected)
        self.assertGreater(assessment.metrics.document_area_ratio or 0, 0.12)
        self.assertIsNotNone(assessment.metrics.minimum_margin_ratio)
        self.assertIsNotNone(assessment.metrics.perspective_score)
        self.assertEqual(0.0, assessment.metrics.glare_ratio)

    def test_close_document_with_all_corners_visible_is_not_cropped_for_small_margin(self) -> None:
        close = np.asarray(((9, 10), (890, 11), (889, 585), (10, 584)), dtype=np.int32)
        codes = self._codes(_document_scene(close))

        self.assertNotIn(CaptureIssueCode.DOCUMENT_CROPPED, codes)

    def test_rejects_a_blurry_capture(self) -> None:
        blurred = cv2.GaussianBlur(_document_scene(), (51, 51), 0)
        self.assertIn(CaptureIssueCode.TOO_BLURRY, self._codes(blurred))

    def test_rejects_a_dark_capture(self) -> None:
        dark = np.clip(_document_scene().astype(np.int16) - 180, 0, 255).astype(np.uint8)
        self.assertIn(CaptureIssueCode.TOO_DARK, self._codes(dark))

    def test_rejects_an_overexposed_capture(self) -> None:
        self.assertIn(CaptureIssueCode.OVEREXPOSED, self._codes(_flat(245)))

    def test_rejects_a_low_contrast_capture(self) -> None:
        self.assertIn(CaptureIssueCode.LOW_CONTRAST, self._codes(_flat(128)))

    def test_rejects_a_plain_background_without_dependent_geometry_issues(self) -> None:
        codes = self._codes(_flat())

        self.assertIn(CaptureIssueCode.DOCUMENT_NOT_DETECTED, codes)
        self.assertNotIn(CaptureIssueCode.DOCUMENT_CROPPED, codes)
        self.assertNotIn(CaptureIssueCode.DOCUMENT_TOO_SMALL, codes)
        self.assertNotIn(CaptureIssueCode.EXCESSIVE_PERSPECTIVE, codes)
        self.assertNotIn(CaptureIssueCode.GLARE_DETECTED, codes)

    def test_rejects_a_small_document(self) -> None:
        small = np.asarray(((355, 245), (545, 242), (550, 360), (350, 364)), dtype=np.int32)
        self.assertIn(CaptureIssueCode.DOCUMENT_TOO_SMALL, self._codes(_document_scene(small)))

    def test_rejects_a_document_clipped_by_the_frame(self) -> None:
        cropped = np.asarray(((130, 100), (780, 100), (850, 650), (60, 650)), dtype=np.int32)
        self.assertIn(CaptureIssueCode.DOCUMENT_CROPPED, self._codes(_document_scene(cropped)))

    def test_rejects_strong_perspective_distortion(self) -> None:
        skewed = np.asarray(((275, 155), (625, 170), (800, 490), (95, 490)), dtype=np.int32)
        self.assertIn(CaptureIssueCode.EXCESSIVE_PERSPECTIVE, self._codes(_document_scene(skewed)))

    def test_rejects_an_obvious_glare_region(self) -> None:
        image = _document_scene()
        cv2.circle(image, (455, 315), 72, (255, 255, 255), -1, cv2.LINE_AA)
        assessment = self.assessor.assess(_encode(image))

        self.assertIn(CaptureIssueCode.GLARE_DETECTED, [issue.code for issue in assessment.issues])
        self.assertGreater(assessment.metrics.glare_ratio or 0, 0.035)

    def test_does_not_treat_an_arbitrary_wide_background_rectangle_as_a_document(self) -> None:
        image = np.full((600, 900, 3), (48, 70, 88), dtype=np.uint8)
        cv2.rectangle(image, (90, 230), (810, 345), (210, 210, 210), -1)
        cv2.rectangle(image, (90, 230), (810, 345), (245, 245, 245), 6)

        self.assertIn(CaptureIssueCode.DOCUMENT_NOT_DETECTED, self._codes(image))

    def test_multiple_independent_issues_are_stably_ordered(self) -> None:
        codes = self._codes(_flat())

        self.assertEqual(
            [
                CaptureIssueCode.DOCUMENT_NOT_DETECTED,
                CaptureIssueCode.TOO_BLURRY,
                CaptureIssueCode.LOW_CONTRAST,
            ],
            codes,
        )

    def test_invalid_encoded_input_is_a_safe_typed_error(self) -> None:
        with self.assertRaises(CaptureAssessmentInputError) as raised:
            self.assessor.assess(b"not-an-image")
        self.assertEqual("DECODE_FAILED", raised.exception.code)

    def test_intake_limits_remain_enforced(self) -> None:
        assessor = DocumentCaptureAssessor(ImageIntake(IntakeLimits(max_pixels=100)))
        with self.assertRaises(CaptureAssessmentInputError) as raised:
            assessor.assess(_encode(_document_scene()))
        self.assertEqual("IMAGE_TOO_LARGE", raised.exception.code)

    def test_thresholds_are_validated_and_injectable(self) -> None:
        for invalid in (
            {"min_brightness": 220, "max_brightness": 220},
            {"min_document_area_ratio": 1.0},
            {"min_edge_margin_ratio": 0.5},
            {"max_perspective_distortion": 0.0},
            {"max_glare_ratio": float("nan")},
        ):
            with self.assertRaises(ValueError):
                CaptureAssessmentConfig(**invalid)
        assessor = DocumentCaptureAssessor(
            ImageIntake(), CaptureAssessmentConfig(min_sharpness=1_000_000)
        )
        self.assertFalse(assessor.assess(_encode(_document_scene())).accepted)


if __name__ == "__main__":
    unittest.main()
