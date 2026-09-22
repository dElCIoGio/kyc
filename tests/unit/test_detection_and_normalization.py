import unittest

import cv2
import numpy as np

from kyc_engine.contracts import DetectionResult, DocumentProfile, FieldDefinition, BoundingBox, Point, Quadrilateral
from kyc_engine.detection import DetectionError, OpenCVDocumentDetector
from kyc_engine.normalization import DocumentNormalizer


def synthetic_scene(two_cards: bool = False) -> np.ndarray:
    image = np.zeros((600, 900, 3), dtype=np.uint8)
    if two_cards:
        for card in (
            np.asarray(((25, 285), (390, 270), (400, 510), (35, 525)), dtype=np.int32),
            np.asarray(((500, 55), (865, 40), (875, 280), (510, 295)), dtype=np.int32),
        ):
            cv2.fillConvexPoly(image, card, (230, 230, 230))
            cv2.polylines(image, (card,), True, (255, 255, 255), 6)
        return image
    first = np.asarray(((140, 130), (700, 95), (735, 455), (115, 480)), dtype=np.int32)
    cv2.fillConvexPoly(image, first, (230, 230, 230))
    cv2.polylines(image, (first,), True, (255, 255, 255), 8)
    return image


class DetectionTests(unittest.TestCase):
    def test_detects_card_quadrilateral(self) -> None:
        result = OpenCVDocumentDetector().detect(synthetic_scene())
        self.assertEqual("ao_id_card", result.document_type)
        self.assertGreaterEqual(result.confidence, 0.48)
        self.assertEqual(4, len(result.corners.points))

    def test_back_detector_preserves_requested_side(self) -> None:
        result = OpenCVDocumentDetector(side="back").detect(synthetic_scene())
        self.assertEqual("back", result.side)

    def test_rejects_image_without_document(self) -> None:
        with self.assertRaises(DetectionError) as raised:
            OpenCVDocumentDetector().detect(np.zeros((300, 400, 3), dtype=np.uint8))
        self.assertEqual("DOCUMENT_NOT_DETECTED", raised.exception.code)

    def test_rejects_multiple_plausible_documents(self) -> None:
        with self.assertRaises(DetectionError) as raised:
            OpenCVDocumentDetector().detect(synthetic_scene(two_cards=True))
        self.assertEqual("MULTIPLE_DOCUMENTS", raised.exception.code)


class NormalizationTests(unittest.TestCase):
    def test_warps_to_profile_dimensions_and_preserves_inverse(self) -> None:
        image = synthetic_scene()
        profile = DocumentProfile(
            "test/front/v1",
            "test",
            "front",
            320,
            200,
            "reviewed",
            (FieldDefinition("value", BoundingBox(10, 10, 50, 20)),),
        )
        detection = DetectionResult(
            "test",
            "front",
            0.9,
            Quadrilateral(
                (
                    Point(140, 130),
                    Point(700, 95),
                    Point(735, 455),
                    Point(115, 480),
                )
            ),
            "fake",
            "1",
        )
        result = DocumentNormalizer().normalize(image, detection, profile)
        self.assertEqual((200, 320, 3), result.image.shape)
        forward = np.asarray(result.forward_transform)
        inverse = np.asarray(result.inverse_transform)
        self.assertTrue(np.allclose(forward @ inverse, np.eye(3), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
