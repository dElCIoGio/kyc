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

    def test_back_refinement_moves_expanded_search_geometry_toward_visible_border(self) -> None:
        image = np.zeros((600, 900, 3), dtype=np.uint8)
        truth = np.asarray(((160, 130), (710, 105), (745, 465), (125, 490)), dtype=np.int32)
        cv2.fillConvexPoly(image, truth, (210, 210, 210))
        cv2.polylines(image, (truth,), True, (255, 255, 255), 7, cv2.LINE_AA)
        expanded = np.asarray(((105, 95), (755, 75), (785, 500), (80, 520)), dtype=np.float32)
        expanded_quad = Quadrilateral(tuple(Point(float(x), float(y)) for x, y in expanded))  # type: ignore[arg-type]

        refined, support = OpenCVDocumentDetector(side="back")._refine_back_edges(
            image,
            expanded_quad,
        )

        self.assertIsNotNone(refined)
        assert refined is not None
        self.assertTrue(all(value >= 0.50 for value in support.values()))
        expanded_error = float(np.linalg.norm(expanded - truth, axis=1).mean())
        refined_error = float(np.linalg.norm(refined.as_array() - truth, axis=1).mean())
        self.assertLess(refined_error, expanded_error)

    def test_back_refinement_falls_back_to_expanded_discovery_geometry_when_weak(self) -> None:
        image = np.full((300, 500, 3), 128, dtype=np.uint8)
        result, debug = OpenCVDocumentDetector(side="back").detect_with_debug(synthetic_scene())

        refined, support = OpenCVDocumentDetector(side="back")._refine_back_edges(
            image,
            debug.coarse_corners,
        )

        self.assertIsNone(refined)
        self.assertLess(len(support), 4)
        self.assertEqual("back", result.side)

    def test_rejects_image_without_document(self) -> None:
        with self.assertRaises(DetectionError) as raised:
            OpenCVDocumentDetector().detect(np.zeros((300, 400, 3), dtype=np.uint8))
        self.assertEqual("DOCUMENT_NOT_DETECTED", raised.exception.code)

    def test_rejects_multiple_plausible_documents(self) -> None:
        with self.assertRaises(DetectionError) as raised:
            OpenCVDocumentDetector().detect(synthetic_scene(two_cards=True))
        self.assertEqual("MULTIPLE_DOCUMENTS", raised.exception.code)

    def test_front_refinement_moves_coarse_geometry_toward_visible_card_border(self) -> None:
        image = np.zeros((600, 900, 3), dtype=np.uint8)
        truth = np.asarray(((160, 130), (710, 105), (745, 465), (125, 490)), dtype=np.int32)
        cv2.fillConvexPoly(image, truth, (210, 210, 210))
        cv2.polylines(image, (truth,), True, (255, 255, 255), 7, cv2.LINE_AA)
        for x in range(210, 690, 35):
            cv2.line(image, (x, 155), (x + 20, 430), (165, 165, 165), 2, cv2.LINE_AA)
        for y in range(170, 440, 30):
            cv2.line(image, (180, y), (700, y - 20), (150, 150, 150), 2, cv2.LINE_AA)
        coarse = np.asarray(((145, 120), (725, 95), (760, 475), (110, 500)), dtype=np.float32)
        coarse_quad = Quadrilateral(tuple(Point(float(x), float(y)) for x, y in coarse))  # type: ignore[arg-type]

        refined, support = OpenCVDocumentDetector()._refine_front_edges(image, coarse_quad)

        self.assertIsNotNone(refined)
        assert refined is not None
        self.assertTrue(all(value >= 0.45 for value in support.values()))
        coarse_error = float(np.linalg.norm(coarse - truth, axis=1).mean())
        refined_error = float(np.linalg.norm(refined.as_array() - truth, axis=1).mean())
        self.assertLess(refined_error, coarse_error)

    def test_front_refinement_falls_back_when_outer_border_evidence_is_missing(self) -> None:
        image = np.full((300, 500, 3), 128, dtype=np.uint8)
        coarse = Quadrilateral(
            (Point(70, 70), Point(430, 70), Point(430, 250), Point(70, 250))
        )

        refined, support = OpenCVDocumentDetector()._refine_front_edges(image, coarse)

        self.assertIsNone(refined)
        self.assertLess(len(support), 4)

    def test_front_debug_reports_final_or_tight_fallback_geometry(self) -> None:
        result, debug = OpenCVDocumentDetector().detect_with_debug(synthetic_scene())

        self.assertEqual(result.corners, debug.final_corners)
        self.assertIn(debug.candidate_kind, {"contour_quadrilateral", "rotated_rectangle"})
        self.assertIn("front_refinement_accepted", result.confidence_components)


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
