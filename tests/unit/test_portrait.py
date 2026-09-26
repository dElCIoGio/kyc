import unittest

import numpy as np

from kyc_engine.contracts import (
    BoundingBox,
    FaceCandidate,
    FaceDetectionResult,
    PortraitDefinition,
    PortraitStatus,
)
from kyc_engine.portrait import PortraitExtractor
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore


class _Detector:
    def __init__(self, faces=(), error: Exception | None = None) -> None:
        self.faces = tuple(faces)
        self.error = error
        self.images = []

    def detect(self, image):
        self.images.append(image.copy())
        if self.error is not None:
            raise self.error
        return FaceDetectionResult(self.faces)


class _FailingStore(InMemoryPortraitArtifactStore):
    def put(self, image):
        raise RuntimeError("store unavailable")


class PortraitExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((300, 400, 3), dtype=np.uint8)
        self.region = PortraitDefinition(100, 40, 180, 200)
        self.store = InMemoryPortraitArtifactStore()

    def test_valid_region_produces_readonly_correctly_sized_crop(self) -> None:
        detector = _Detector((FaceCandidate(BoundingBox(30, 30, 80, 90)),))
        result = PortraitExtractor(detector, self.store).extract(self.image, self.region)
        self.assertEqual(PortraitStatus.AVAILABLE, result.status)
        self.assertTrue(result.eligible_for_face_match)
        self.assertEqual((200, 180, 3), detector.images[0].shape)
        assert result.artifact_id is not None
        self.assertFalse(self.store.get(result.artifact_id, session_id="other") is not None)

    def test_region_is_clamped(self) -> None:
        detector = _Detector((FaceCandidate(BoundingBox(10, 10, 80, 90)),))
        result = PortraitExtractor(detector, self.store).extract(
            self.image, PortraitDefinition(-20, 250, 180, 100)
        )
        self.assertEqual(BoundingBox(0, 250, 160, 50), result.clamped_region)
        self.assertEqual(PortraitStatus.TOO_SMALL, result.status)

    def test_invalid_and_undersized_regions_are_safe(self) -> None:
        invalid = PortraitExtractor(_Detector(), self.store).extract(
            self.image, PortraitDefinition(500, 10, 20, 20)
        )
        small = PortraitExtractor(_Detector(), self.store).extract(
            self.image, PortraitDefinition(1, 1, 127, 200)
        )
        self.assertEqual(PortraitStatus.CROP_INVALID, invalid.status)
        self.assertIn("PORTRAIT_CROP_INVALID", invalid.warnings)
        self.assertEqual(PortraitStatus.TOO_SMALL, small.status)
        self.assertIn("PORTRAIT_TOO_SMALL", small.warnings)

    def test_face_outcomes_keep_artifact_but_control_eligibility(self) -> None:
        cases = (
            ((), PortraitStatus.FACE_NOT_FOUND, "PORTRAIT_FACE_NOT_FOUND"),
            ((FaceCandidate(BoundingBox(1, 1, 63, 64)),), PortraitStatus.FACE_TOO_SMALL, "PORTRAIT_FACE_TOO_SMALL"),
            ((FaceCandidate(BoundingBox(1, 1, 70, 70)), FaceCandidate(BoundingBox(80, 1, 70, 70))), PortraitStatus.MULTIPLE_FACES, "PORTRAIT_MULTIPLE_FACES"),
        )
        for faces, status, warning in cases:
            with self.subTest(status=status):
                result = PortraitExtractor(_Detector(faces), self.store).extract(self.image, self.region)
                self.assertEqual(status, result.status)
                self.assertFalse(result.eligible_for_face_match)
                self.assertIsNotNone(result.artifact_id)
                self.assertIn(warning, result.warnings)

    def test_detector_and_artifact_failure_are_recoverable(self) -> None:
        detector_failure = PortraitExtractor(_Detector(error=RuntimeError("detector")), self.store).extract(self.image, self.region)
        artifact_failure = PortraitExtractor(_Detector(), _FailingStore()).extract(self.image, self.region)
        self.assertEqual(PortraitStatus.DETECTOR_FAILED, detector_failure.status)
        self.assertIsNotNone(detector_failure.artifact_id)
        self.assertEqual(PortraitStatus.ARTIFACT_FAILED, artifact_failure.status)
        self.assertIsNone(artifact_failure.artifact_id)

    def test_store_claim_is_exclusive_and_release_is_owner_scoped(self) -> None:
        artifact_id = self.store.put(self.image)
        self.assertTrue(self.store.claim((artifact_id,), "session-a"))
        self.assertFalse(self.store.claim((artifact_id,), "session-b"))
        self.store.release_owned((artifact_id,), "session-b")
        self.assertTrue(self.store.exists(artifact_id))
        self.store.release_owned((artifact_id,), "session-a")
        self.assertFalse(self.store.exists(artifact_id))


if __name__ == "__main__":
    unittest.main()
