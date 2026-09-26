import unittest
from unittest.mock import patch

from pathlib import Path

from kyc_engine import IntakeLimits, MiniFASNetInitializationError
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore
from kyc_api.composition import create_coordinator, create_liveness_evaluator

from helpers import FakeCoordinator, settings


class ApiCompositionTests(unittest.TestCase):
    @patch("kyc_api.composition.build_paddle_document_coordinator")
    def test_uses_the_library_two_sided_factory(self, build_coordinator) -> None:
        expected = FakeCoordinator()
        configured = settings()
        build_coordinator.return_value = expected

        artifacts = InMemoryPortraitArtifactStore()
        coordinator = create_coordinator(configured, portrait_artifacts=artifacts)

        self.assertIs(expected, coordinator)
        build_coordinator.assert_called_once_with(
            model_manifest=configured.ocr_model_manifest,
            device=configured.ocr_device,
            intake_limits=IntakeLimits(max_encoded_bytes=configured.max_upload_bytes),
            portrait_artifacts=artifacts,
        )

    @patch("kyc_api.composition.OpenCVHaarFaceDetector")
    @patch("kyc_api.composition.MiniFASNetAntiSpoofDetector")
    def test_liveness_composition_constructs_detectors_for_passive_and_face_selection(
        self, detector_class, face_detector_class
    ) -> None:
        detector = object()
        face_detector = object()
        detector_class.return_value = detector
        face_detector_class.return_value = face_detector
        configured = settings(
            liveness_enabled=True,
            liveness_model_root=Path("private-models/liveness/minifasnet"),
            liveness_frame_count=4,
            liveness_min_real_ratio=0.75,
        )

        evaluator = create_liveness_evaluator(configured)

        assert evaluator is not None
        self.assertEqual(4, evaluator.frame_count)
        self.assertIs(detector, evaluator._detector)
        self.assertIs(face_detector, evaluator._face_detector)
        detector_class.assert_called_once_with(configured.liveness_model_root)
        face_detector_class.assert_called_once_with()

    def test_enabled_liveness_missing_models_fails_during_composition(self) -> None:
        configured = settings(
            liveness_enabled=True,
            liveness_model_root=Path("missing-private-models"),
        )
        with self.assertRaises(MiniFASNetInitializationError):
            create_liveness_evaluator(configured)


if __name__ == "__main__":
    unittest.main()
