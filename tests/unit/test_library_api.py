import unittest
from pathlib import Path
from unittest.mock import patch

from kyc_engine import (
    DocumentCoordinator,
    ImageSource,
    IntakeLimits,
    KycExtractionResult,
    ProcessingStatus,
    build_paddle_document_coordinator,
)


class _Pipeline:
    def process(self, source: ImageSource) -> KycExtractionResult:
        raise AssertionError("The construction test must not process an image")


class LibraryApiTests(unittest.TestCase):
    def test_top_level_consumer_api_is_importable(self) -> None:
        self.assertTrue(callable(build_paddle_document_coordinator))
        self.assertIsInstance(ProcessingStatus.SUCCESS, ProcessingStatus)

    @patch("kyc_engine.defaults.build_paddle_pipeline")
    def test_two_sided_factory_builds_independent_side_pipelines(self, build_pipeline) -> None:
        front_pipeline = _Pipeline()
        back_pipeline = _Pipeline()
        front_detector = object()
        back_detector = object()
        limits = IntakeLimits(max_encoded_bytes=1024)
        build_pipeline.side_effect = (front_pipeline, back_pipeline)

        coordinator = build_paddle_document_coordinator(
            model_manifest=Path("private-models/manifest.json"),
            device="cpu",
            intake_limits=limits,
            front_detector=front_detector,  # type: ignore[arg-type]
            back_detector=back_detector,  # type: ignore[arg-type]
        )

        self.assertIsInstance(coordinator, DocumentCoordinator)
        self.assertIs(front_pipeline, coordinator.front_pipeline)
        self.assertIs(back_pipeline, coordinator.back_pipeline)
        self.assertEqual(
            [
                {
                    "model_manifest": Path("private-models/manifest.json"),
                    "device": "cpu",
                    "detector": front_detector,
                    "intake_limits": limits,
                    "side": "front",
                },
                {
                    "model_manifest": Path("private-models/manifest.json"),
                    "device": "cpu",
                    "detector": back_detector,
                    "intake_limits": limits,
                    "side": "back",
                },
            ],
            [call.kwargs for call in build_pipeline.call_args_list],
        )

    @patch("kyc_engine.defaults.build_paddle_pipeline", side_effect=ValueError("bad manifest"))
    def test_factory_propagates_model_configuration_failures(self, build_pipeline) -> None:
        with self.assertRaisesRegex(ValueError, "bad manifest"):
            build_paddle_document_coordinator(model_manifest="missing.json")
        self.assertEqual(1, build_pipeline.call_count)


if __name__ == "__main__":
    unittest.main()
