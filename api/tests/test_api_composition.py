import unittest
from unittest.mock import patch

from kyc_engine import IntakeLimits
from kyc_api.composition import create_coordinator

from helpers import FakeCoordinator, settings


class ApiCompositionTests(unittest.TestCase):
    @patch("kyc_api.composition.build_paddle_document_coordinator")
    def test_uses_the_library_two_sided_factory(self, build_coordinator) -> None:
        expected = FakeCoordinator()
        configured = settings()
        build_coordinator.return_value = expected

        coordinator = create_coordinator(configured)

        self.assertIs(expected, coordinator)
        build_coordinator.assert_called_once_with(
            model_manifest=configured.ocr_model_manifest,
            device=configured.ocr_device,
            intake_limits=IntakeLimits(max_encoded_bytes=configured.max_upload_bytes),
        )


if __name__ == "__main__":
    unittest.main()
