import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from image_processor.validator import validate_image


class ValidateImageTests(unittest.TestCase):
    def test_accepts_valid_images(self) -> None:
        validate_image(np.zeros((4, 4), dtype=np.uint8))
        validate_image(np.zeros((4, 4, 3), dtype=np.uint8))
        validate_image(np.zeros((4, 4, 4), dtype=np.uint8))

    def test_rejects_none(self) -> None:
        with self.assertRaises(ValueError):
            validate_image(None)  # type: ignore[arg-type]

    def test_rejects_non_array(self) -> None:
        with self.assertRaises(TypeError):
            validate_image("not an image")  # type: ignore[arg-type]

    def test_rejects_empty_array(self) -> None:
        with self.assertRaises(ValueError):
            validate_image(np.array([], dtype=np.uint8))

    def test_rejects_non_uint8_array(self) -> None:
        with self.assertRaises(ValueError):
            validate_image(np.zeros((4, 4), dtype=np.float32))  # type: ignore[arg-type]

    def test_rejects_unsupported_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            validate_image(np.zeros((1, 1, 1, 1), dtype=np.uint8))

    def test_rejects_unsupported_channel_count(self) -> None:
        with self.assertRaises(ValueError):
            validate_image(np.zeros((4, 4, 2), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
