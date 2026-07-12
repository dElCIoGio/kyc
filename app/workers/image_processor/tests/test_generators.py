import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    OPENCV_AVAILABLE = False
else:
    OPENCV_AVAILABLE = True

if OPENCV_AVAILABLE:
    from image_processor.generators import (
        ClaheVariantGenerator,
        ContrastBrightnessVariantGenerator,
        DenoisingVariantGenerator,
        GammaVariantGenerator,
        GrayscaleVariantGenerator,
        OriginalVariantGenerator,
        SharpeningVariantGenerator,
        ThresholdVariantGenerator,
    )
    from image_processor.validator import validate_image


def sample_image() -> np.ndarray:
    return np.arange(16 * 16 * 3, dtype=np.uint8).reshape((16, 16, 3))


@unittest.skipUnless(OPENCV_AVAILABLE, "cv2 is not installed")
class GeneratorTests(unittest.TestCase):
    def assert_valid_batch(
        self,
        generator: object,
        expected_count: int,
        *,
        parameters: dict[str, object] | None = None,
    ) -> None:
        image = sample_image()
        original = image.copy()

        batch = generator.generate(image, parameters=parameters)  # type: ignore[attr-defined]

        self.assertEqual(generator.name, batch.generator)  # type: ignore[attr-defined]
        self.assertEqual(expected_count, len(batch.variants))
        self.assertTrue(np.array_equal(original, image))

        for variant in batch.variants:
            validate_image(variant.image)

    def test_original_generator(self) -> None:
        self.assert_valid_batch(OriginalVariantGenerator(), 1)

    def test_grayscale_generator(self) -> None:
        batch = GrayscaleVariantGenerator().generate(sample_image())

        self.assertEqual(1, len(batch.variants))
        self.assertEqual(2, batch.variants[0].image.ndim)
        validate_image(batch.variants[0].image)

    def test_clahe_generator(self) -> None:
        self.assert_valid_batch(
            ClaheVariantGenerator(
                clip_limits=(1.0, 2.0),
                tile_grid_sizes=((4, 4), (8, 8)),
            ),
            4,
        )

    def test_gamma_generator(self) -> None:
        self.assert_valid_batch(GammaVariantGenerator(gamma_values=(0.8, 1.0, 1.2)), 3)

    def test_gamma_runtime_parameters_override_defaults(self) -> None:
        batch = GammaVariantGenerator(gamma_values=(1.0,)).generate(
            sample_image(),
            parameters={"gamma_values": [0.7, 1.2]},
        )

        self.assertEqual(2, len(batch.variants))
        self.assertEqual({"gamma": 0.7}, batch.variants[0].parameters)
        self.assertEqual({"gamma": 1.2}, batch.variants[1].parameters)

    def test_contrast_brightness_generator(self) -> None:
        self.assert_valid_batch(
            ContrastBrightnessVariantGenerator(
                adjustments=(
                    {"alpha": 1.0, "beta": 0},
                    {"alpha": 1.2, "beta": 10},
                )
            ),
            2,
        )

    def test_sharpening_generator(self) -> None:
        self.assert_valid_batch(SharpeningVariantGenerator(strengths=(0.5, 1.0)), 2)

    def test_denoising_generator(self) -> None:
        self.assert_valid_batch(DenoisingVariantGenerator(h_values=(5.0, 10.0)), 2)

    def test_threshold_generator(self) -> None:
        self.assert_valid_batch(
            ThresholdVariantGenerator(
                methods=("otsu", "adaptive_mean", "adaptive_gaussian"),
                adaptive_block_sizes=(3, 5),
                adaptive_c_values=(1, 2),
            ),
            9,
        )

    def test_rejects_invalid_gamma_values(self) -> None:
        with self.assertRaises(ValueError):
            GammaVariantGenerator().generate(
                sample_image(),
                parameters={"gamma_values": [0]},
            )

    def test_rejects_invalid_threshold_block_size(self) -> None:
        with self.assertRaises(ValueError):
            ThresholdVariantGenerator().generate(
                sample_image(),
                parameters={"adaptive_block_sizes": [4]},
            )


if __name__ == "__main__":
    unittest.main()
