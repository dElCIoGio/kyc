import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from image_processor.quality import (
    BrightnessAssessor,
    ContrastAssessor,
    DynamicRangeAssessor,
    HistogramClippingAssessor,
    SharpnessAssessor,
)
from image_processor.quality.utils import to_grayscale


class QualityAssessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.black = np.zeros((16, 16), dtype=np.uint8)
        self.white = np.full((16, 16), 255, dtype=np.uint8)
        self.grey = np.full((16, 16), 90, dtype=np.uint8)
        self.gradient = np.tile(
            np.arange(256, dtype=np.uint8),
            (16, 1),
        )

    def test_black_image_metrics(self) -> None:
        brightness = BrightnessAssessor().assess(self.black).metrics
        contrast = ContrastAssessor().assess(self.black).metrics
        dynamic_range = DynamicRangeAssessor().assess(self.black).metrics
        clipping = HistogramClippingAssessor().assess(self.black).metrics

        self.assertEqual(0.0, brightness["mean_intensity"])
        self.assertEqual(0.0, brightness["median_intensity"])
        self.assertEqual(0.0, contrast["intensity_stddev"])
        self.assertEqual(0.0, dynamic_range["absolute_range"])
        self.assertEqual(0.0, dynamic_range["percentile_range"])
        self.assertEqual(1.0, clipping["exact_black_ratio"])
        self.assertEqual(1.0, clipping["dark_pixel_ratio"])

    def test_white_image_metrics(self) -> None:
        brightness = BrightnessAssessor().assess(self.white).metrics
        contrast = ContrastAssessor().assess(self.white).metrics
        dynamic_range = DynamicRangeAssessor().assess(self.white).metrics
        clipping = HistogramClippingAssessor().assess(self.white).metrics

        self.assertEqual(255.0, brightness["mean_intensity"])
        self.assertEqual(255.0, brightness["median_intensity"])
        self.assertEqual(0.0, contrast["intensity_stddev"])
        self.assertEqual(0.0, dynamic_range["absolute_range"])
        self.assertEqual(0.0, dynamic_range["percentile_range"])
        self.assertEqual(1.0, clipping["exact_white_ratio"])
        self.assertEqual(1.0, clipping["bright_pixel_ratio"])

    def test_constant_grey_image_metrics(self) -> None:
        brightness = BrightnessAssessor().assess(self.grey).metrics
        contrast = ContrastAssessor().assess(self.grey).metrics
        dynamic_range = DynamicRangeAssessor().assess(self.grey).metrics

        self.assertEqual(90.0, brightness["mean_intensity"])
        self.assertEqual(90.0, brightness["median_intensity"])
        self.assertEqual(0.0, contrast["intensity_stddev"])
        self.assertEqual(0.0, dynamic_range["absolute_range"])
        self.assertEqual(0.0, dynamic_range["percentile_range"])

    def test_gradient_image_metrics(self) -> None:
        brightness = BrightnessAssessor().assess(self.gradient).metrics
        contrast = ContrastAssessor().assess(self.gradient).metrics
        dynamic_range = DynamicRangeAssessor().assess(self.gradient).metrics

        self.assertAlmostEqual(127.5, brightness["mean_intensity"])
        self.assertAlmostEqual(127.5, brightness["median_intensity"])
        self.assertGreater(contrast["intensity_stddev"], 0.0)
        self.assertEqual(0.0, dynamic_range["absolute_min"])
        self.assertEqual(255.0, dynamic_range["absolute_max"])
        self.assertEqual(255.0, dynamic_range["absolute_range"])
        self.assertGreater(dynamic_range["percentile_range"], 200.0)
        self.assertLess(
            dynamic_range["percentile_low"],
            dynamic_range["percentile_high"],
        )

    def test_sharp_checkerboard_has_more_laplacian_variance_than_blur(self) -> None:
        checkerboard = (
            (
                np.indices((64, 64))[0] // 8
                + np.indices((64, 64))[1] // 8
            )
            % 2
            * 255
        ).astype(np.uint8)
        blurred = cv2.GaussianBlur(checkerboard, (9, 9), 0)
        assessor = SharpnessAssessor()

        sharp_result = assessor.assess(checkerboard)
        blurred_result = assessor.assess(blurred)

        self.assertGreater(
            sharp_result.metrics["laplacian_variance"],
            blurred_result.metrics["laplacian_variance"],
        )

    def test_grayscale_conversion_supports_all_shared_image_formats(self) -> None:
        single_channel = self.grey[:, :, np.newaxis]
        bgr = cv2.cvtColor(self.grey, cv2.COLOR_GRAY2BGR)
        bgra = cv2.cvtColor(self.grey, cv2.COLOR_GRAY2BGRA)

        for image in (self.grey, single_channel, bgr, bgra):
            with self.subTest(shape=image.shape):
                converted = to_grayscale(image)
                self.assertEqual((16, 16), converted.shape)
                self.assertEqual(np.uint8, converted.dtype)
                self.assertTrue(np.array_equal(self.grey, converted))
                self.assertFalse(np.shares_memory(image, converted))

    def test_runtime_overrides_are_returned_as_effective_parameters(self) -> None:
        sharpness = SharpnessAssessor(kernel_size=3).assess(
            self.gradient,
            parameters={"kernel_size": 5},
        )
        dynamic_range = DynamicRangeAssessor().assess(
            self.gradient,
            parameters={"lower_percentile": 5, "upper_percentile": 95},
        )
        clipping = HistogramClippingAssessor().assess(
            self.gradient,
            parameters={"dark_threshold": 10, "bright_threshold": 240},
        )

        self.assertEqual({"kernel_size": 5}, dict(sharpness.parameters))
        self.assertEqual(
            {"lower_percentile": 5.0, "upper_percentile": 95.0},
            dict(dynamic_range.parameters),
        )
        self.assertEqual(
            {"dark_threshold": 10, "bright_threshold": 240},
            dict(clipping.parameters),
        )

    def test_rejects_invalid_dynamic_range_percentiles(self) -> None:
        invalid_values = ((-1, 99), (50, 50), (80, 20), (1, 101))
        for lower, upper in invalid_values:
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    DynamicRangeAssessor(
                        lower_percentile=lower,
                        upper_percentile=upper,
                    )

        with self.assertRaises(TypeError):
            DynamicRangeAssessor().assess(
                self.grey,
                parameters={"lower_percentile": True},
            )

    def test_rejects_invalid_histogram_clipping_thresholds(self) -> None:
        invalid_values = ((-1, 250), (5, 5), (250, 5), (5, 256))
        for dark, bright in invalid_values:
            with self.subTest(dark=dark, bright=bright):
                with self.assertRaises(ValueError):
                    HistogramClippingAssessor(
                        dark_threshold=dark,
                        bright_threshold=bright,
                    )

        with self.assertRaises(TypeError):
            HistogramClippingAssessor().assess(
                self.grey,
                parameters={"dark_threshold": 5.5},
            )

    def test_rejects_invalid_sharpness_kernel_sizes(self) -> None:
        for kernel_size in (0, 2, 32, 33):
            with self.subTest(kernel_size=kernel_size):
                with self.assertRaises(ValueError):
                    SharpnessAssessor(kernel_size=kernel_size)

        with self.assertRaises(TypeError):
            SharpnessAssessor().assess(
                self.grey,
                parameters={"kernel_size": 3.0},
            )

    def test_all_assessors_reject_unknown_parameters(self) -> None:
        assessors = (
            SharpnessAssessor(),
            BrightnessAssessor(),
            ContrastAssessor(),
            DynamicRangeAssessor(),
            HistogramClippingAssessor(),
        )

        for assessor in assessors:
            with self.subTest(assessor=assessor.name):
                with self.assertRaises(ValueError):
                    assessor.assess(self.grey, parameters={"unknown": 1})

    def test_all_assessors_leave_input_unchanged(self) -> None:
        image = cv2.cvtColor(self.gradient, cv2.COLOR_GRAY2BGRA)
        original = image.copy()
        assessors = (
            SharpnessAssessor(),
            BrightnessAssessor(),
            ContrastAssessor(),
            DynamicRangeAssessor(),
            HistogramClippingAssessor(),
        )

        for assessor in assessors:
            with self.subTest(assessor=assessor.name):
                assessor.assess(image)
                self.assertTrue(np.array_equal(original, image))


if __name__ == "__main__":
    unittest.main()
