import unittest

import cv2
import numpy as np

from kyc_engine.contracts import VariantBatch, VariantInfo
from kyc_engine.quality import (
    BrightnessAssessor,
    ContrastAssessor,
    DynamicRangeAssessor,
    HistogramClippingAssessor,
    SharpnessAssessor,
    VariantQualityAssessmentPipeline,
    to_grayscale,
)


class AssessorTests(unittest.TestCase):
    def test_constant_images_have_expected_measurements(self) -> None:
        black = np.zeros((20, 30), dtype=np.uint8)
        white = np.full((20, 30), 255, dtype=np.uint8)
        grey = np.full((20, 30), 80, dtype=np.uint8)

        self.assertEqual(0.0, BrightnessAssessor().assess(black).metrics["mean_intensity"])
        self.assertEqual(255.0, BrightnessAssessor().assess(white).metrics["median_intensity"])
        self.assertEqual(80.0, BrightnessAssessor().assess(grey).metrics["mean_intensity"])
        self.assertEqual(0.0, ContrastAssessor().assess(grey).metrics["intensity_stddev"])
        self.assertEqual(0.0, DynamicRangeAssessor().assess(grey).metrics["absolute_range"])
        self.assertEqual(1.0, HistogramClippingAssessor().assess(black).metrics["exact_black_ratio"])
        self.assertEqual(1.0, HistogramClippingAssessor().assess(white).metrics["bright_pixel_ratio"])

    def test_gradient_has_contrast_range_and_midpoint_brightness(self) -> None:
        gradient = np.tile(np.arange(256, dtype=np.uint8), (20, 1))
        self.assertGreater(ContrastAssessor().assess(gradient).metrics["intensity_stddev"], 70)
        dynamic = DynamicRangeAssessor().assess(gradient).metrics
        self.assertEqual(255.0, dynamic["absolute_range"])
        self.assertGreater(dynamic["percentile_range"], 240)
        self.assertAlmostEqual(127.5, BrightnessAssessor().assess(gradient).metrics["mean_intensity"])

    def test_checkerboard_is_sharper_than_blurred_copy(self) -> None:
        rows, columns = np.indices((80, 80))
        checker = (((rows // 8 + columns // 8) % 2) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(checker, (9, 9), 0)
        assessor = SharpnessAssessor()
        self.assertGreater(
            assessor.assess(checker).metrics["laplacian_variance"],
            assessor.assess(blurred).metrics["laplacian_variance"],
        )

    def test_grayscale_conversion_supports_all_allowed_channels(self) -> None:
        gray = np.full((8, 9), 100, dtype=np.uint8)
        for image in (
            gray,
            gray[:, :, None],
            np.repeat(gray[:, :, None], 3, axis=2),
            np.repeat(gray[:, :, None], 4, axis=2),
        ):
            converted = to_grayscale(image)
            self.assertEqual((8, 9), converted.shape)
            self.assertEqual(np.uint8, converted.dtype)

    def test_rejects_invalid_configuration_and_unknown_parameters(self) -> None:
        with self.assertRaises(ValueError):
            DynamicRangeAssessor(90, 10)
        with self.assertRaises(ValueError):
            HistogramClippingAssessor(250, 10)
        with self.assertRaises(ValueError):
            SharpnessAssessor(2)
        with self.assertRaises(ValueError):
            BrightnessAssessor().assess(np.zeros((5, 5), dtype=np.uint8), {"unused": 1})


class QualityPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        image = np.tile(np.arange(20, dtype=np.uint8), (10, 1))
        self.batch = VariantBatch(
            "gamma",
            (
                VariantInfo("gamma_0.8", image, {"gamma": 0.8}),
                VariantInfo("gamma_1.2", image.copy(), {"gamma": 1.2}),
            ),
        )

    def test_runs_all_assessors_for_every_variant_in_stable_order(self) -> None:
        pipeline = VariantQualityAssessmentPipeline()
        original = tuple(variant.image.copy() for variant in self.batch.variants)
        result = pipeline.assess((self.batch,))[0]
        self.assertEqual(2, len(result.assessments))
        self.assertEqual(pipeline.assessor_names, tuple(item.assessor for item in result.assessments[0].results))
        self.assertEqual(0.8, result.assessments[0].variant_parameters["gamma"])
        self.assertTrue(all(np.array_equal(item.image, before) for item, before in zip(self.batch.variants, original)))

    def test_selection_and_runtime_parameters_are_applied(self) -> None:
        pipeline = VariantQualityAssessmentPipeline()
        result = pipeline.assess(
            (self.batch,),
            selected_assessors=("dynamic_range",),
            parameters={"dynamic_range": {"lower_percentile": 0, "upper_percentile": 100}},
        )[0]
        assessment = result.assessments[0].results[0]
        self.assertEqual("dynamic_range", assessment.assessor)
        self.assertEqual(0.0, assessment.parameters["lower_percentile"])

    def test_rejects_duplicate_and_unknown_assessors_and_bad_parameters(self) -> None:
        with self.assertRaises(ValueError):
            VariantQualityAssessmentPipeline((BrightnessAssessor(), BrightnessAssessor()))
        pipeline = VariantQualityAssessmentPipeline()
        with self.assertRaises(KeyError):
            pipeline.assess((self.batch,), selected_assessors=("unknown",))
        with self.assertRaises(KeyError):
            pipeline.assess((self.batch,), parameters={"unknown": {}})
        with self.assertRaises(TypeError):
            pipeline.assess((self.batch,), parameters={"brightness": 1})


if __name__ == "__main__":
    unittest.main()
