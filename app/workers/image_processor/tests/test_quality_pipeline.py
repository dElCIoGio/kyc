import sys
import unittest
from pathlib import Path
from typing import Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from image_processor.models import Image, VariantBatch, VariantInfo
from image_processor.quality import (
    AssessmentResult,
    VariantQualityAssessmentPipeline,
)


class RecordingAssessor:
    def __init__(self, name: str, *, mutate_image: bool = False) -> None:
        self.name = name
        self.mutate_image = mutate_image
        self.received_parameters: list[Mapping[str, object] | None] = []

    def assess(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> AssessmentResult:
        self.received_parameters.append(parameters)
        if self.mutate_image:
            image.fill(255)
        return AssessmentResult(
            assessor=self.name,
            metrics={"mean": float(np.mean(image))},
            parameters=dict(parameters or {}),
        )


def make_batch() -> VariantBatch:
    return VariantBatch(
        generator="gamma",
        variants=(
            VariantInfo(
                name="gamma_0.8",
                image=np.zeros((4, 4, 3), dtype=np.uint8),
                parameters={"gamma": 0.8},
            ),
            VariantInfo(
                name="gamma_1.2",
                image=np.full((4, 4, 3), 100, dtype=np.uint8),
                parameters={"gamma": 1.2},
            ),
        ),
    )


class VariantQualityAssessmentPipelineTests(unittest.TestCase):
    def test_runs_all_assessors_for_every_variant_in_registration_order(self) -> None:
        pipeline = VariantQualityAssessmentPipeline(
            assessors=(RecordingAssessor("first"), RecordingAssessor("second"))
        )

        result = pipeline.assess(make_batch())

        self.assertEqual(("first", "second"), pipeline.assessor_names)
        self.assertEqual(2, len(result.assessments))
        for assessment in result.assessments:
            self.assertEqual(
                ("first", "second"),
                tuple(item.assessor for item in assessment.results),
            )

    def test_runs_only_selected_assessors_in_selected_order(self) -> None:
        pipeline = VariantQualityAssessmentPipeline(
            assessors=(RecordingAssessor("first"), RecordingAssessor("second"))
        )

        result = pipeline.assess(
            make_batch(),
            selected_assessors=("second",),
        )

        for assessment in result.assessments:
            self.assertEqual(
                ("second",),
                tuple(item.assessor for item in assessment.results),
            )

    def test_preserves_batch_and_variant_metadata(self) -> None:
        result = VariantQualityAssessmentPipeline(
            assessors=(RecordingAssessor("metric"),)
        ).assess(make_batch())

        self.assertEqual("gamma", result.generator)
        self.assertEqual(
            ("gamma_0.8", "gamma_1.2"),
            tuple(item.variant_name for item in result.assessments),
        )
        self.assertEqual("gamma", result.assessments[0].generator)
        self.assertEqual(
            {"gamma": 0.8},
            dict(result.assessments[0].variant_parameters),
        )

    def test_passes_runtime_parameters_to_the_correct_assessor(self) -> None:
        first = RecordingAssessor("first")
        second = RecordingAssessor("second")
        pipeline = VariantQualityAssessmentPipeline(assessors=(first, second))

        pipeline.assess(
            make_batch(),
            parameters={"second": {"threshold": 12}},
        )

        self.assertEqual([None, None], first.received_parameters)
        self.assertEqual(
            [{"threshold": 12}, {"threshold": 12}],
            second.received_parameters,
        )

    def test_rejects_duplicate_assessor_names(self) -> None:
        with self.assertRaises(ValueError):
            VariantQualityAssessmentPipeline(
                assessors=(RecordingAssessor("same"), RecordingAssessor("same"))
            )

    def test_rejects_empty_assessor_names(self) -> None:
        with self.assertRaises(ValueError):
            VariantQualityAssessmentPipeline(assessors=(RecordingAssessor("  "),))

    def test_rejects_unknown_selected_assessors(self) -> None:
        pipeline = VariantQualityAssessmentPipeline(
            assessors=(RecordingAssessor("known"),)
        )

        with self.assertRaises(KeyError):
            pipeline.assess(make_batch(), selected_assessors=("missing",))

    def test_rejects_parameters_for_unknown_assessors(self) -> None:
        pipeline = VariantQualityAssessmentPipeline(
            assessors=(RecordingAssessor("known"),)
        )

        with self.assertRaises(KeyError):
            pipeline.assess(make_batch(), parameters={"missing": {}})

    def test_rejects_malformed_top_level_parameters(self) -> None:
        pipeline = VariantQualityAssessmentPipeline(
            assessors=(RecordingAssessor("known"),)
        )

        with self.assertRaises(TypeError):
            pipeline.assess(
                make_batch(),
                parameters=["not", "a", "mapping"],  # type: ignore[arg-type]
            )

    def test_rejects_malformed_per_assessor_parameters(self) -> None:
        pipeline = VariantQualityAssessmentPipeline(
            assessors=(RecordingAssessor("known"),)
        )

        with self.assertRaises(TypeError):
            pipeline.assess(
                make_batch(),
                parameters={
                    "known": ["not", "a", "mapping"],  # type: ignore[dict-item]
                },
            )

    def test_rejects_invalid_batches(self) -> None:
        pipeline = VariantQualityAssessmentPipeline(
            assessors=(RecordingAssessor("known"),)
        )
        invalid_batch = VariantBatch(
            generator="gamma",
            variants=(
                VariantInfo(
                    name="invalid",
                    image=np.zeros((4, 4), dtype=np.float32),  # type: ignore[arg-type]
                    parameters={},
                ),
            ),
        )

        with self.assertRaises(ValueError):
            pipeline.assess(invalid_batch)

    def test_source_images_are_protected_from_assessor_mutation(self) -> None:
        batch = make_batch()
        originals = tuple(variant.image.copy() for variant in batch.variants)
        pipeline = VariantQualityAssessmentPipeline(
            assessors=(RecordingAssessor("mutating", mutate_image=True),)
        )

        pipeline.assess(batch)

        for original, variant in zip(originals, batch.variants, strict=True):
            self.assertTrue(np.array_equal(original, variant.image))


if __name__ == "__main__":
    unittest.main()
