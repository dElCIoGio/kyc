import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from image_processor.main import (
    _normalize_extension,
    _safe_path_part,
    assess_variant_batches,
    build_default_context,
    build_default_quality_pipeline,
    main,
    save_variant_batches,
)
from image_processor.models import VariantBatch, VariantInfo

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    OPENCV_AVAILABLE = False
else:
    OPENCV_AVAILABLE = True


class MainHelperTests(unittest.TestCase):
    def test_normalizes_extension(self) -> None:
        self.assertEqual(".png", _normalize_extension("png"))
        self.assertEqual(".jpg", _normalize_extension(".jpg"))

    def test_rejects_empty_extension(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_extension("")

    def test_safe_path_part_removes_path_separators(self) -> None:
        self.assertEqual("threshold_otsu", _safe_path_part("threshold/otsu"))
        self.assertEqual("variant", _safe_path_part("///"))

    def test_default_quality_pipeline_registers_all_assessors(self) -> None:
        pipeline = build_default_quality_pipeline()

        self.assertEqual(
            (
                "sharpness",
                "brightness",
                "contrast",
                "dynamic_range",
                "histogram_clipping",
            ),
            pipeline.assessor_names,
        )


@unittest.skipUnless(OPENCV_AVAILABLE, "cv2 is not installed")
class SaveVariantBatchTests(unittest.TestCase):
    def test_saves_variants_under_generator_directories(self) -> None:
        image = np.zeros((4, 4), dtype=np.uint8)
        batches = (
            VariantBatch(
                generator="gamma",
                variants=(
                    VariantInfo(
                        name="gamma_1.0",
                        image=image,
                        parameters={"gamma": 1.0},
                    ),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = save_variant_batches(batches, tmpdir)

            self.assertEqual(1, len(saved_paths))
            self.assertEqual("gamma", saved_paths[0].parent.name)
            self.assertEqual("gamma_1.0.png", saved_paths[0].name)
            self.assertTrue(saved_paths[0].exists())

    def test_assesses_every_generated_variant(self) -> None:
        image = np.zeros((4, 4), dtype=np.uint8)
        batches = (
            VariantBatch(
                generator="original",
                variants=(
                    VariantInfo(
                        name="original",
                        image=image,
                        parameters={},
                    ),
                ),
            ),
        )

        assessments = assess_variant_batches(batches)

        self.assertEqual(1, len(assessments))
        self.assertEqual("original", assessments[0].generator)
        self.assertEqual(1, len(assessments[0].assessments))
        self.assertEqual(
            (
                "sharpness",
                "brightness",
                "contrast",
                "dynamic_range",
                "histogram_clipping",
            ),
            tuple(
                result.assessor
                for result in assessments[0].assessments[0].results
            ),
        )

    def test_main_prints_quality_assessments_for_generated_variants(self) -> None:
        import cv2

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        batches = build_default_context().generate(image)
        variant_count = sum(len(batch.variants) for batch in batches)
        assessor_count = len(build_default_quality_pipeline().assessor_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))

            output = StringIO()
            with redirect_stdout(output):
                exit_code = main([str(image_path)])

        self.assertEqual(0, exit_code)
        text = output.getvalue()
        self.assertIn(
            f"Generated {variant_count} variants across {len(batches)} batches.",
            text,
        )
        self.assertIn(
            f"Assessed {variant_count} variants with "
            f"{variant_count * assessor_count} quality results.",
            text,
        )
        self.assertIn("sharpness(laplacian_variance=", text)
        self.assertIn("histogram_clipping(", text)


if __name__ == "__main__":
    unittest.main()
