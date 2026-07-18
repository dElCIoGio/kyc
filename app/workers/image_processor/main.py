from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_processor.context import VariantGenerationContext
from image_processor.models import VariantBatch
from image_processor.quality import (
    BrightnessAssessor,
    ContrastAssessor,
    DynamicRangeAssessor,
    HistogramClippingAssessor,
    SharpnessAssessor,
    VariantBatchAssessment,
    VariantQualityAssessmentPipeline,
)
from image_processor.quality.models import AssessmentResult


def build_default_context() -> VariantGenerationContext:
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

    return VariantGenerationContext(
        generators=(
            OriginalVariantGenerator(),
            GrayscaleVariantGenerator(),
            ClaheVariantGenerator(),
            GammaVariantGenerator(),
            ContrastBrightnessVariantGenerator(),
            SharpeningVariantGenerator(),
            DenoisingVariantGenerator(),
            ThresholdVariantGenerator(),
        )
    )


def build_default_quality_pipeline() -> VariantQualityAssessmentPipeline:
    return VariantQualityAssessmentPipeline(
        assessors=(
            SharpnessAssessor(),
            BrightnessAssessor(),
            ContrastAssessor(),
            DynamicRangeAssessor(),
            HistogramClippingAssessor(),
        )
    )


def process_image(path: str | Path) -> tuple[VariantBatch, ...]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return build_default_context().generate(image)


def assess_variant_batches(
    batches: Iterable[VariantBatch],
) -> tuple[VariantBatchAssessment, ...]:
    pipeline = build_default_quality_pipeline()
    return tuple(pipeline.assess(batch) for batch in batches)


def save_variant_batches(
    batches: Iterable[VariantBatch],
    output_dir: str | Path,
    *,
    extension: str = ".png",
) -> tuple[Path, ...]:
    import cv2

    normalized_extension = _normalize_extension(extension)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    used_paths: set[Path] = set()

    for batch in batches:
        batch_dir = output_path / _safe_path_part(batch.generator)
        batch_dir.mkdir(parents=True, exist_ok=True)

        for variant in batch.variants:
            filename = f"{_safe_path_part(variant.name)}{normalized_extension}"
            variant_path = _unique_path(batch_dir / filename, used_paths)

            if not cv2.imwrite(str(variant_path), variant.image):
                raise OSError(f"Could not save variant image: {variant_path}")

            used_paths.add(variant_path)
            saved_paths.append(variant_path)

    return tuple(saved_paths)


def _normalize_extension(extension: str) -> str:
    if not extension:
        raise ValueError("Image extension cannot be empty")

    return extension if extension.startswith(".") else f".{extension}"


def _safe_path_part(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe_value or "variant"


def _unique_path(path: Path, used_paths: set[Path]) -> Path:
    if path not in used_paths:
        return path

    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if candidate not in used_paths:
            return candidate

    raise RuntimeError(f"Could not create a unique filename for {path}")


def _format_assessment_result(result: AssessmentResult) -> str:
    metrics = ", ".join(
        f"{name}={_format_metric_value(value)}"
        for name, value in result.metrics.items()
    )
    return f"{result.assessor}({metrics})"


def _format_metric_value(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate preprocessing variants for an image."
    )
    parser.add_argument("image_path", help="Path to the image file to process.")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory where generated variants should be saved.",
    )
    parser.add_argument(
        "--extension",
        default=".png",
        help="Image extension to use when saving variants. Defaults to .png.",
    )
    args = parser.parse_args(argv)

    batches = process_image(args.image_path)
    variant_count = sum(len(batch.variants) for batch in batches)
    assessments = assess_variant_batches(batches)
    assessment_count = sum(
        len(variant_assessment.results)
        for batch_assessment in assessments
        for variant_assessment in batch_assessment.assessments
    )

    saved_paths: tuple[Path, ...] = ()
    if args.output_dir:
        saved_paths = save_variant_batches(
            batches,
            args.output_dir,
            extension=args.extension,
        )

    print(f"Generated {variant_count} variants across {len(batches)} batches.")
    for batch in batches:
        print(f"- {batch.generator}: {len(batch.variants)} variants")

    print(
        f"Assessed {variant_count} variants with {assessment_count} quality results."
    )
    for batch_assessment in assessments:
        for variant_assessment in batch_assessment.assessments:
            results = "; ".join(
                _format_assessment_result(result)
                for result in variant_assessment.results
            )
            print(
                f"- {batch_assessment.generator}/{variant_assessment.variant_name}: "
                f"{results}"
            )

    if saved_paths:
        print(f"Saved {len(saved_paths)} variants to {Path(args.output_dir).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
