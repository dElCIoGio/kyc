from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from image_processor.context import VariantGenerationContext
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
from image_processor.models import VariantBatch


def build_default_context() -> VariantGenerationContext:
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


def process_image(path: str | Path) -> tuple[VariantBatch, ...]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return build_default_context().generate(image)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate in-memory preprocessing variants for an image."
    )
    parser.add_argument("image_path", help="Path to the image file to process.")
    args = parser.parse_args(argv)

    batches = process_image(args.image_path)
    variant_count = sum(len(batch.variants) for batch in batches)

    print(f"Generated {variant_count} variants across {len(batches)} batches.")
    for batch in batches:
        print(f"- {batch.generator}: {len(batch.variants)} variants")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
