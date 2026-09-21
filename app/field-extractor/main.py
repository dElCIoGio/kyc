"""
run_extraction.py
==================

Small CLI runner that wires `field_extractor.py` up to real files on disk.

Usage:
    python run_extraction.py --images-dir ./variants --fields-config ./fields.json

Expects:
    --images-dir     A directory containing the image variants for ONE document.
                      Every image in it is treated as a variant of the same
                      document, already cropped consistently, so one set of
                      field locations applies to all of them. The variant id
                      used in the output is each file's name (without
                      extension).
    --fields-config  A JSON file describing the fields to extract and where
                      they are (see fields_config.example.json alongside this
                      script for the format).

Prints a per-variant, per-field summary to stdout, and optionally writes the
full results (value + confidence + error per field per variant) as JSON via
--output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from field_extractor import (
    BoundingBox,
    FieldDefinition,
    FieldExtractor,
    ImageVariant,
    PaddleOCREngine,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_field_definitions(config_path: Path) -> List[FieldDefinition]:
    """Load field locations from a JSON config file.

    Expected format:
        {
          "fields": [
            {"name": "full_name", "x": 20, "y": 15, "width": 250, "height": 40},
            {"name": "address", "x": 20, "y": 215, "width": 400, "height": 40, "multi_line": true}
          ]
        }
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    try:
        raw_fields = raw["fields"]
    except KeyError as exc:
        raise ValueError(f"{config_path} must have a top-level 'fields' list") from exc

    field_definitions = []
    for entry in raw_fields:
        try:
            field_definitions.append(
                FieldDefinition(
                    name=entry["name"],
                    bounding_box=BoundingBox(
                        x=entry["x"],
                        y=entry["y"],
                        width=entry["width"],
                        height=entry["height"],
                    ),
                    multi_line=entry.get("multi_line", False),
                    description=entry.get("description"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Field entry missing required key {exc} in: {entry}") from exc

    return field_definitions


def load_image_variants(images_dir: Path) -> List[ImageVariant]:
    """Build one ImageVariant per image file found directly in `images_dir`.

    The variant id is the file's stem (name without extension), sorted
    alphabetically for a stable, predictable order.
    """
    paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(
            f"No image files found in {images_dir} (looked for {sorted(IMAGE_EXTENSIONS)})"
        )
    return [ImageVariant(variant_id=p.stem, image=p) for p in paths]


def print_summary(extractor: FieldExtractor, results) -> None:
    """Print a compact, human-readable per-variant/per-field summary."""
    for image_result in results:
        print(f"\n=== variant: {image_result.variant_id} ===")
        for name in extractor.field_names:
            result = image_result.get(name)
            if result.error:
                print(f"  {name:<20} ERROR: {result.error}")
            else:
                value_preview = result.value.replace("\n", " \\n ")
                print(f"  {name:<20} {value_preview!r}  (confidence={result.confidence:.2f})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True, type=Path, help="Directory of image variants for one document")
    parser.add_argument("--fields-config", required=True, type=Path, help="JSON file describing field locations")
    parser.add_argument("--lang", default="pt", help="PaddleOCR language code (default: en)")
    parser.add_argument("--confidence-strategy", choices=["min", "mean"], default="min")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write full results as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO-level logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if not args.images_dir.is_dir():
        parser.error(f"--images-dir does not exist or is not a directory: {args.images_dir}")
    if not args.fields_config.is_file():
        parser.error(f"--fields-config does not exist: {args.fields_config}")

    try:
        field_definitions = load_field_definitions(args.fields_config)
        variants = load_image_variants(args.images_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    logger.info("Loaded %d field(s) and %d variant(s)", len(field_definitions), len(variants))

    engine = PaddleOCREngine(lang=args.lang, confidence_strategy=args.confidence_strategy)
    extractor = FieldExtractor(field_definitions=field_definitions, ocr_engine=engine)

    results = extractor.extract_fields_batch(variants)

    print_summary(extractor, results)

    if args.output:
        payload = [r.to_dict() for r in results]
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nFull results written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())