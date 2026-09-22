from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from kyc_engine.contracts import Image, Quadrilateral
from kyc_engine.detection import OpenCVDocumentDetector
from kyc_engine.fields import FieldLocalizer
from kyc_engine.intake import ImageIntake
from kyc_engine.normalization import DocumentNormalizer
from kyc_engine.profiles import load_default_profile
from kyc_engine.quality import QualityAssessmentPipeline
from kyc_engine.variants import BalancedVariantPolicy


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate private local debug images for the Angolan ID front profile."
    )
    parser.add_argument("image", type=Path, help="Path to a private front-side card image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("private-data/ao-id-front/debug"),
        help="Ignored local directory for generated debug images.",
    )
    args = parser.parse_args()

    profile = load_default_profile()
    intake = ImageIntake().load(args.image)
    detection, geometry_debug = OpenCVDocumentDetector().detect_with_debug(intake.image)
    normalized = DocumentNormalizer().normalize(intake.image, detection, profile)
    batches = BalancedVariantPolicy().generate(normalized.image)
    assessments = QualityAssessmentPipeline().assess(batches)
    crops = FieldLocalizer().localize(batches, profile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_overlay = intake.image.copy()
    _draw_quadrilateral(
        geometry_overlay,
        geometry_debug.coarse_corners,
        color=(0, 165, 255),
        label="coarse",
    )
    _draw_quadrilateral(
        geometry_overlay,
        geometry_debug.final_corners,
        color=(0, 255, 0),
        label="final",
    )
    cv2.imwrite(str(args.output_dir / "front_geometry_overlay.png"), geometry_overlay)
    cv2.imwrite(str(args.output_dir / "normalized_front.png"), normalized.image)

    overlay = normalized.image.copy()
    for field in profile.fields:
        box = field.bounding_box
        cv2.rectangle(
            overlay,
            (box.x, box.y),
            (box.right, box.bottom),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            overlay,
            field.name,
            (box.x, max(12, box.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(args.output_dir / "field_overlay.png"), overlay)

    print(f"detection_confidence={detection.confidence:.4f}")
    print(f"candidate_type={geometry_debug.candidate_kind}")
    print(f"refinement_accepted={geometry_debug.refinement_accepted}")
    print(f"edge_support={dict(geometry_debug.edge_support)}")
    print(f"components={dict(detection.confidence_components)}")
    print(
        "corners="
        + repr([(round(point.x, 1), round(point.y, 1)) for point in detection.corners.points])
    )
    print(f"normalized={normalized.width}x{normalized.height}")
    print(f"variant_batches={[(batch.generator, len(batch.variants)) for batch in batches]}")
    print(f"quality_assessments={sum(len(batch.assessments) for batch in assessments)}")
    print(f"field_crops={len(crops)}")
    print(f"debug_dir={args.output_dir}")
    return 0


def _draw_quadrilateral(
    image: Image,
    corners: Quadrilateral,
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    points = corners.as_array().round().astype("int32")
    cv2.polylines(image, (points,), True, color, 3, cv2.LINE_AA)
    x, y = points[0]
    cv2.putText(
        image,
        label,
        (int(x), max(18, int(y) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    raise SystemExit(main())
