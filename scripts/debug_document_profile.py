from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from kyc_engine.contracts import Image, Quadrilateral
from kyc_engine.detection import OpenCVDocumentDetector
from kyc_engine.intake import ImageIntake
from kyc_engine.normalization import DocumentNormalizer
from kyc_engine.profiles import load_default_profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write ignored local normalization and field-geometry debug images."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--side", choices=("front", "back"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--show-labels", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profile = next(profile for profile in load_default_profiles() if profile.side == args.side)
    input_image = ImageIntake().load(args.image)
    detection, geometry_debug = OpenCVDocumentDetector(side=args.side).detect_with_debug(
        input_image.image
    )
    normalized = DocumentNormalizer().normalize(input_image.image, detection, profile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry = input_image.image.copy()
    _draw_quadrilateral(
        geometry,
        geometry_debug.coarse_corners,
        color=(0, 165, 255),
        label="coarse",
    )
    _draw_quadrilateral(
        geometry,
        geometry_debug.final_corners,
        color=(0, 255, 0),
        label="final",
    )
    cv2.imwrite(str(args.output_dir / f"{args.side}_geometry_overlay.png"), geometry)
    cv2.imwrite(str(args.output_dir / f"normalized_{args.side}.png"), normalized.image)

    legend_width = 180 if args.show_labels else 0
    if args.show_labels:
        overlay = cv2.copyMakeBorder(
            normalized.image,
            0,
            0,
            legend_width,
            0,
            cv2.BORDER_CONSTANT,
            value=(250, 250, 250),
        )
    else:
        overlay = normalized.image.copy()

    for index, field in enumerate(profile.fields, start=1):
        box = field.bounding_box
        left = box.x + legend_width
        right = box.right + legend_width
        cv2.rectangle(overlay, (left, box.y), (right, box.bottom), (0, 0, 255), 1)
        if args.show_labels:
            cv2.putText(
                overlay,
                f"{index}. {field.name}",
                (8, 20 + (index - 1) * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
    if profile.qr_code is not None:
        box = profile.qr_code.bounding_box
        cv2.rectangle(
            overlay,
            (box.x + legend_width, box.y),
            (box.right + legend_width, box.bottom),
            (255, 0, 0),
            1,
        )
        if args.show_labels:
            cv2.putText(
                overlay,
                f"{len(profile.fields) + 1}. qr_code",
                (8, 20 + len(profile.fields) * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(args.output_dir / f"{args.side}_field_overlay.png"), overlay)

    print(f"detection_confidence={detection.confidence:.4f}")
    print(f"candidate_type={geometry_debug.candidate_kind}")
    print(f"refinement_accepted={geometry_debug.refinement_accepted}")
    print(f"edge_support={dict(geometry_debug.edge_support)}")
    print(
        "corners="
        + repr(
            [
                (round(point.x, 1), round(point.y, 1))
                for point in detection.corners.points
            ]
        )
    )
    print(f"normalized={normalized.width}x{normalized.height}")
    print(f"field_count={len(profile.fields)}")
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
