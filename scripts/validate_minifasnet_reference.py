"""Optional local MiniFASNet parity check; excluded from normal test discovery.

Run with a Python environment containing the ``liveness`` extra and explicitly
point at a private model root and the upstream sample-image directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from kyc_engine.minifasnet import MiniFASNetAntiSpoofDetector


_REFERENCE = {
    "image_F1.jpg": (False, 0.072331),
    "image_F2.jpg": (False, 0.181286),
    "image_T1.jpg": (True, 0.993568),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local MiniFASNet parity with published sample inputs")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    arguments = parser.parse_args()

    detector = MiniFASNetAntiSpoofDetector(arguments.model_dir)
    failures: list[str] = []
    for filename, (expected_real, expected_score) in _REFERENCE.items():
        frame = cv2.imread(str(arguments.images_dir / filename))
        if frame is None:
            failures.append(f"{filename}: unavailable")
            continue
        actual = detector.detect(frame)
        delta = abs(actual.score - expected_score)
        print(f"{filename}: real={actual.is_real} score={actual.score:.6f} delta={delta:.6f}")
        if actual.is_real != expected_real or delta > arguments.tolerance:
            failures.append(filename)
    if failures:
        raise SystemExit(f"MiniFASNet reference validation failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
