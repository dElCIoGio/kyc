from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from kyc_engine.liveness import LivenessEvaluationConfig, LivenessEvaluator
from kyc_engine.minifasnet import (
    MiniFASNetAntiSpoofDetector,
    MiniFASNetNoFaceError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Development webcam test for passive liveness."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("private-models/liveness/minifasnet"),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=3,
        help="Number of frames evaluated per liveness attempt.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="Seconds between sampled frames.",
    )
    return parser.parse_args()


def draw_text(
    frame,
    text: str,
    y: int,
    colour=(255, 255, 255),
    scale: float = 0.65,
):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be greater than 0")

    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be greater than 0")

    print("Loading MiniFASNet models...")

    detector = MiniFASNetAntiSpoofDetector(args.model_dir)

    config = LivenessEvaluationConfig(
        frame_count=args.frames,
        minimum_real_ratio=2.0 / 3.0,
    )

    evaluator = LivenessEvaluator(
        detector=detector,
        config=config,
    )

    print("Models loaded.")
    print("SPACE = start liveness attempt")
    print("Q = quit")

    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    collecting = False
    collected_frames = []
    next_sample_at = 0.0

    final_result = None
    result_display_until = 0.0

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Failed to read webcam frame.")
                break

            now = time.monotonic()

            # -----------------------------------
            # Capture frames for liveness attempt
            # -----------------------------------

            if collecting and now >= next_sample_at:
                collected_frames.append(frame.copy())

                print(f"Captured frame {len(collected_frames)}/{args.frames}")

                next_sample_at = now + args.sample_interval

                if len(collected_frames) >= args.frames:
                    collecting = False

                    print("Evaluating liveness...")

                    started = time.perf_counter()

                    try:
                        final_result = evaluator.evaluate(collected_frames)

                        elapsed_ms = (time.perf_counter() - started) * 1000

                        print()
                        print("Liveness result")
                        print("----------------")
                        print(f"passed:           {final_result.passed}")
                        print(f"passive_score:    {final_result.passive_score:.6f}")
                        print(f"frames_evaluated: {final_result.frames_evaluated}")
                        print(f"real_frames:      {final_result.real_frames}")
                        print(f"inference time:   {elapsed_ms:.0f} ms")
                        print()

                    except MiniFASNetNoFaceError:
                        final_result = None
                        print("Liveness attempt failed: no face detected.")

                    collected_frames = []
                    result_display_until = now + 4.0

            # -------------------------
            # Draw webcam UI
            # -------------------------

            if collecting:
                draw_text(
                    frame,
                    "CHECKING LIVENESS...",
                    40,
                    (0, 255, 255),
                    0.8,
                )

                draw_text(
                    frame,
                    (f"Samples: {len(collected_frames)}/{args.frames}"),
                    75,
                )

                draw_text(
                    frame,
                    "Keep your face visible",
                    110,
                )

            elif final_result is not None and now < result_display_until:
                if final_result.passed:
                    colour = (0, 255, 0)
                    status = "LIVENESS PASSED"
                else:
                    colour = (0, 0, 255)
                    status = "LIVENESS FAILED"

                draw_text(
                    frame,
                    status,
                    40,
                    colour,
                    0.8,
                )

                draw_text(
                    frame,
                    f"Real score: {final_result.passive_score:.3f}",
                    75,
                    colour,
                )

                draw_text(
                    frame,
                    (
                        f"Real frames: "
                        f"{final_result.real_frames}/"
                        f"{final_result.frames_evaluated}"
                    ),
                    110,
                    colour,
                )

            else:
                draw_text(
                    frame,
                    "Press SPACE to check liveness",
                    40,
                )

            draw_text(
                frame,
                "Q = quit",
                frame.shape[0] - 20,
                (255, 255, 255),
                0.5,
            )

            cv2.imshow(
                "KYC - Passive Liveness Test",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" ") and not collecting:
                print()
                print("Starting liveness attempt...")

                collected_frames = []
                final_result = None
                collecting = True
                next_sample_at = time.monotonic()

    finally:
        camera.release()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
