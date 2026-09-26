from pathlib import Path
import time

import cv2


OUTPUT = Path("private-data/liveness")
OUTPUT.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

print("Look at the camera.")
print("Capturing 3 liveness test frames...")

try:
    # Give the camera time to adjust exposure/focus.
    time.sleep(1)

    for index in range(1, 4):
        ok, frame = camera.read()

        if not ok:
            raise RuntimeError("Failed to read webcam frame")

        path = OUTPUT / f"frame-{index}.jpg"

        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write {path}")

        print(f"Captured {path}")
        time.sleep(0.5)

finally:
    camera.release()

print("Done.")
