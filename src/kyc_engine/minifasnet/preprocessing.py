"""MiniFASNet inference preprocessing adapted from Silent-Face-Anti-Spoofing.

Source: Minivision Silent-Face-Anti-Spoofing, Apache-2.0. This file retains
only the inference crop and tensor-value behavior required by its checkpoints.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .errors import MiniFASNetInitializationError, MiniFASNetInputError, MiniFASNetNoFaceError


FaceBox = tuple[int, int, int, int]


def validate_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise MiniFASNetInputError("MiniFASNet requires a uint8 image frame")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise MiniFASNetInputError("MiniFASNet requires a non-empty BGR frame")
    return frame


class CaffeFaceRegionDetector:
    """Minimal upstream RetinaFace Caffe bbox extraction for anti-spoof crops."""

    def __init__(self, deploy_path: str, weights_path: str) -> None:
        try:
            self._detector = cv2.dnn.readNetFromCaffe(deploy_path, weights_path)
        except cv2.error as exc:
            raise MiniFASNetInitializationError("MiniFASNet face detector could not be initialized") from exc

    def detect(self, frame: np.ndarray) -> FaceBox:
        frame = validate_frame(frame)
        height, width = frame.shape[:2]
        detector_input = frame
        if width * height >= 192 * 192:
            aspect_ratio = width / height
            detector_input = cv2.resize(
                frame,
                (
                    int(192 * math.sqrt(aspect_ratio)),
                    int(192 / math.sqrt(aspect_ratio)),
                ),
                interpolation=cv2.INTER_LINEAR,
            )
        try:
            blob = cv2.dnn.blobFromImage(detector_input, 1, mean=(104, 117, 123))
            self._detector.setInput(blob, "data")
            detections = np.asarray(self._detector.forward("detection_out")).squeeze()
        except cv2.error as exc:
            raise MiniFASNetNoFaceError("MiniFASNet face detection failed") from exc
        if detections.size == 0 or detections.size % 7:
            raise MiniFASNetNoFaceError("MiniFASNet found no usable face region")
        detections = detections.reshape(-1, 7)
        row = detections[int(np.argmax(detections[:, 2]))]
        left, top, right, bottom = row[3] * width, row[4] * height, row[5] * width, row[6] * height
        box = (int(left), int(top), int(right - left + 1), int(bottom - top + 1))
        if box[2] <= 0 or box[3] <= 0:
            raise MiniFASNetNoFaceError("MiniFASNet found no usable face region")
        return box


def crop_frame(
    frame: np.ndarray,
    bbox: FaceBox,
    *,
    scale: float | None,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Apply the upstream original-image or expanded-face crop semantics."""
    frame = validate_frame(frame)
    if output_width <= 0 or output_height <= 0:
        raise MiniFASNetInputError("MiniFASNet requires a positive model input size")
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise MiniFASNetInputError("MiniFASNet requires a valid face crop scale")
    if scale is None:
        return cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    source_height, source_width = frame.shape[:2]
    left, top, right, bottom = _scaled_box(source_width, source_height, bbox, scale)
    crop = frame[top : bottom + 1, left : right + 1]
    if crop.size == 0:
        raise MiniFASNetNoFaceError("MiniFASNet found no usable face region")
    return cv2.resize(crop, (output_width, output_height), interpolation=cv2.INTER_LINEAR)


def tensor_values(frame: np.ndarray) -> np.ndarray:
    """Match upstream ToTensor: BGR HWC -> CHW float32, preserving 0-255 values."""
    frame = validate_frame(frame)
    return np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32)


def _scaled_box(source_width: int, source_height: int, bbox: FaceBox, scale: float) -> tuple[int, int, int, int]:
    try:
        valid_bbox = len(bbox) == 4 and all(isinstance(value, (int, np.integer)) for value in bbox)
    except TypeError:
        valid_bbox = False
    if not valid_bbox:
        raise MiniFASNetNoFaceError("MiniFASNet found no usable face region")
    x, y, box_width, box_height = bbox
    if box_width <= 0 or box_height <= 0:
        raise MiniFASNetNoFaceError("MiniFASNet found no usable face region")
    scale = min((source_height - 1) / box_height, min((source_width - 1) / box_width, scale))
    new_width, new_height = box_width * scale, box_height * scale
    center_x, center_y = box_width / 2 + x, box_height / 2 + y
    left, top = center_x - new_width / 2, center_y - new_height / 2
    right, bottom = center_x + new_width / 2, center_y + new_height / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > source_width - 1:
        left -= right - source_width + 1
        right = source_width - 1
    if bottom > source_height - 1:
        top -= bottom - source_height + 1
        bottom = source_height - 1
    return int(left), int(top), int(right), int(bottom)
