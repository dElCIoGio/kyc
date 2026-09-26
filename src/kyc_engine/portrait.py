from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .contracts import (
    BoundingBox,
    FaceCandidate,
    FaceDetectionResult,
    FaceDetector,
    Image,
    PortraitDefinition,
    PortraitExtractionResult,
    PortraitStatus,
)
from .portrait_artifacts import InMemoryPortraitArtifactStore


class OpenCVHaarFaceDetector:
    """Lightweight crop-local face detector, not an identity verification model."""

    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: tuple[int, int] = (64, 64),
    ) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError("OpenCV frontal-face cascade is unavailable")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_face_size = min_face_size

    def detect(self, image: Image) -> FaceDetectionResult:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Portrait crop must have one, three, or four channels")
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_face_size,
        )
        return FaceDetectionResult(
            tuple(
                FaceCandidate(BoundingBox(int(x), int(y), int(width), int(height)))
                for x, y, width, height in faces
            )
        )


@dataclass(frozen=True)
class PortraitExtractionConfig:
    min_crop_width: int = 128
    min_crop_height: int = 128
    min_face_width: int = 64
    min_face_height: int = 64


class PortraitExtractor:
    def __init__(
        self,
        detector: FaceDetector,
        artifacts: InMemoryPortraitArtifactStore,
        config: PortraitExtractionConfig | None = None,
    ) -> None:
        self._detector = detector
        self._artifacts = artifacts
        self._config = config or PortraitExtractionConfig()

    def extract(
        self,
        image: Image,
        definition: PortraitDefinition | None,
    ) -> PortraitExtractionResult:
        if definition is None:
            return _result(PortraitStatus.NOT_CONFIGURED, None, None, warnings=())
        try:
            height, width = image.shape[:2]
            region = _clamp(definition, width, height)
        except Exception:
            return _result(PortraitStatus.CROP_INVALID, definition, None, warnings=("PORTRAIT_CROP_INVALID",))
        if region is None:
            return _result(PortraitStatus.CROP_INVALID, definition, None, warnings=("PORTRAIT_CROP_INVALID",))
        if region.width < self._config.min_crop_width or region.height < self._config.min_crop_height:
            return _result(PortraitStatus.TOO_SMALL, definition, region, warnings=("PORTRAIT_TOO_SMALL",))

        crop = image[region.y : region.bottom, region.x : region.right].copy()
        if crop.size == 0:
            return _result(PortraitStatus.CROP_INVALID, definition, region, warnings=("PORTRAIT_CROP_INVALID",))
        try:
            artifact_id = self._artifacts.put(crop)
        except Exception:
            return _result(PortraitStatus.ARTIFACT_FAILED, definition, region, warnings=("PORTRAIT_ARTIFACT_PERSIST_FAILED",))
        try:
            faces = self._detector.detect(crop).candidates
            if not faces:
                return _result(
                    PortraitStatus.FACE_NOT_FOUND,
                    definition,
                    region,
                    artifact_id=artifact_id,
                    warnings=("PORTRAIT_FACE_NOT_FOUND",),
                )
            if len(faces) > 1:
                return _result(
                    PortraitStatus.MULTIPLE_FACES,
                    definition,
                    region,
                    artifact_id=artifact_id,
                    face_count=len(faces),
                    warnings=("PORTRAIT_MULTIPLE_FACES",),
                )
            face = faces[0]
            if face.bounding_box.right > region.width or face.bounding_box.bottom > region.height:
                raise ValueError("Face detector returned a box outside the portrait crop")
            if (
                face.bounding_box.width < self._config.min_face_width
                or face.bounding_box.height < self._config.min_face_height
            ):
                return _result(
                    PortraitStatus.FACE_TOO_SMALL,
                    definition,
                    region,
                    artifact_id=artifact_id,
                    face=face,
                    warnings=("PORTRAIT_FACE_TOO_SMALL",),
                )
            return _result(
                PortraitStatus.AVAILABLE,
                definition,
                region,
                artifact_id=artifact_id,
                face=face,
                eligible=True,
            )
        except Exception:
            return _result(
                PortraitStatus.DETECTOR_FAILED,
                definition,
                region,
                artifact_id=artifact_id,
                warnings=("PORTRAIT_DETECTOR_FAILED",),
            )


def _clamp(definition: PortraitDefinition, image_width: int, image_height: int) -> BoundingBox | None:
    left = max(0, definition.x)
    top = max(0, definition.y)
    right = min(image_width, definition.right)
    bottom = min(image_height, definition.bottom)
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left, top, right - left, bottom - top)


def _result(
    status: PortraitStatus,
    requested_region: PortraitDefinition | None,
    clamped_region: BoundingBox | None,
    *,
    artifact_id: str | None = None,
    face: FaceCandidate | None = None,
    face_count: int | None = None,
    eligible: bool = False,
    warnings: tuple[str, ...] = (),
) -> PortraitExtractionResult:
    count = face_count if face_count is not None else (1 if face is not None else 0)
    return PortraitExtractionResult(
        status=status,
        requested_region=requested_region,
        clamped_region=clamped_region,
        face_detected=count > 0,
        face_count=count,
        face=face,
        eligible_for_face_match=eligible,
        warnings=warnings,
        artifact_id=artifact_id,
    )
