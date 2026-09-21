from __future__ import annotations

import cv2
import numpy as np

from .contracts import DetectionResult, DocumentProfile, Image, NormalizedDocument


class NormalizationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DocumentNormalizer:
    def normalize(
        self,
        image: Image,
        detection: DetectionResult,
        profile: DocumentProfile,
    ) -> NormalizedDocument:
        source = detection.corners.as_array()
        self._validate_corners(source, image)
        width = profile.canonical_width
        height = profile.canonical_height
        destination = np.asarray(
            ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)),
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        if detection.orientation_degrees == 180:
            rotation = np.asarray(
                ((-1.0, 0.0, width - 1.0), (0.0, -1.0, height - 1.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            matrix = rotation @ matrix
        elif detection.orientation_degrees != 0:
            raise NormalizationError(
                "UNSUPPORTED_ORIENTATION",
                "Only 0 and 180 degree document orientation are supported",
            )

        warped = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if warped.size == 0:
            raise NormalizationError("EMPTY_CROP", "Document normalization produced an empty image")
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:
            raise NormalizationError("INVALID_TRANSFORM", "Document transform is not invertible") from exc

        return NormalizedDocument(
            image=warped,
            profile_id=profile.profile_id,
            width=width,
            height=height,
            forward_transform=_matrix_tuple(matrix),
            inverse_transform=_matrix_tuple(inverse),
        )

    @staticmethod
    def _validate_corners(corners: np.ndarray, image: Image) -> None:
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise NormalizationError("INVALID_CORNERS", "Document corners are invalid")
        height, width = image.shape[:2]
        if (
            (corners[:, 0] < 0).any()
            or (corners[:, 1] < 0).any()
            or (corners[:, 0] > width - 1).any()
            or (corners[:, 1] > height - 1).any()
        ):
            raise NormalizationError("CORNERS_OUT_OF_BOUNDS", "Document corners lie outside the image")
        area = abs(cv2.contourArea(corners))
        if area < width * height * 0.01:
            raise NormalizationError("DOCUMENT_TOO_SMALL", "Detected document area is too small")
        if not cv2.isContourConvex(corners.astype(np.float32)):
            raise NormalizationError("INVALID_CORNERS", "Document corners do not form a convex polygon")


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)
