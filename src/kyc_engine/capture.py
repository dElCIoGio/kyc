from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np
from numpy.typing import NDArray

from .contracts import (
    CaptureAssessment,
    CaptureIssue,
    CaptureIssueCode,
    CaptureMetrics,
)
from .intake import ImageIntake, IntakeError
from .quality import BrightnessAssessor, ContrastAssessor, SharpnessAssessor


class CaptureAssessmentInputError(ValueError):
    """A safe, machine-readable failure while validating a capture input."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class CaptureAssessmentConfig:
    """Initial uncalibrated defaults for the deterministic capture gate."""

    min_sharpness: float = 25.0
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    min_contrast: float = 12.0
    min_document_area_ratio: float = 0.12
    min_edge_margin_ratio: float = 0.015
    max_perspective_distortion: float = 0.30
    max_glare_ratio: float = 0.035

    def __post_init__(self) -> None:
        values = (
            self.min_sharpness,
            self.min_brightness,
            self.max_brightness,
            self.min_contrast,
            self.min_document_area_ratio,
            self.min_edge_margin_ratio,
            self.max_perspective_distortion,
            self.max_glare_ratio,
        )
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            raise TypeError("Capture assessment thresholds must be numeric")
        if any(not isfinite(float(value)) for value in values):
            raise ValueError("Capture assessment thresholds must be finite")
        if self.min_sharpness < 0 or self.min_contrast < 0:
            raise ValueError("Capture sharpness and contrast thresholds must be non-negative")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ValueError("Capture brightness thresholds must satisfy 0 <= min < max <= 255")
        if not 0 < self.min_document_area_ratio < 1:
            raise ValueError("Capture minimum document area ratio must be between 0 and 1")
        if not 0 <= self.min_edge_margin_ratio < 0.5:
            raise ValueError("Capture edge margin ratio must be between 0 and 0.5")
        if not 0 < self.max_perspective_distortion <= 1:
            raise ValueError("Capture maximum perspective distortion must be between 0 and 1")
        if not 0 < self.max_glare_ratio < 1:
            raise ValueError("Capture maximum glare ratio must be between 0 and 1")


@dataclass(frozen=True)
class _DocumentGeometry:
    """Private, non-serializable capture geometry used only by the quality gate."""

    corners: NDArray[np.float32]
    contour: NDArray[np.int32]
    area_ratio: float
    minimum_margin_ratio: float
    perspective_score: float
    rectangularity: float
    clipping_evidence: bool
    fallback: bool = False


class _CaptureQuadrilateralDetector:
    """Fast generic rectangle finder; it has no document type or side semantics."""

    _MAX_PROCESSING_SIDE = 1024
    _MIN_CANDIDATE_AREA_RATIO = 0.03
    _MIN_RECTANGULARITY = 0.72
    _MIN_ASPECT_RATIO = 1.15
    _MAX_ASPECT_RATIO = 2.80

    def detect(self, image: np.ndarray, *, safety_margin_ratio: float) -> _DocumentGeometry | None:
        resized, scale = self._resize(image)
        height, width = resized.shape[:2]
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[_DocumentGeometry] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
            candidate = self._from_contour(contour, (height, width), safety_margin_ratio)
            if candidate is not None:
                candidates.append(candidate)
                continue
            fallback = self._from_rotated_rectangle(contour, (height, width), safety_margin_ratio)
            if fallback is not None:
                candidates.append(fallback)
        if not candidates:
            return None

        # Favor well-supported, rectangular boundaries rather than merely large ones.
        direct_candidates = [candidate for candidate in candidates if not candidate.fallback]
        selected = max(
            direct_candidates or candidates,
            key=lambda item: (item.rectangularity * 0.55 + min(1.0, item.area_ratio / 0.35) * 0.30 + (1.0 - item.perspective_score) * 0.15),
        )
        if scale == 1.0:
            return selected
        restored = selected.corners / scale
        restored_contour = np.rint(selected.contour.astype(np.float32) / scale).astype(np.int32)
        original_height, original_width = image.shape[:2]
        return self._geometry(
            restored,
            restored_contour,
            (original_height, original_width),
            safety_margin_ratio,
            selected.rectangularity,
            fallback=selected.fallback,
        )

    def _resize(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self._MAX_PROCESSING_SIDE:
            return image, 1.0
        scale = self._MAX_PROCESSING_SIDE / float(longest)
        return (
            cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA),
            scale,
        )

    def _from_contour(
        self,
        contour: NDArray[np.int32],
        shape: tuple[int, int],
        safety_margin_ratio: float,
    ) -> _DocumentGeometry | None:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1:
            return None
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            return None
        return self._candidate(
            polygon.reshape(4, 2).astype(np.float32),
            contour,
            shape,
            safety_margin_ratio,
        )

    def _from_rotated_rectangle(
        self,
        contour: NDArray[np.int32],
        shape: tuple[int, int],
        safety_margin_ratio: float,
    ) -> _DocumentGeometry | None:
        if cv2.contourArea(contour) <= 0:
            return None
        rect = cv2.minAreaRect(contour)
        if min(rect[1]) <= 1:
            return None
        corners = cv2.boxPoints(rect).astype(np.float32)
        candidate = self._candidate(corners, contour, shape, safety_margin_ratio)
        if candidate is None or not self._has_four_sided_support(contour, candidate.corners):
            return None
        return _DocumentGeometry(
            corners=candidate.corners,
            contour=candidate.contour,
            area_ratio=candidate.area_ratio,
            minimum_margin_ratio=candidate.minimum_margin_ratio,
            perspective_score=candidate.perspective_score,
            rectangularity=candidate.rectangularity,
            clipping_evidence=candidate.clipping_evidence,
            fallback=True,
        )

    def _candidate(
        self,
        raw_corners: NDArray[np.float32],
        contour: NDArray[np.int32],
        shape: tuple[int, int],
        safety_margin_ratio: float,
    ) -> _DocumentGeometry | None:
        try:
            corners = _order_corners(raw_corners)
        except ValueError:
            return None
        if not _valid_quadrilateral(corners, shape):
            return None
        area = abs(float(cv2.contourArea(corners)))
        image_area = float(shape[0] * shape[1])
        if area / image_area < self._MIN_CANDIDATE_AREA_RATIO:
            return None
        rect = cv2.minAreaRect(corners)
        rect_area = max(float(rect[1][0] * rect[1][1]), 1.0)
        rectangularity = min(1.0, area / rect_area)
        if rectangularity < self._MIN_RECTANGULARITY:
            return None
        lengths = _side_lengths(corners)
        aspect = max(lengths) / max(min(lengths), 1.0)
        if not self._MIN_ASPECT_RATIO <= aspect <= self._MAX_ASPECT_RATIO:
            return None
        return self._geometry(corners, contour, shape, safety_margin_ratio, rectangularity)

    @staticmethod
    def _has_four_sided_support(contour: NDArray[np.int32], corners: NDArray[np.float32]) -> bool:
        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 16:
            return False
        side_lengths = _side_lengths(corners)
        for index in range(4):
            start, end = corners[index], corners[(index + 1) % 4]
            direction = end - start
            length = max(float(np.linalg.norm(direction)), 1.0)
            direction /= length
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float32)
            relative = points - start
            along = relative @ direction
            distance = np.abs(relative @ normal)
            near_side = (along >= 0) & (along <= length) & (distance <= max(3.0, side_lengths[index] * 0.025))
            if int(np.count_nonzero(near_side)) < 4:
                return False
        return True

    @staticmethod
    def _geometry(
        corners: NDArray[np.float32],
        contour: NDArray[np.int32],
        shape: tuple[int, int],
        safety_margin_ratio: float,
        rectangularity: float,
        *,
        fallback: bool = False,
    ) -> _DocumentGeometry:
        height, width = shape
        area_ratio = abs(float(cv2.contourArea(corners))) / float(width * height)
        margins = np.column_stack((corners[:, 0], corners[:, 1], width - 1 - corners[:, 0], height - 1 - corners[:, 1]))
        minimum_margin_ratio = max(0.0, float(np.min(margins)) / float(min(width, height)))
        clipping_evidence = _has_clipping_evidence(
            corners,
            contour,
            shape,
            safety_margin_ratio,
        )
        return _DocumentGeometry(
            corners=corners,
            contour=contour,
            area_ratio=area_ratio,
            minimum_margin_ratio=minimum_margin_ratio,
            perspective_score=_perspective_score(corners),
            rectangularity=rectangularity,
            clipping_evidence=clipping_evidence,
            fallback=fallback,
        )


class DocumentCaptureAssessor:
    """Fast deterministic gate deciding whether an image merits OCR processing."""

    def __init__(
        self,
        intake: ImageIntake,
        config: CaptureAssessmentConfig | None = None,
    ) -> None:
        self._intake = intake
        self._config = config or CaptureAssessmentConfig()
        self._sharpness = SharpnessAssessor()
        self._brightness = BrightnessAssessor()
        self._contrast = ContrastAssessor()
        self._geometry = _CaptureQuadrilateralDetector()

    def assess(self, source: bytes) -> CaptureAssessment:
        try:
            input_image = self._intake.load(source)
        except IntakeError as exc:
            raise CaptureAssessmentInputError(exc.code) from exc

        image = input_image.image
        sharpness = float(self._sharpness.assess(image).metrics["laplacian_variance"])
        brightness = float(self._brightness.assess(image).metrics["mean_intensity"])
        contrast = float(self._contrast.assess(image).metrics["intensity_stddev"])
        geometry = self._geometry.detect(
            image,
            safety_margin_ratio=self._config.min_edge_margin_ratio,
        )
        issues: list[CaptureIssue] = []
        glare_ratio: float | None = None
        if geometry is None:
            issues.append(CaptureIssue(CaptureIssueCode.DOCUMENT_NOT_DETECTED, "Make sure the whole document is visible."))
        else:
            if geometry.clipping_evidence:
                issues.append(CaptureIssue(CaptureIssueCode.DOCUMENT_CROPPED, "Make sure all four corners of the document are visible."))
            if geometry.area_ratio < self._config.min_document_area_ratio:
                issues.append(CaptureIssue(CaptureIssueCode.DOCUMENT_TOO_SMALL, "Move closer so the document fills more of the frame."))
            if geometry.perspective_score > self._config.max_perspective_distortion:
                issues.append(CaptureIssue(CaptureIssueCode.EXCESSIVE_PERSPECTIVE, "Hold the camera more directly above the document."))
        if sharpness < self._config.min_sharpness:
            issues.append(CaptureIssue(CaptureIssueCode.TOO_BLURRY, "The document image is too blurry."))
        if brightness < self._config.min_brightness:
            issues.append(CaptureIssue(CaptureIssueCode.TOO_DARK, "The document image is too dark."))
        if brightness > self._config.max_brightness:
            issues.append(CaptureIssue(CaptureIssueCode.OVEREXPOSED, "The document image is overexposed."))
        if contrast < self._config.min_contrast:
            issues.append(CaptureIssue(CaptureIssueCode.LOW_CONTRAST, "The document image has too little contrast."))
        if geometry is not None:
            glare_ratio = _glare_ratio(image, geometry.corners)
            if glare_ratio > self._config.max_glare_ratio:
                issues.append(CaptureIssue(CaptureIssueCode.GLARE_DETECTED, "Reduce reflections or direct light on the document."))
        return CaptureAssessment(
            accepted=not issues,
            issues=tuple(issues),
            metrics=CaptureMetrics(
                width=input_image.width,
                height=input_image.height,
                sharpness=sharpness,
                brightness=brightness,
                contrast=contrast,
                document_detected=geometry is not None,
                document_area_ratio=geometry.area_ratio if geometry is not None else None,
                minimum_margin_ratio=geometry.minimum_margin_ratio if geometry is not None else None,
                perspective_score=geometry.perspective_score if geometry is not None else None,
                glare_ratio=glare_ratio,
            ),
        )


def _order_corners(points: NDArray[np.float32]) -> NDArray[np.float32]:
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Four finite points are required")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)].astype(np.float32)
    top_left = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -top_left, axis=0)
    # Clockwise ordering is TL, TR, BR, BL in image coordinates.
    first, second = ordered[1] - ordered[0], ordered[2] - ordered[1]
    if float(first[0] * second[1] - first[1] * second[0]) < 0:
        ordered = np.asarray((ordered[0], ordered[3], ordered[2], ordered[1]), dtype=np.float32)
    if len({(float(x), float(y)) for x, y in ordered}) != 4:
        raise ValueError("Corners must be distinct")
    return ordered


def _valid_quadrilateral(corners: NDArray[np.float32], shape: tuple[int, int]) -> bool:
    height, width = shape
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return False
    tolerance = 2.0
    if (corners[:, 0] < -tolerance).any() or (corners[:, 0] > width - 1 + tolerance).any():
        return False
    if (corners[:, 1] < -tolerance).any() or (corners[:, 1] > height - 1 + tolerance).any():
        return False
    if not cv2.isContourConvex(corners):
        return False
    return abs(float(cv2.contourArea(corners))) > 16.0


def _side_lengths(corners: NDArray[np.float32]) -> tuple[float, float, float, float]:
    return tuple(
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    )  # type: ignore[return-value]


def _perspective_score(corners: NDArray[np.float32]) -> float:
    top, right, bottom, left = _side_lengths(corners)
    horizontal_imbalance = abs(top - bottom) / max(top, bottom, 1.0)
    vertical_imbalance = abs(left - right) / max(left, right, 1.0)
    angle_deviation = 0.0
    for index in range(4):
        previous = corners[(index - 1) % 4] - corners[index]
        following = corners[(index + 1) % 4] - corners[index]
        denominator = max(float(np.linalg.norm(previous) * np.linalg.norm(following)), 1e-6)
        angle_deviation += abs(float(np.dot(previous, following))) / denominator
    angle_deviation /= 4.0
    maximum_imbalance = max(horizontal_imbalance, vertical_imbalance)
    mean_imbalance = (horizontal_imbalance + vertical_imbalance) / 2.0
    return float(min(1.0, 0.50 * maximum_imbalance + 0.30 * mean_imbalance + 0.20 * angle_deviation))


def _has_clipping_evidence(
    corners: NDArray[np.float32],
    contour: NDArray[np.int32],
    shape: tuple[int, int],
    safety_margin_ratio: float,
) -> bool:
    """Require actual frame contact, not merely a small visible margin."""
    height, width = shape
    points = contour.reshape(-1, 2)
    frame_epsilon = 2
    touches_frame = (
        (points[:, 0] <= frame_epsilon)
        | (points[:, 1] <= frame_epsilon)
        | (points[:, 0] >= width - 1 - frame_epsilon)
        | (points[:, 1] >= height - 1 - frame_epsilon)
    )
    if not bool(np.any(touches_frame)):
        return False
    safety_pixels = max(2.0, min(width, height) * safety_margin_ratio)
    near_safety_boundary = (
        (corners[:, 0] <= safety_pixels)
        | (corners[:, 1] <= safety_pixels)
        | (corners[:, 0] >= width - 1 - safety_pixels)
        | (corners[:, 1] >= height - 1 - safety_pixels)
    )
    return bool(np.any(near_safety_boundary))


def _glare_ratio(image: np.ndarray, corners: NDArray[np.float32]) -> float:
    """Initial uncalibrated highlight-region heuristic, evaluated only inside the document."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
    document_pixels = max(int(np.count_nonzero(mask)), 1)
    # Do not mistake a bright document border for a specular reflection.
    interior_mask = cv2.erode(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(3, (min(image.shape[:2]) // 30) * 2 + 1),) * 2,
        ),
    )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    local_background = cv2.GaussianBlur(value, (31, 31), 0)
    highlight_edges = (
        (interior_mask > 0)
        & (hsv[:, :, 1] <= 50)
        & (value >= 245)
        & ((value.astype(np.int16) - local_background.astype(np.int16)) >= 18)
    ).astype(np.uint8)
    # A broad reflection is locally bright chiefly around its boundary. Fill only
    # those bright-edge contours so a uniformly white document surface is ignored.
    contours, _ = cv2.findContours(highlight_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlights = np.zeros_like(highlight_edges)
    cv2.drawContours(highlights, contours, -1, 1, thickness=cv2.FILLED)
    labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(highlights, connectivity=8)
    minimum_pixels = max(8, round(document_pixels * 0.002))
    largest = 0
    for label in range(1, labels):
        largest = max(largest, int(stats[label, cv2.CC_STAT_AREA]))
    return float(largest / document_pixels) if largest >= minimum_pixels else 0.0
