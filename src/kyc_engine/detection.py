from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Protocol

import cv2
import numpy as np

from .contracts import DetectionResult, Image, Point, Quadrilateral


class DetectionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DocumentDetector(Protocol):
    def detect(self, image: Image) -> DetectionResult:
        ...


@dataclass(frozen=True)
class ClassicalDetectionConfig:
    expected_aspect_ratio: float = 1.537
    min_area_ratio: float = 0.08
    fallback_min_area_ratio: float = 0.03
    fallback_min_rectangularity: float = 0.50
    min_confidence: float = 0.48
    max_processing_side: int = 1400
    light_mask_max_saturation: int = 25
    light_mask_min_value: int = 100
    light_mask_kernel_size: int = 15
    light_mask_border_expansion_ratio: float = 0.09
    ambiguity_confidence_margin: float = 0.15


@dataclass(frozen=True)
class _Candidate:
    corners: Quadrilateral
    confidence: float
    components: dict[str, float]
    bounding_rect: tuple[int, int, int, int]


class OpenCVDocumentDetector:
    name = "opencv_quadrilateral"
    version = "1"

    def __init__(
        self,
        config: ClassicalDetectionConfig | None = None,
        *,
        side: str = "front",
    ) -> None:
        if side not in {"front", "back"}:
            raise ValueError("Document side must be 'front' or 'back'")
        if not 0 <= (config or ClassicalDetectionConfig()).light_mask_border_expansion_ratio < 0.25:
            raise ValueError("Light-mask border expansion ratio must be between 0 and 0.25")
        self.config = config or ClassicalDetectionConfig()
        self.side = side

    def detect(self, image: Image) -> DetectionResult:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise TypeError("Detector expects a BGR uint8 image")

        resized, scale = self._resize_for_detection(image)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = self._build_candidates(contours, resized.shape[:2], scale)
        candidates.extend(
            self._build_rotated_rect_candidates(
                self._fallback_contours(gray),
                resized.shape[:2],
                scale,
            )
        )
        candidates.extend(
            self._build_rotated_rect_candidates(
                self._light_mask_contours(resized),
                resized.shape[:2],
                scale,
                border_expansion_ratio=self.config.light_mask_border_expansion_ratio,
            )
        )
        candidates = self._suppress_duplicates(
            sorted(candidates, key=lambda item: item.confidence, reverse=True)
        )
        plausible = [item for item in candidates if item.confidence >= self.config.min_confidence]
        if not plausible:
            raise DetectionError("DOCUMENT_NOT_DETECTED", "No supported document was detected")
        if (
            len(plausible) > 1
            and plausible[0].confidence - plausible[1].confidence
            < self.config.ambiguity_confidence_margin
        ):
            raise DetectionError(
                "MULTIPLE_DOCUMENTS",
                "Multiple plausible documents were detected",
            )

        selected = plausible[0]
        return DetectionResult(
            document_type="ao_id_card",
            side=self.side,
            confidence=selected.confidence,
            corners=selected.corners,
            detector=self.name,
            detector_version=self.version,
            confidence_components=selected.components,
        )

    def _resize_for_detection(self, image: Image) -> tuple[Image, float]:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.config.max_processing_side:
            return image, 1.0
        scale = self.config.max_processing_side / longest
        resized = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _fallback_contours(gray: np.ndarray) -> list[np.ndarray]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 25, 90)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def _light_mask_contours(self, image: Image) -> list[np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.config.light_mask_min_value], dtype=np.uint8),
            np.array([180, self.config.light_mask_max_saturation, 255], dtype=np.uint8),
        )
        kernel_size = self.config.light_mask_kernel_size
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        )
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def _build_candidates(
        self,
        contours: tuple[np.ndarray, ...] | list[np.ndarray],
        shape: tuple[int, int],
        scale: float,
    ) -> list[_Candidate]:
        height, width = shape
        image_area = float(width * height)
        candidates: list[_Candidate] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            area = abs(cv2.contourArea(polygon))
            area_ratio = area / image_area
            if area_ratio < self.config.min_area_ratio:
                continue

            raw_points = polygon.reshape(4, 2).astype(np.float32)
            ordered = order_corners(raw_points)
            side_lengths = _side_lengths(ordered)
            long_side = max(side_lengths)
            short_side = min(side_lengths)
            if short_side <= 1:
                continue
            aspect = long_side / short_side
            rect = cv2.minAreaRect(polygon)
            rect_area = max(rect[1][0] * rect[1][1], 1.0)
            rectangularity = min(1.0, area / rect_area)
            aspect_score = exp(-abs(log(aspect / self.config.expected_aspect_ratio)) / 0.35)
            angle_score = _right_angle_score(ordered)
            area_score = min(1.0, area_ratio / 0.35)
            confidence = float(
                max(0.0, min(1.0, 0.35 * aspect_score + 0.30 * rectangularity + 0.25 * angle_score + 0.10 * area_score))
            )
            restored = ordered / scale
            corners = Quadrilateral(
                tuple(Point(float(x), float(y)) for x, y in restored)  # type: ignore[arg-type]
            )
            x, y, rect_width, rect_height = cv2.boundingRect(polygon)
            candidates.append(
                _Candidate(
                    corners=corners,
                    confidence=confidence,
                    components={
                        "aspect": float(aspect_score),
                        "rectangularity": float(rectangularity),
                        "right_angles": float(angle_score),
                        "area": float(area_score),
                    },
                    bounding_rect=(x, y, rect_width, rect_height),
                )
            )
        return sorted(candidates, key=lambda item: item.confidence, reverse=True)

    def _build_rotated_rect_candidates(
        self,
        contours: tuple[np.ndarray, ...] | list[np.ndarray],
        shape: tuple[int, int],
        scale: float,
        *,
        border_expansion_ratio: float = 0.0,
    ) -> list[_Candidate]:
        height, width = shape
        image_area = float(width * height)
        candidates: list[_Candidate] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
            area = abs(cv2.contourArea(contour))
            area_ratio = area / image_area
            if area_ratio < self.config.fallback_min_area_ratio:
                continue

            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = rect[1]
            rect_area = max(rect_width * rect_height, 1.0)
            rectangularity = min(1.0, area / rect_area)
            if rectangularity < self.config.fallback_min_rectangularity:
                continue

            short_side = max(min(rect_width, rect_height), 1.0)
            aspect = max(rect_width, rect_height) / short_side
            aspect_score = exp(-abs(log(aspect / self.config.expected_aspect_ratio)) / 0.35)
            area_score = min(1.0, area_ratio / 0.20)
            confidence = float(
                max(0.0, min(1.0, 0.45 * aspect_score + 0.40 * rectangularity + 0.15 * area_score))
            )
            raw_points = cv2.boxPoints(rect).astype(np.float32)
            ordered = _expand_quadrilateral(
                order_corners(raw_points),
                border_expansion_ratio,
                width,
                height,
            )
            restored = ordered / scale
            corners = Quadrilateral(
                tuple(Point(float(x), float(y)) for x, y in restored)  # type: ignore[arg-type]
            )
            x, y, rect_width_int, rect_height_int = cv2.boundingRect(raw_points.astype(np.int32))
            candidates.append(
                _Candidate(
                    corners=corners,
                    confidence=confidence,
                    components={
                        "aspect": float(aspect_score),
                        "rectangularity": float(rectangularity),
                        "right_angles": 1.0,
                        "area": float(area_score),
                    },
                    bounding_rect=(x, y, rect_width_int, rect_height_int),
                )
            )
        return sorted(candidates, key=lambda item: item.confidence, reverse=True)

    @staticmethod
    def _suppress_duplicates(candidates: list[_Candidate]) -> list[_Candidate]:
        selected: list[_Candidate] = []
        for candidate in candidates:
            if any(_rect_iou(candidate.bounding_rect, kept.bounding_rect) >= 0.80 for kept in selected):
                continue
            selected.append(candidate)
        return selected


def order_corners(points: np.ndarray) -> NDArray[np.float32]:
    if points.shape != (4, 2):
        raise ValueError("Exactly four 2D points are required")
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    if len({(float(x), float(y)) for x, y in ordered}) != 4:
        raise ValueError("Document corners must be distinct")
    return ordered


def _side_lengths(points: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(
        float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
        for index in range(4)
    )  # type: ignore[return-value]


def _expand_quadrilateral(
    points: np.ndarray,
    ratio: float,
    width: int,
    height: int,
) -> np.ndarray:
    if ratio == 0:
        return points
    center = points.mean(axis=0)
    expanded = center + (points - center) * (1 + ratio)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded.astype(np.float32)


def _right_angle_score(points: np.ndarray) -> float:
    scores = []
    for index in range(4):
        previous = points[(index - 1) % 4] - points[index]
        following = points[(index + 1) % 4] - points[index]
        denominator = np.linalg.norm(previous) * np.linalg.norm(following)
        if denominator <= 1e-6:
            return 0.0
        cosine = abs(float(np.dot(previous, following) / denominator))
        scores.append(max(0.0, 1.0 - cosine))
    return float(sum(scores) / len(scores))


def _rect_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0
