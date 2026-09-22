from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from types import MappingProxyType
from typing import Mapping, Protocol

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
    front_edge_refinement_band_ratio: float = 0.045
    front_edge_refinement_min_support: float = 0.45
    front_edge_refinement_max_corner_shift_ratio: float = 0.08
    back_edge_refinement_search_expansion_ratio: float = 0.14
    back_edge_refinement_band_ratio: float = 0.15
    back_edge_refinement_min_support: float = 0.50
    back_edge_refinement_max_corner_shift_ratio: float = 0.22


@dataclass(frozen=True)
class _Candidate:
    corners: Quadrilateral
    tight_corners: Quadrilateral
    confidence: float
    components: dict[str, float]
    bounding_rect: tuple[int, int, int, int]
    kind: str


@dataclass(frozen=True)
class DetectionDebug:
    """Geometry diagnostics intended for local private debug tooling only."""

    coarse_corners: Quadrilateral
    final_corners: Quadrilateral
    candidate_kind: str
    refinement_accepted: bool
    edge_support: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_support", MappingProxyType(dict(self.edge_support)))


class OpenCVDocumentDetector:
    name = "opencv_quadrilateral"
    version = "3"

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
        if not 0 <= (config or ClassicalDetectionConfig()).back_edge_refinement_search_expansion_ratio < 0.25:
            raise ValueError("Back edge-refinement search expansion ratio must be between 0 and 0.25")
        self.config = config or ClassicalDetectionConfig()
        self.side = side

    def detect(self, image: Image) -> DetectionResult:
        result, _ = self._detect(image)
        return result

    def detect_with_debug(self, image: Image) -> tuple[DetectionResult, DetectionDebug]:
        """Detect a document and return local geometry diagnostics for debug scripts."""
        return self._detect(image)

    def _detect(self, image: Image) -> tuple[DetectionResult, DetectionDebug]:
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
        final_corners = selected.tight_corners
        coarse_corners = selected.tight_corners
        refinement_accepted = False
        edge_support: dict[str, float] = {}
        if self.side == "front":
            refined, edge_support = self._refine_front_edges(image, selected.tight_corners)
            if refined is not None:
                final_corners = refined
                refinement_accepted = True
            prefix = "front"
        else:
            # The back's translucent holder can hide the light-mask boundary. Search
            # outward for the physical card edge, but preserve the historical expanded
            # proposal if that evidence is incomplete.
            coarse_corners = self._expand_back_search_geometry(image, selected.tight_corners)
            final_corners = selected.corners
            refined, edge_support = self._refine_back_edges(image, coarse_corners)
            if refined is not None:
                final_corners = refined
                refinement_accepted = True
            prefix = "back"

        components = dict(selected.components)
        components[f"{prefix}_refinement_accepted"] = float(refinement_accepted)
        components.update(
            {f"{prefix}_edge_support_{side}": support for side, support in edge_support.items()}
        )
        result = DetectionResult(
            document_type="ao_id_card",
            side=self.side,
            confidence=selected.confidence,
            corners=final_corners,
            detector=self.name,
            detector_version=self.version,
            confidence_components=components,
        )
        return result, DetectionDebug(
            coarse_corners=coarse_corners,
            final_corners=final_corners,
            candidate_kind=selected.kind,
            refinement_accepted=refinement_accepted,
            edge_support=edge_support,
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
                    tight_corners=corners,
                    confidence=confidence,
                    components={
                        "aspect": float(aspect_score),
                        "rectangularity": float(rectangularity),
                        "right_angles": float(angle_score),
                        "area": float(area_score),
                    },
                    bounding_rect=(x, y, rect_width, rect_height),
                    kind="contour_quadrilateral",
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
            tight_ordered = order_corners(raw_points)
            ordered = _expand_quadrilateral(
                tight_ordered,
                border_expansion_ratio,
                width,
                height,
            )
            restored = ordered / scale
            corners = Quadrilateral(
                tuple(Point(float(x), float(y)) for x, y in restored)  # type: ignore[arg-type]
            )
            tight_restored = tight_ordered / scale
            tight_corners = Quadrilateral(
                tuple(Point(float(x), float(y)) for x, y in tight_restored)  # type: ignore[arg-type]
            )
            x, y, rect_width_int, rect_height_int = cv2.boundingRect(raw_points.astype(np.int32))
            candidates.append(
                _Candidate(
                    corners=corners,
                    tight_corners=tight_corners,
                    confidence=confidence,
                    components={
                        "aspect": float(aspect_score),
                        "rectangularity": float(rectangularity),
                        "right_angles": 1.0,
                        "area": float(area_score),
                    },
                    bounding_rect=(x, y, rect_width_int, rect_height_int),
                    kind="rotated_rectangle",
                )
            )
        return sorted(candidates, key=lambda item: item.confidence, reverse=True)

    def _refine_front_edges(
        self,
        image: Image,
        coarse: Quadrilateral,
    ) -> tuple[Quadrilateral | None, dict[str, float]]:
        return self._refine_edges(
            image,
            coarse,
            band_ratio=self.config.front_edge_refinement_band_ratio,
            minimum_support=self.config.front_edge_refinement_min_support,
            max_corner_shift_ratio=self.config.front_edge_refinement_max_corner_shift_ratio,
        )

    def _refine_back_edges(
        self,
        image: Image,
        coarse: Quadrilateral,
    ) -> tuple[Quadrilateral | None, dict[str, float]]:
        return self._refine_edges(
            image,
            coarse,
            band_ratio=self.config.back_edge_refinement_band_ratio,
            minimum_support=self.config.back_edge_refinement_min_support,
            max_corner_shift_ratio=self.config.back_edge_refinement_max_corner_shift_ratio,
        )

    def _refine_edges(
        self,
        image: Image,
        coarse: Quadrilateral,
        *,
        band_ratio: float,
        minimum_support: float,
        max_corner_shift_ratio: float,
    ) -> tuple[Quadrilateral | None, dict[str, float]]:
        """Fit visible outer card edges near a coarse quadrilateral.

        A per-side signed-distance histogram lets the long physical border win over
        document texture, which produces short, scattered edges in the same band.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 140)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        points = coarse.as_array().astype(np.float64)
        lengths = _side_lengths(points.astype(np.float32))
        band = max(8.0, min(lengths) * band_ratio)
        fitted_lines: list[tuple[np.ndarray, np.ndarray]] = []
        supports: dict[str, float] = {}
        for index, name in enumerate(("top", "right", "bottom", "left")):
            start = points[index]
            end = points[(index + 1) % 4]
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length <= 1.0:
                return None, supports
            direction /= length
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
            support_points, coverage = _edge_support_points(
                edges,
                grad_x,
                grad_y,
                start,
                direction,
                normal,
                length,
                band,
            )
            supports[name] = coverage
            if coverage < minimum_support:
                return None, supports
            fitted_lines.append(_fit_line(support_points))

        refined = np.asarray(
            [
                _line_intersection(fitted_lines[index - 1], fitted_lines[index])
                for index in range(4)
            ],
            dtype=np.float64,
        )
        if not self._valid_refinement(
            refined,
            points,
            image.shape[:2],
            band,
            max_corner_shift_ratio=max_corner_shift_ratio,
        ):
            return None, supports
        return Quadrilateral(
            tuple(Point(float(x), float(y)) for x, y in refined)  # type: ignore[arg-type]
        ), supports

    def _valid_refinement(
        self,
        refined: np.ndarray,
        coarse: np.ndarray,
        shape: tuple[int, int],
        band: float,
        *,
        max_corner_shift_ratio: float,
    ) -> bool:
        height, width = shape
        if refined.shape != (4, 2) or not np.isfinite(refined).all():
            return False
        if len({(float(x), float(y)) for x, y in refined}) != 4:
            return False
        if (refined[:, 0] < 0).any() or (refined[:, 1] < 0).any():
            return False
        if (refined[:, 0] >= width).any() or (refined[:, 1] >= height).any():
            return False
        if not cv2.isContourConvex(refined.astype(np.float32)):
            return False
        area_ratio = abs(cv2.contourArea(refined.astype(np.float32))) / float(width * height)
        if area_ratio < self.config.fallback_min_area_ratio:
            return False
        rect = cv2.minAreaRect(refined.astype(np.float32))
        rectangularity = abs(cv2.contourArea(refined.astype(np.float32))) / max(
            rect[1][0] * rect[1][1],
            1.0,
        )
        if rectangularity < self.config.fallback_min_rectangularity:
            return False
        side_lengths = _side_lengths(refined.astype(np.float32))
        aspect = max(side_lengths) / max(min(side_lengths), 1.0)
        if abs(log(aspect / self.config.expected_aspect_ratio)) > 0.35:
            return False
        if _right_angle_score(refined.astype(np.float32)) < 0.70:
            return False
        max_shift = float(np.max(np.linalg.norm(refined - coarse, axis=1)))
        diagonal = float(np.linalg.norm(np.asarray((width, height), dtype=np.float64)))
        return max_shift <= max(band * 2.0, diagonal * max_corner_shift_ratio)

    def _expand_back_search_geometry(
        self,
        image: Image,
        tight_corners: Quadrilateral,
    ) -> Quadrilateral:
        expanded = _expand_quadrilateral(
            tight_corners.as_array(),
            self.config.back_edge_refinement_search_expansion_ratio,
            image.shape[1],
            image.shape[0],
        )
        return Quadrilateral(
            tuple(Point(float(x), float(y)) for x, y in expanded)  # type: ignore[arg-type]
        )

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


def _edge_support_points(
    edges: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    length: float,
    band: float,
) -> tuple[np.ndarray, float]:
    """Return a narrow, consistently supported edge closest to one expected side."""
    height, width = edges.shape
    margin = int(np.ceil(band + 2))
    endpoints = np.asarray((start, start + direction * length))
    x0 = max(0, int(np.floor(endpoints[:, 0].min())) - margin)
    x1 = min(width, int(np.ceil(endpoints[:, 0].max())) + margin + 1)
    y0 = max(0, int(np.floor(endpoints[:, 1].min())) - margin)
    y1 = min(height, int(np.ceil(endpoints[:, 1].max())) + margin + 1)
    ys, xs = np.nonzero(edges[y0:y1, x0:x1])
    if not len(xs):
        return np.empty((0, 2), dtype=np.float64), 0.0

    candidates = np.column_stack((xs + x0, ys + y0)).astype(np.float64)
    relative = candidates - start
    along = relative @ direction
    distance = relative @ normal
    in_band = (along >= 0) & (along <= length) & (np.abs(distance) <= band)
    candidates = candidates[in_band]
    along = along[in_band]
    distance = distance[in_band]
    if not len(candidates):
        return np.empty((0, 2), dtype=np.float64), 0.0

    gradients = np.column_stack((grad_x[candidates[:, 1].astype(int), candidates[:, 0].astype(int)], grad_y[candidates[:, 1].astype(int), candidates[:, 0].astype(int)]))
    magnitudes = np.linalg.norm(gradients, axis=1)
    aligned = np.abs((gradients @ normal) / np.maximum(magnitudes, 1e-6)) >= 0.70
    candidates = candidates[aligned]
    along = along[aligned]
    distance = distance[aligned]
    if not len(candidates):
        return np.empty((0, 2), dtype=np.float64), 0.0

    # A physical border has support across the side. Texture may be strong, but
    # its signed distances are distributed rather than concentrated in one bin.
    bin_width = 2.0
    bin_indices = np.floor((distance + band) / bin_width).astype(int)
    bins = max(1, int(np.ceil((2 * band) / bin_width)))
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    counts = np.bincount(bin_indices, minlength=bins)
    winning = int(np.argmax(counts))
    center = -band + (winning + 0.5) * bin_width
    selected = np.abs(distance - center) <= 2.0
    selected_points = candidates[selected]
    selected_along = along[selected]
    if len(selected_points) < 8:
        return selected_points, 0.0
    segment_count = 12
    segments = np.minimum(segment_count - 1, (selected_along / length * segment_count).astype(int))
    coverage = float(len(np.unique(segments)) / segment_count)
    return selected_points, coverage


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise ValueError("At least two points are required to fit a document edge")
    vector = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    direction = np.asarray((vector[0], vector[1]), dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    origin = np.asarray((vector[2], vector[3]), dtype=np.float64)
    return origin, direction


def _line_intersection(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    first_origin, first_direction = first
    second_origin, second_direction = second
    matrix = np.column_stack((first_direction, -second_direction))
    try:
        first_scale, _ = np.linalg.solve(matrix, second_origin - first_origin)
    except np.linalg.LinAlgError:
        return np.asarray((np.nan, np.nan), dtype=np.float64)
    return first_origin + first_scale * first_direction


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
