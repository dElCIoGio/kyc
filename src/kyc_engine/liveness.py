from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from math import isfinite
from typing import Iterable

import cv2
import numpy as np

from .contracts import (
    AntiSpoofDetector,
    AntiSpoofResult,
    FaceDetector,
    Image,
    LivenessResult,
    readonly_image,
)
from .quality import BrightnessAssessor, ContrastAssessor, SharpnessAssessor


class LivenessFrameError(ValueError):
    """A supplied frame cannot produce a passive liveness evaluation."""

    code = "LIVENESS_INVALID_FRAME"


class LivenessNoFaceError(LivenessFrameError):
    """A valid image did not contain a usable face region."""

    code = "LIVENESS_NO_FACE"


@dataclass(frozen=True)
class LivenessEvaluationConfig:
    """Initial deterministic policy for passive liveness and live-face selection.

    The face-quality thresholds are deliberately modest eligibility gates, not an
    identity-verification policy.  The weighted score is only used to choose
    between already eligible frames; it never changes the passive-liveness
    decision.
    """

    frame_count: int = 3
    minimum_real_ratio: float = 2.0 / 3.0
    min_face_width: int = 64
    min_face_height: int = 64
    min_face_area_ratio: float = 0.08
    min_sharpness: float = 25.0
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    min_contrast: float = 12.0

    def __post_init__(self) -> None:
        if not isinstance(self.frame_count, int) or isinstance(self.frame_count, bool):
            raise TypeError("Liveness frame_count must be an integer")
        if self.frame_count <= 0:
            raise ValueError("Liveness frame_count must be positive")
        if (
            not isinstance(self.minimum_real_ratio, (int, float))
            or isinstance(self.minimum_real_ratio, bool)
            or not isfinite(self.minimum_real_ratio)
            or not 0.0 < self.minimum_real_ratio <= 1.0
        ):
            raise ValueError("Liveness minimum_real_ratio must be finite and in (0, 1]")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.min_face_width, self.min_face_height)
        ):
            raise TypeError("Liveness minimum face dimensions must be integers")
        if self.min_face_width <= 0 or self.min_face_height <= 0:
            raise ValueError("Liveness minimum face dimensions must be positive")
        quality_values = (
            self.min_face_area_ratio,
            self.min_sharpness,
            self.min_brightness,
            self.max_brightness,
            self.min_contrast,
        )
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value)
            for value in quality_values
        ):
            raise TypeError("Liveness face-quality thresholds must be finite numbers")
        if not 0.0 < self.min_face_area_ratio <= 1.0:
            raise ValueError("Liveness minimum face area ratio must be in (0, 1]")
        if self.min_sharpness < 0 or self.min_contrast < 0:
            raise ValueError("Liveness sharpness and contrast thresholds must be non-negative")
        if not 0.0 <= self.min_brightness < self.max_brightness <= 255.0:
            raise ValueError("Liveness brightness thresholds must satisfy 0 <= min < max <= 255")
        object.__setattr__(self, "minimum_real_ratio", float(self.minimum_real_ratio))
        object.__setattr__(self, "min_face_area_ratio", float(self.min_face_area_ratio))
        object.__setattr__(self, "min_sharpness", float(self.min_sharpness))
        object.__setattr__(self, "min_brightness", float(self.min_brightness))
        object.__setattr__(self, "max_brightness", float(self.max_brightness))
        object.__setattr__(self, "min_contrast", float(self.min_contrast))


@dataclass(frozen=True)
class LiveFaceSelection:
    """Internal-only live-face selection metadata and an optional readonly crop.

    This object is intentionally not part of :class:`LivenessResult` and must
    never be serialized by an API boundary.
    """

    candidate_frame_count: int
    eligible_frame_count: int
    selected_frame_index: int | None
    face_detected: bool
    face_count: int
    quality_score: float | None
    selection_outcome: str
    face_crop: Image | None = None

    def __post_init__(self) -> None:
        if self.candidate_frame_count < 0 or self.eligible_frame_count < 0:
            raise ValueError("Live-face counts cannot be negative")
        if self.eligible_frame_count > self.candidate_frame_count:
            raise ValueError("Eligible live-face count cannot exceed candidate count")
        if self.selected_frame_index is not None and self.selected_frame_index < 0:
            raise ValueError("Selected live-face frame index cannot be negative")
        if self.face_count < 0:
            raise ValueError("Live-face count cannot be negative")
        if self.quality_score is not None and not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("Live-face quality score must be between 0 and 1")
        if (self.selected_frame_index is None) != (self.face_crop is None):
            raise ValueError("A selected live-face frame must have exactly one crop")
        if self.face_crop is not None:
            object.__setattr__(self, "face_crop", readonly_image(self.face_crop, copy=True))


@dataclass(frozen=True)
class LivenessEvaluation:
    """Internal result pairing the stable public decision with probe selection."""

    result: LivenessResult
    live_face: LiveFaceSelection


class LivenessEvaluator:
    """Aggregates passive liveness and optionally selects one future-match probe."""

    def __init__(
        self,
        detector: AntiSpoofDetector,
        config: LivenessEvaluationConfig | None = None,
        face_detector: FaceDetector | None = None,
    ) -> None:
        self._detector = detector
        self._config = config or LivenessEvaluationConfig()
        self._face_detector = face_detector

    @property
    def frame_count(self) -> int:
        """Configured frame count without exposing detector implementation details."""
        return self._config.frame_count

    def evaluate(self, frames: Iterable[Image]) -> LivenessResult:
        """Return the established passive aggregate without face-probe work."""
        return self._passive_evaluate(tuple(islice(frames, self._config.frame_count)))[0]

    def evaluate_with_live_face(self, frames: Iterable[Image]) -> LivenessEvaluation:
        """Evaluate passive liveness and choose one eligible face crop internally.

        Face-localization failures and unsuitable faces only prevent retention;
        they do not alter the established passive-liveness decision.
        """
        materialized = tuple(islice(frames, self._config.frame_count))
        result, evaluations = self._passive_evaluate(materialized)
        if self._face_detector is None:
            return LivenessEvaluation(result, _empty_selection("face_detector_unavailable"))

        candidate_count = 0
        eligible_count = 0
        face_detected = False
        max_face_count = 0
        selected: tuple[int, float, Image] | None = None
        for index, (frame, passive) in enumerate(zip(materialized, evaluations, strict=True)):
            if not passive.is_real:
                continue
            candidate_count += 1
            try:
                faces = self._face_detector.detect(frame).candidates
            except Exception:
                continue
            face_count = len(faces)
            face_detected = face_detected or face_count > 0
            max_face_count = max(max_face_count, face_count)
            if face_count != 1:
                continue
            candidate = _eligible_face_crop(frame, faces[0].bounding_box, self._config)
            if candidate is None:
                continue
            crop, quality_score = candidate
            eligible_count += 1
            # Deliberately exclude anti-spoof score. Ties retain earliest input order.
            if selected is None or quality_score > selected[1]:
                selected = (index, quality_score, crop)

        if selected is None:
            outcome = "no_real_frame" if candidate_count == 0 else "no_eligible_frame"
            return LivenessEvaluation(
                result,
                LiveFaceSelection(
                    candidate_frame_count=candidate_count,
                    eligible_frame_count=eligible_count,
                    selected_frame_index=None,
                    face_detected=face_detected,
                    face_count=max_face_count,
                    quality_score=None,
                    selection_outcome=outcome,
                ),
            )
        index, quality_score, crop = selected
        return LivenessEvaluation(
            result,
            LiveFaceSelection(
                candidate_frame_count=candidate_count,
                eligible_frame_count=eligible_count,
                selected_frame_index=index,
                face_detected=True,
                face_count=1,
                quality_score=quality_score,
                selection_outcome="selected",
                face_crop=crop,
            ),
        )

    def _passive_evaluate(
        self, frames: tuple[Image, ...]
    ) -> tuple[LivenessResult, list[AntiSpoofResult]]:
        evaluations = [self._detector.detect(frame) for frame in frames]
        frames_evaluated = len(evaluations)
        real_frames = sum(result.is_real for result in evaluations)
        passive_score = (
            sum(result.score for result in evaluations) / frames_evaluated
            if frames_evaluated
            else 0.0
        )
        passed = (
            frames_evaluated == self._config.frame_count
            and real_frames / frames_evaluated >= self._config.minimum_real_ratio
        )
        return (
            LivenessResult(
                passed=passed,
                passive_score=passive_score,
                frames_evaluated=frames_evaluated,
                real_frames=real_frames,
            ),
            evaluations,
        )


def _empty_selection(outcome: str) -> LiveFaceSelection:
    return LiveFaceSelection(0, 0, None, False, 0, None, outcome)


def _eligible_face_crop(
    frame: Image, box, config: LivenessEvaluationConfig
) -> tuple[Image, float] | None:
    """Return a crop only when geometry and basic face-crop quality pass."""
    try:
        height, width = frame.shape[:2]
        if box.right > width or box.bottom > height:
            return None
        if box.width < config.min_face_width or box.height < config.min_face_height:
            return None
        area_ratio = (box.width * box.height) / float(width * height)
        if area_ratio < config.min_face_area_ratio:
            return None
        crop = frame[box.y : box.bottom, box.x : box.right].copy()
        if crop.size == 0:
            return None
        sharpness = float(SharpnessAssessor().assess(crop).metrics["laplacian_variance"])
        brightness = float(BrightnessAssessor().assess(crop).metrics["mean_intensity"])
        contrast = float(ContrastAssessor().assess(crop).metrics["intensity_stddev"])
        if (
            sharpness < config.min_sharpness
            or not config.min_brightness <= brightness <= config.max_brightness
            or contrast < config.min_contrast
        ):
            return None
        return crop, _quality_score(
            box.x,
            box.y,
            box.width,
            box.height,
            width,
            height,
            area_ratio,
            sharpness,
            brightness,
        )
    except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
        return None


def _quality_score(
    x: int,
    y: int,
    face_width: int,
    face_height: int,
    image_width: int,
    image_height: int,
    area_ratio: float,
    sharpness: float,
    brightness: float,
) -> float:
    """Small deterministic score: 40% size, 30% sharpness, 20% exposure, 10% centering."""
    size = min(1.0, area_ratio / 0.20)
    sharp = min(1.0, sharpness / 10_000.0)
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    face_center = np.array((x + face_width / 2.0, y + face_height / 2.0))
    image_center = np.array((image_width / 2.0, image_height / 2.0))
    max_distance = float(np.hypot(image_width / 2.0, image_height / 2.0))
    center = max(0.0, 1.0 - float(np.linalg.norm(face_center - image_center)) / max_distance)
    return float(0.40 * size + 0.30 * sharp + 0.20 * exposure + 0.10 * center)
