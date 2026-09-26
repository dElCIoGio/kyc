from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from math import isfinite
from typing import Iterable

from .contracts import AntiSpoofDetector, Image, LivenessResult


@dataclass(frozen=True)
class LivenessEvaluationConfig:
    """Initial uncalibrated development policy for passive liveness aggregation."""

    frame_count: int = 3
    minimum_real_ratio: float = 2.0 / 3.0

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
        object.__setattr__(self, "minimum_real_ratio", float(self.minimum_real_ratio))


class LivenessEvaluator:
    """Aggregates ordered frame-level anti-spoof decisions without retaining frames."""

    def __init__(
        self,
        detector: AntiSpoofDetector,
        config: LivenessEvaluationConfig | None = None,
    ) -> None:
        self._detector = detector
        self._config = config or LivenessEvaluationConfig()

    def evaluate(self, frames: Iterable[Image]) -> LivenessResult:
        evaluations = [
            self._detector.detect(frame)
            for frame in islice(frames, self._config.frame_count)
        ]
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
        return LivenessResult(
            passed=passed,
            passive_score=passive_score,
            frames_evaluated=frames_evaluated,
            real_frames=real_frames,
        )
