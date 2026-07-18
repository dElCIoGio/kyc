from typing import Mapping

import numpy as np

from ...models import Image
from ..models import AssessmentResult
from ..utils import normalize_parameters, require_real, to_grayscale


class DynamicRangeAssessor:
    name = "dynamic_range"
    _allowed_parameters = frozenset({"lower_percentile", "upper_percentile"})

    def __init__(
        self,
        *,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> None:
        self.lower_percentile, self.upper_percentile = self._validate_percentiles(
            lower_percentile,
            upper_percentile,
        )

    def assess(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> AssessmentResult:
        normalized = normalize_parameters(
            parameters,
            allowed_keys=self._allowed_parameters,
        )
        lower, upper = self._validate_percentiles(
            normalized.get("lower_percentile", self.lower_percentile),
            normalized.get("upper_percentile", self.upper_percentile),
        )
        grayscale = to_grayscale(image)
        absolute_min = float(np.min(grayscale))
        absolute_max = float(np.max(grayscale))
        percentile_low = float(np.percentile(grayscale, lower))
        percentile_high = float(np.percentile(grayscale, upper))

        return AssessmentResult(
            assessor=self.name,
            metrics={
                "absolute_min": absolute_min,
                "absolute_max": absolute_max,
                "absolute_range": absolute_max - absolute_min,
                "percentile_low": percentile_low,
                "percentile_high": percentile_high,
                "percentile_range": percentile_high - percentile_low,
            },
            parameters={
                "lower_percentile": lower,
                "upper_percentile": upper,
            },
        )

    @staticmethod
    def _validate_percentiles(
        lower_value: object,
        upper_value: object,
    ) -> tuple[float, float]:
        lower = require_real(lower_value, "lower_percentile")
        upper = require_real(upper_value, "upper_percentile")
        if not 0 <= lower < upper <= 100:
            raise ValueError(
                "percentiles must satisfy "
                "0 <= lower_percentile < upper_percentile <= 100"
            )
        return lower, upper
