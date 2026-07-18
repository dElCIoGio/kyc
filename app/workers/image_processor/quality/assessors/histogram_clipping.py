from typing import Mapping

import numpy as np

from ...models import Image
from ..models import AssessmentResult
from ..utils import normalize_parameters, require_int, to_grayscale


class HistogramClippingAssessor:
    name = "histogram_clipping"
    _allowed_parameters = frozenset({"dark_threshold", "bright_threshold"})

    def __init__(
        self,
        *,
        dark_threshold: int = 5,
        bright_threshold: int = 250,
    ) -> None:
        self.dark_threshold, self.bright_threshold = self._validate_thresholds(
            dark_threshold,
            bright_threshold,
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
        dark_threshold, bright_threshold = self._validate_thresholds(
            normalized.get("dark_threshold", self.dark_threshold),
            normalized.get("bright_threshold", self.bright_threshold),
        )
        grayscale = to_grayscale(image)

        return AssessmentResult(
            assessor=self.name,
            metrics={
                "exact_black_ratio": float(np.mean(grayscale == 0)),
                "exact_white_ratio": float(np.mean(grayscale == 255)),
                "dark_pixel_ratio": float(np.mean(grayscale <= dark_threshold)),
                "bright_pixel_ratio": float(
                    np.mean(grayscale >= bright_threshold)
                ),
            },
            parameters={
                "dark_threshold": dark_threshold,
                "bright_threshold": bright_threshold,
            },
        )

    @staticmethod
    def _validate_thresholds(
        dark_value: object,
        bright_value: object,
    ) -> tuple[int, int]:
        dark = require_int(dark_value, "dark_threshold")
        bright = require_int(bright_value, "bright_threshold")
        if not 0 <= dark < bright <= 255:
            raise ValueError(
                "thresholds must satisfy "
                "0 <= dark_threshold < bright_threshold <= 255"
            )
        return dark, bright
