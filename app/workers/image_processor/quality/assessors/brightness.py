from typing import Mapping

import numpy as np

from ...models import Image
from ..models import AssessmentResult
from ..utils import normalize_parameters, to_grayscale


class BrightnessAssessor:
    name = "brightness"
    _allowed_parameters: frozenset[str] = frozenset()

    def assess(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> AssessmentResult:
        normalize_parameters(parameters, allowed_keys=self._allowed_parameters)
        grayscale = to_grayscale(image)

        return AssessmentResult(
            assessor=self.name,
            metrics={
                "mean_intensity": float(np.mean(grayscale)),
                "median_intensity": float(np.median(grayscale)),
            },
            parameters={},
        )
