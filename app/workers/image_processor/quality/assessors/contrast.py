from typing import Mapping

import numpy as np

from ...models import Image
from ..models import AssessmentResult
from ..utils import normalize_parameters, to_grayscale


class ContrastAssessor:
    name = "contrast"
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
            metrics={"intensity_stddev": float(np.std(grayscale))},
            parameters={},
        )
