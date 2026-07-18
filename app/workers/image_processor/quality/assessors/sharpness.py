from typing import Mapping

import cv2

from ...models import Image
from ..models import AssessmentResult
from ..utils import normalize_parameters, require_int, to_grayscale


class SharpnessAssessor:
    name = "sharpness"
    _allowed_parameters = frozenset({"kernel_size"})

    def __init__(self, *, kernel_size: int = 3) -> None:
        self.kernel_size = self._validate_kernel_size(kernel_size)

    def assess(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> AssessmentResult:
        normalized = normalize_parameters(
            parameters,
            allowed_keys=self._allowed_parameters,
        )
        kernel_size = self._validate_kernel_size(
            normalized.get("kernel_size", self.kernel_size)
        )
        grayscale = to_grayscale(image)
        laplacian = cv2.Laplacian(grayscale, cv2.CV_64F, ksize=kernel_size)

        return AssessmentResult(
            assessor=self.name,
            metrics={"laplacian_variance": float(laplacian.var())},
            parameters={"kernel_size": kernel_size},
        )

    @staticmethod
    def _validate_kernel_size(value: object) -> int:
        kernel_size = require_int(value, "kernel_size")
        if kernel_size < 1 or kernel_size > 31 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer no greater than 31"
            )
        return kernel_size
