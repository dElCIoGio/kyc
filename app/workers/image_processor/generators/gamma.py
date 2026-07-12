from typing import Mapping, Sequence

import cv2
import numpy as np

from ..models import Image, VariantBatch, VariantInfo
from .utils import format_value, get_parameter, require_numeric_sequence


class GammaVariantGenerator:
    name = "gamma"

    def __init__(self, *, gamma_values: Sequence[float] = (0.8, 1.0, 1.2)) -> None:
        self.gamma_values = tuple(gamma_values)

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        gamma_values = require_numeric_sequence(
            get_parameter(parameters, "gamma_values", self.gamma_values),
            "gamma_values",
            positive=True,
        )

        variants = []
        for gamma in gamma_values:
            inverse_gamma = 1.0 / gamma
            table = np.array(
                [
                    ((value / 255.0) ** inverse_gamma) * 255
                    for value in range(256)
                ],
                dtype=np.uint8,
            )
            variants.append(
                VariantInfo(
                    name=f"gamma_{format_value(gamma)}",
                    image=cv2.LUT(image, table),
                    parameters={"gamma": gamma},
                )
            )

        return VariantBatch(generator=self.name, variants=tuple(variants))
