from typing import Mapping, Sequence

import cv2

from ..models import Image, VariantBatch, VariantInfo
from .utils import format_value, get_parameter, require_numeric_sequence


class SharpeningVariantGenerator:
    name = "sharpening"

    def __init__(self, *, strengths: Sequence[float] = (1.0,)) -> None:
        self.strengths = tuple(strengths)

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        strengths = require_numeric_sequence(
            get_parameter(parameters, "strengths", self.strengths),
            "strengths",
            non_negative=True,
        )

        variants = []
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)

        for strength in strengths:
            variants.append(
                VariantInfo(
                    name=f"sharpen_{format_value(strength)}",
                    image=cv2.addWeighted(
                        image,
                        1.0 + strength,
                        blurred,
                        -strength,
                        0,
                    ),
                    parameters={"strength": strength},
                )
            )

        return VariantBatch(generator=self.name, variants=tuple(variants))
