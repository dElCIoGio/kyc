from typing import Mapping, Sequence

import cv2

from ..models import Image, VariantBatch, VariantInfo
from .utils import format_value, get_parameter, require_adjustments


class ContrastBrightnessVariantGenerator:
    name = "contrast_brightness"

    def __init__(
        self,
        *,
        adjustments: Sequence[Mapping[str, float | int]] = (
            {"alpha": 1.0, "beta": 0},
            {"alpha": 1.2, "beta": 10},
        ),
    ) -> None:
        self.adjustments = tuple(dict(adjustment) for adjustment in adjustments)

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        adjustments = require_adjustments(
            get_parameter(parameters, "adjustments", self.adjustments)
        )

        variants = []
        for adjustment in adjustments:
            alpha = adjustment["alpha"]
            beta = adjustment["beta"]
            variants.append(
                VariantInfo(
                    name=(
                        "contrast_"
                        f"{format_value(alpha)}_"
                        f"brightness_{format_value(beta)}"
                    ),
                    image=cv2.convertScaleAbs(image, alpha=alpha, beta=beta),
                    parameters={"alpha": alpha, "beta": beta},
                )
            )

        return VariantBatch(generator=self.name, variants=tuple(variants))
