from typing import Mapping, Sequence

import cv2

from ..models import Image, VariantBatch, VariantInfo
from .utils import format_value, get_parameter, require_numeric_sequence, require_sequence, to_grayscale


class ThresholdVariantGenerator:
    name = "threshold"

    _SUPPORTED_METHODS = ("otsu", "adaptive_mean", "adaptive_gaussian")

    def __init__(
        self,
        *,
        methods: Sequence[str] = ("otsu", "adaptive_mean", "adaptive_gaussian"),
        adaptive_block_sizes: Sequence[int] = (11,),
        adaptive_c_values: Sequence[int | float] = (2,),
    ) -> None:
        self.methods = tuple(methods)
        self.adaptive_block_sizes = tuple(adaptive_block_sizes)
        self.adaptive_c_values = tuple(adaptive_c_values)

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        methods = self._resolve_methods(
            get_parameter(parameters, "methods", self.methods)
        )
        adaptive_block_sizes = self._resolve_block_sizes(
            get_parameter(
                parameters,
                "adaptive_block_sizes",
                self.adaptive_block_sizes,
            )
        )
        adaptive_c_values = require_numeric_sequence(
            get_parameter(parameters, "adaptive_c_values", self.adaptive_c_values),
            "adaptive_c_values",
        )

        gray_image = to_grayscale(image)
        variants = []

        for method in methods:
            if method == "otsu":
                _, thresholded = cv2.threshold(
                    gray_image,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                variants.append(
                    VariantInfo(
                        name="threshold_otsu",
                        image=thresholded,
                        parameters={"method": method},
                    )
                )
                continue

            adaptive_method = (
                cv2.ADAPTIVE_THRESH_MEAN_C
                if method == "adaptive_mean"
                else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            )

            for block_size in adaptive_block_sizes:
                for c_value in adaptive_c_values:
                    variants.append(
                        VariantInfo(
                            name=(
                                f"threshold_{method}_"
                                f"block_{block_size}_"
                                f"c_{format_value(c_value)}"
                            ),
                            image=cv2.adaptiveThreshold(
                                gray_image,
                                255,
                                adaptive_method,
                                cv2.THRESH_BINARY,
                                block_size,
                                c_value,
                            ),
                            parameters={
                                "method": method,
                                "block_size": block_size,
                                "c": c_value,
                            },
                        )
                    )

        return VariantBatch(generator=self.name, variants=tuple(variants))

    def _resolve_methods(self, value: object) -> tuple[str, ...]:
        methods = []

        for method in require_sequence(value, "methods"):
            if not isinstance(method, str):
                raise TypeError("methods values must be strings")

            normalized = method.lower()
            if normalized not in self._SUPPORTED_METHODS:
                raise ValueError(f"Unsupported threshold method: {method}")

            methods.append(normalized)

        return tuple(methods)

    def _resolve_block_sizes(self, value: object) -> tuple[int, ...]:
        block_sizes = []

        for block_size in require_sequence(value, "adaptive_block_sizes"):
            if isinstance(block_size, bool) or not isinstance(block_size, int):
                raise TypeError("adaptive_block_sizes values must be integers")

            if block_size <= 1 or block_size % 2 == 0:
                raise ValueError(
                    "adaptive_block_sizes values must be odd integers greater than 1"
                )

            block_sizes.append(block_size)

        return tuple(block_sizes)
