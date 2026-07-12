from typing import Mapping, Sequence

import cv2
import numpy as np

from ..models import Image, VariantBatch, VariantInfo
from .utils import format_value, get_parameter, require_numeric_sequence


class DenoisingVariantGenerator:
    name = "denoising"

    def __init__(self, *, h_values: Sequence[float] = (10.0,)) -> None:
        self.h_values = tuple(h_values)

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        h_values = require_numeric_sequence(
            get_parameter(parameters, "h_values", self.h_values),
            "h_values",
            non_negative=True,
        )

        variants = []
        for h_value in h_values:
            variants.append(
                VariantInfo(
                    name=f"denoise_h_{format_value(h_value)}",
                    image=self._denoise(image, h_value),
                    parameters={"h": h_value},
                )
            )

        return VariantBatch(generator=self.name, variants=tuple(variants))

    def _denoise(self, image: Image, h_value: float) -> Image:
        if image.ndim == 2:
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h=h_value,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        channels = image.shape[2]
        if channels == 1:
            denoised = cv2.fastNlMeansDenoising(
                image[:, :, 0],
                None,
                h=h_value,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            return denoised[:, :, np.newaxis]

        if channels == 3:
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=h_value,
                hColor=h_value,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        if channels == 4:
            color = cv2.fastNlMeansDenoisingColored(
                image[:, :, :3],
                None,
                h=h_value,
                hColor=h_value,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            return np.dstack((color, image[:, :, 3]))

        raise ValueError(f"Unsupported number of channels: {channels}")
