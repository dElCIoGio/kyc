# variants/validation.py

import numpy as np

from .models import Image


def validate_image(image: Image) -> None:
    if image is None:
        raise ValueError("Image cannot be None")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, received {type(image).__name__}"
        )

    if image.size == 0:
        raise ValueError("Image cannot be empty")

    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, received {image.dtype}"
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D or 3D image, received shape {image.shape}"
        )

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"Unsupported number of channels: {image.shape[2]}"
        )