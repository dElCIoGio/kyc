from collections.abc import Mapping as MappingABC
from numbers import Real
from typing import Mapping

import cv2

from ..models import Image
from ..validator import validate_image


def normalize_parameters(
    parameters: Mapping[str, object] | None,
    *,
    allowed_keys: frozenset[str],
) -> Mapping[str, object]:
    if parameters is None:
        return {}

    if not isinstance(parameters, MappingABC):
        raise TypeError(
            f"Expected parameters to be a mapping, received {type(parameters).__name__}"
        )

    unknown = tuple(key for key in parameters if key not in allowed_keys)
    if unknown:
        formatted = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"Unknown assessor parameter keys: {formatted}")

    return parameters


def require_real(value: object, parameter_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{parameter_name} must be numeric")

    return float(value)


def require_int(value: object, parameter_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{parameter_name} must be an integer")

    return value


def to_grayscale(image: Image) -> Image:
    validate_image(image)

    if image.ndim == 2:
        return image.copy()

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0].copy()

    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported number of channels: {channels}")
