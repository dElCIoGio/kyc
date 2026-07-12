from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from numbers import Real
from typing import Mapping

import cv2

from ..models import Image


def normalize_parameters(
    parameters: Mapping[str, object] | None,
) -> Mapping[str, object]:
    if parameters is None:
        return {}

    if not isinstance(parameters, MappingABC):
        raise TypeError(
            f"Expected parameters to be a mapping, received {type(parameters).__name__}"
        )

    return parameters


def get_parameter(
    parameters: Mapping[str, object] | None,
    name: str,
    default: object,
) -> object:
    normalized = normalize_parameters(parameters)
    return normalized.get(name, default)


def require_sequence(value: object, parameter_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, SequenceABC):
        raise TypeError(f"{parameter_name} must be a sequence")

    items = tuple(value)
    if not items:
        raise ValueError(f"{parameter_name} cannot be empty")

    return items


def require_numeric_sequence(
    value: object,
    parameter_name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> tuple[float, ...]:
    values = []

    for item in require_sequence(value, parameter_name):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{parameter_name} values must be numeric")

        number = float(item)
        if positive and number <= 0:
            raise ValueError(f"{parameter_name} values must be greater than 0")

        if non_negative and number < 0:
            raise ValueError(f"{parameter_name} values cannot be negative")

        values.append(number)

    return tuple(values)


def require_grid_sizes(value: object, parameter_name: str) -> tuple[tuple[int, int], ...]:
    grid_sizes = []

    for item in require_sequence(value, parameter_name):
        if (
            isinstance(item, (str, bytes))
            or not isinstance(item, SequenceABC)
            or len(item) != 2
        ):
            raise TypeError(f"{parameter_name} values must be two-item sequences")

        width, height = item
        if isinstance(width, bool) or isinstance(height, bool):
            raise TypeError(f"{parameter_name} values must contain integers")

        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError(f"{parameter_name} values must contain integers")

        if width <= 0 or height <= 0:
            raise ValueError(f"{parameter_name} values must be greater than 0")

        grid_sizes.append((width, height))

    return tuple(grid_sizes)


def require_adjustments(value: object) -> tuple[dict[str, float], ...]:
    adjustments = []

    for item in require_sequence(value, "adjustments"):
        if not isinstance(item, MappingABC):
            raise TypeError("adjustments values must be mappings")

        alpha = item.get("alpha")
        beta = item.get("beta")

        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("adjustment alpha values must be numeric")

        if isinstance(beta, bool) or not isinstance(beta, Real):
            raise TypeError("adjustment beta values must be numeric")

        if float(alpha) < 0:
            raise ValueError("adjustment alpha values cannot be negative")

        adjustments.append({"alpha": float(alpha), "beta": float(beta)})

    return tuple(adjustments)


def to_grayscale(image: Image) -> Image:
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


def format_value(value: object) -> str:
    return str(value)
