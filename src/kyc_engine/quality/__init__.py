from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import Mapping, Protocol, Sequence

import cv2
import numpy as np

from ..contracts import (
    AssessmentResult,
    Image,
    VariantAssessment,
    VariantBatch,
    VariantBatchAssessment,
)


class ImageQualityAssessor(Protocol):
    name: str

    def assess(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> AssessmentResult:
        ...


class SharpnessAssessor:
    name = "sharpness"

    def __init__(self, kernel_size: int = 3) -> None:
        self.kernel_size = _kernel_size(kernel_size)

    def assess(
        self, image: Image, parameters: Mapping[str, object] | None = None
    ) -> AssessmentResult:
        values = _parameters(parameters, {"kernel_size"})
        kernel_size = _kernel_size(values.get("kernel_size", self.kernel_size))
        variance = float(
            cv2.Laplacian(to_grayscale(image), cv2.CV_64F, ksize=kernel_size).var()
        )
        return AssessmentResult(
            self.name,
            {"laplacian_variance": variance},
            {"kernel_size": kernel_size},
        )


class BrightnessAssessor:
    name = "brightness"

    def assess(
        self, image: Image, parameters: Mapping[str, object] | None = None
    ) -> AssessmentResult:
        _parameters(parameters, set())
        gray = to_grayscale(image)
        return AssessmentResult(
            self.name,
            {
                "mean_intensity": float(np.mean(gray)),
                "median_intensity": float(np.median(gray)),
            },
            {},
        )


class ContrastAssessor:
    name = "contrast"

    def assess(
        self, image: Image, parameters: Mapping[str, object] | None = None
    ) -> AssessmentResult:
        _parameters(parameters, set())
        return AssessmentResult(
            self.name,
            {"intensity_stddev": float(np.std(to_grayscale(image)))},
            {},
        )


class DynamicRangeAssessor:
    name = "dynamic_range"

    def __init__(
        self, lower_percentile: float = 1.0, upper_percentile: float = 99.0
    ) -> None:
        self.lower_percentile, self.upper_percentile = _percentiles(
            lower_percentile, upper_percentile
        )

    def assess(
        self, image: Image, parameters: Mapping[str, object] | None = None
    ) -> AssessmentResult:
        values = _parameters(parameters, {"lower_percentile", "upper_percentile"})
        lower, upper = _percentiles(
            values.get("lower_percentile", self.lower_percentile),
            values.get("upper_percentile", self.upper_percentile),
        )
        gray = to_grayscale(image)
        low, high = np.percentile(gray, (lower, upper))
        minimum, maximum = float(np.min(gray)), float(np.max(gray))
        return AssessmentResult(
            self.name,
            {
                "absolute_min": minimum,
                "absolute_max": maximum,
                "absolute_range": maximum - minimum,
                "percentile_low": float(low),
                "percentile_high": float(high),
                "percentile_range": float(high - low),
            },
            {"lower_percentile": lower, "upper_percentile": upper},
        )


class HistogramClippingAssessor:
    name = "histogram_clipping"

    def __init__(self, dark_threshold: int = 5, bright_threshold: int = 250) -> None:
        self.dark_threshold, self.bright_threshold = _thresholds(
            dark_threshold, bright_threshold
        )

    def assess(
        self, image: Image, parameters: Mapping[str, object] | None = None
    ) -> AssessmentResult:
        values = _parameters(parameters, {"dark_threshold", "bright_threshold"})
        dark, bright = _thresholds(
            values.get("dark_threshold", self.dark_threshold),
            values.get("bright_threshold", self.bright_threshold),
        )
        gray = to_grayscale(image)
        return AssessmentResult(
            self.name,
            {
                "exact_black_ratio": float(np.mean(gray == 0)),
                "exact_white_ratio": float(np.mean(gray == 255)),
                "dark_pixel_ratio": float(np.mean(gray <= dark)),
                "bright_pixel_ratio": float(np.mean(gray >= bright)),
            },
            {"dark_threshold": dark, "bright_threshold": bright},
        )


class VariantQualityAssessmentPipeline:
    def __init__(
        self, assessors: Sequence[ImageQualityAssessor] | None = None
    ) -> None:
        selected = tuple(default_assessors() if assessors is None else assessors)
        if not selected:
            raise ValueError("At least one quality assessor is required")
        self._assessors: dict[str, ImageQualityAssessor] = {}
        for assessor in selected:
            name = assessor.name
            if not isinstance(name, str):
                raise TypeError("Assessor name must be a string")
            if not name.strip():
                raise ValueError("Assessor name cannot be empty")
            if name in self._assessors:
                raise ValueError(f"Duplicate assessor name: {name}")
            self._assessors[name] = assessor

    @property
    def assessor_names(self) -> tuple[str, ...]:
        return tuple(self._assessors)

    @property
    def assessors(self) -> tuple[ImageQualityAssessor, ...]:
        return tuple(self._assessors.values())

    def assess(
        self,
        batches: Sequence[VariantBatch],
        *,
        selected_assessors: Sequence[str] | None = None,
        parameters: Mapping[str, Mapping[str, object]] | None = None,
    ) -> tuple[VariantBatchAssessment, ...]:
        names = self._resolve_names(selected_assessors)
        parameter_map = self._resolve_parameters(parameters)
        results = []
        for batch in batches:
            _validate_batch(batch)
            results.append(
                VariantBatchAssessment(
                    generator=batch.generator,
                    assessments=tuple(
                        VariantAssessment(
                            variant_name=variant.name,
                            generator=batch.generator,
                            variant_parameters=variant.parameters,
                            results=tuple(
                                self._assessors[name].assess(
                                    variant.image.copy(), parameter_map.get(name)
                                )
                                for name in names
                            ),
                        )
                        for variant in batch.variants
                    ),
                )
            )
        return tuple(results)

    def _resolve_names(self, selected: Sequence[str] | None) -> tuple[str, ...]:
        if selected is None:
            return self.assessor_names
        if isinstance(selected, (str, bytes)) or not isinstance(selected, SequenceABC):
            raise TypeError("selected_assessors must be a sequence of names")
        names = tuple(selected)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("selected_assessors must contain non-empty names")
        if len(names) != len(set(names)):
            raise ValueError("selected_assessors cannot contain duplicate names")
        unknown = tuple(name for name in names if name not in self._assessors)
        if unknown:
            raise KeyError(f"Unknown assessor names: {', '.join(unknown)}")
        return names

    def _resolve_parameters(
        self, parameters: Mapping[str, Mapping[str, object]] | None
    ) -> Mapping[str, Mapping[str, object]]:
        if parameters is None:
            return {}
        if not isinstance(parameters, MappingABC):
            raise TypeError("parameters must be a mapping")
        unknown = tuple(name for name in parameters if name not in self._assessors)
        if unknown:
            formatted = ", ".join(repr(name) for name in unknown)
            raise KeyError(f"Parameters provided for unknown assessors: {formatted}")
        if any(not isinstance(values, MappingABC) for values in parameters.values()):
            raise TypeError("Each assessor parameter value must be a mapping")
        return parameters


QualityAssessmentPipeline = VariantQualityAssessmentPipeline


class StructuralQualityGate:
    """Reject only constant variants until calibrated gates are available."""

    def select(
        self,
        batches: Sequence[VariantBatch],
        assessments: Sequence[VariantBatchAssessment],
    ) -> tuple[VariantBatch, ...]:
        usable: set[tuple[str, str]] = set()
        for batch in assessments:
            for variant in batch.assessments:
                dynamic = next(
                    (
                        item
                        for item in variant.results
                        if item.assessor == "dynamic_range"
                    ),
                    None,
                )
                if dynamic is not None and dynamic.metrics["absolute_range"] > 0:
                    usable.add((batch.generator, variant.variant_name))

        selected = []
        for batch in batches:
            variants = tuple(
                variant
                for variant in batch.variants
                if (batch.generator, variant.name) in usable
            )
            if variants:
                selected.append(VariantBatch(batch.generator, variants))
        return tuple(selected)


def default_assessors() -> tuple[ImageQualityAssessor, ...]:
    return (
        SharpnessAssessor(),
        BrightnessAssessor(),
        ContrastAssessor(),
        DynamicRangeAssessor(),
        HistogramClippingAssessor(),
    )


def to_grayscale(image: Image) -> Image:
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array")
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Image must be a non-empty uint8 array")
    if image.ndim == 2:
        return image.copy()
    if image.ndim != 3:
        raise ValueError("Image must have two or three dimensions")
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0].copy()
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("Image must have 1, 3, or 4 channels")


def _parameters(
    parameters: Mapping[str, object] | None, allowed: set[str]
) -> Mapping[str, object]:
    if parameters is None:
        return {}
    if not isinstance(parameters, MappingABC):
        raise TypeError("Assessor parameters must be a mapping")
    unknown = tuple(key for key in parameters if key not in allowed)
    if unknown:
        formatted = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"Unknown assessor parameters: {formatted}")
    return parameters


def _kernel_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("kernel_size must be an integer")
    if value <= 0 or value % 2 == 0 or value > 31:
        raise ValueError("kernel_size must be a positive odd integer no greater than 31")
    return value


def _percentiles(lower: object, upper: object) -> tuple[float, float]:
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (lower, upper)
    ):
        raise TypeError("Percentiles must be numeric")
    values = (float(lower), float(upper))
    if not 0.0 <= values[0] < values[1] <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")
    return values


def _thresholds(dark: object, bright: object) -> tuple[int, int]:
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (dark, bright)
    ):
        raise TypeError("Clipping thresholds must be integers")
    if not 0 <= dark < bright <= 255:
        raise ValueError("Thresholds must satisfy 0 <= dark < bright <= 255")
    return dark, bright


def _validate_batch(batch: VariantBatch) -> None:
    if not isinstance(batch, VariantBatch):
        raise TypeError("Quality input must contain VariantBatch values")
    if not batch.generator.strip() or not batch.variants:
        raise ValueError("Variant batches require a generator name and at least one variant")
    for variant in batch.variants:
        if not variant.name.strip():
            raise ValueError("Variant names cannot be empty")
        to_grayscale(variant.image)


__all__ = [
    "BrightnessAssessor",
    "ContrastAssessor",
    "DynamicRangeAssessor",
    "HistogramClippingAssessor",
    "ImageQualityAssessor",
    "QualityAssessmentPipeline",
    "SharpnessAssessor",
    "StructuralQualityGate",
    "VariantQualityAssessmentPipeline",
    "default_assessors",
    "to_grayscale",
]
