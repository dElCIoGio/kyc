"""Variant generator implementations."""

from .base import VariantGenerator

__all__ = [
    "ClaheVariantGenerator",
    "ContrastBrightnessVariantGenerator",
    "DenoisingVariantGenerator",
    "GammaVariantGenerator",
    "GrayscaleVariantGenerator",
    "OriginalVariantGenerator",
    "SharpeningVariantGenerator",
    "ThresholdVariantGenerator",
    "VariantGenerator",
]


def __getattr__(name: str) -> object:
    if name == "ClaheVariantGenerator":
        from .clahe import ClaheVariantGenerator

        return ClaheVariantGenerator

    if name == "ContrastBrightnessVariantGenerator":
        from .contrast_brightness import ContrastBrightnessVariantGenerator

        return ContrastBrightnessVariantGenerator

    if name == "DenoisingVariantGenerator":
        from .denoising import DenoisingVariantGenerator

        return DenoisingVariantGenerator

    if name == "GammaVariantGenerator":
        from .gamma import GammaVariantGenerator

        return GammaVariantGenerator

    if name == "GrayscaleVariantGenerator":
        from .grayscale import GrayscaleVariantGenerator

        return GrayscaleVariantGenerator

    if name == "OriginalVariantGenerator":
        from .original import OriginalVariantGenerator

        return OriginalVariantGenerator

    if name == "SharpeningVariantGenerator":
        from .sharpening import SharpeningVariantGenerator

        return SharpeningVariantGenerator

    if name == "ThresholdVariantGenerator":
        from .threshold import ThresholdVariantGenerator

        return ThresholdVariantGenerator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
