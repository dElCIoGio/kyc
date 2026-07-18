from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC

from ..models import VariantBatch, VariantInfo
from ..validator import validate_image


def validate_variant_batch(batch: VariantBatch) -> None:
    if not isinstance(batch, VariantBatch):
        raise TypeError(
            f"Expected VariantBatch, received {type(batch).__name__}"
        )

    if not isinstance(batch.generator, str):
        raise TypeError("Variant batch generator name must be a string")

    if not batch.generator.strip():
        raise ValueError("Variant batch generator name cannot be empty")

    if isinstance(batch.variants, (str, bytes)) or not isinstance(
        batch.variants, SequenceABC
    ):
        raise TypeError("Variant batch variants must be a sequence")

    for index, variant in enumerate(batch.variants):
        _validate_variant(variant, index=index)


def _validate_variant(variant: object, *, index: int) -> None:
    if not isinstance(variant, VariantInfo):
        raise TypeError(f"Variant at index {index} must be a VariantInfo")

    if not isinstance(variant.name, str):
        raise TypeError(f"Variant name at index {index} must be a string")

    if not variant.name.strip():
        raise ValueError(f"Variant name at index {index} cannot be empty")

    if not isinstance(variant.parameters, MappingABC):
        raise TypeError(f"Variant parameters at index {index} must be a mapping")

    validate_image(variant.image)
