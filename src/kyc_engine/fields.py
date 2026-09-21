from __future__ import annotations

from typing import Sequence

from .contracts import BoundingBox, DocumentProfile, FieldCrop, VariantBatch


class FieldLocalizationError(ValueError):
    pass


class FieldLocalizer:
    def __init__(self, *, max_padding: int = 32) -> None:
        if max_padding < 0:
            raise ValueError("max_padding cannot be negative")
        self.max_padding = max_padding

    def localize(
        self,
        batches: Sequence[VariantBatch],
        profile: DocumentProfile,
    ) -> tuple[FieldCrop, ...]:
        crops: list[FieldCrop] = []
        for batch in batches:
            for variant in batch.variants:
                height, width = variant.image.shape[:2]
                if width != profile.canonical_width or height != profile.canonical_height:
                    raise FieldLocalizationError(
                        f"Variant '{variant.name}' dimensions do not match profile"
                    )
                for field in profile.fields:
                    if field.padding > self.max_padding:
                        raise FieldLocalizationError(
                            f"Field '{field.name}' padding exceeds configured limit"
                        )
                    box = _padded_box(
                        field.bounding_box,
                        field.padding,
                        width,
                        height,
                    )
                    pixels = variant.image[box.y : box.bottom, box.x : box.right].copy()
                    if pixels.size == 0:
                        raise FieldLocalizationError(
                            f"Field '{field.name}' produced an empty crop"
                        )
                    crops.append(
                        FieldCrop(
                            field_name=field.name,
                            generator=batch.generator,
                            variant_name=variant.name,
                            image=pixels,
                            bounding_box=box,
                            ocr_mode=field.ocr_mode,
                        )
                    )
        return tuple(crops)


def _padded_box(
    box: BoundingBox,
    padding: int,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    left = max(0, box.x - padding)
    top = max(0, box.y - padding)
    right = min(image_width, box.right + padding)
    bottom = min(image_height, box.bottom + padding)
    return BoundingBox(left, top, right - left, bottom - top)

