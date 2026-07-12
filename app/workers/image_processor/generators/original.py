from typing import Mapping

from ..models import Image, VariantBatch, VariantInfo


class OriginalVariantGenerator:
    name = "original"

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        return VariantBatch(
            generator=self.name,
            variants=(
                VariantInfo(
                    name="original",
                    image=image.copy(),
                    parameters={},
                ),
            ),
        )
