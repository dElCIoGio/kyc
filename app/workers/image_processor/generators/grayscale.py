from typing import Mapping

from ..models import Image, VariantBatch, VariantInfo
from .utils import to_grayscale


class GrayscaleVariantGenerator:
    name = "grayscale"

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        return VariantBatch(
            generator=self.name,
            variants=(
                VariantInfo(
                    name="grayscale",
                    image=to_grayscale(image),
                    parameters={},
                ),
            ),
        )
