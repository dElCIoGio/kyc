from typing import Mapping, Protocol

from ..models import Image, VariantBatch


class VariantGenerator(Protocol):
    name: str

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        ...
