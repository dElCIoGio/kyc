from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


@dataclass(frozen=True)
class VariantInfo:
    name: str
    image: Image
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class VariantBatch:
    generator: str
    variants: Sequence[VariantInfo]
