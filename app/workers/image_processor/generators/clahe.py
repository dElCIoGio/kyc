from typing import Mapping, Sequence

import cv2

from ..models import Image, VariantBatch, VariantInfo
from .utils import (
    format_value,
    get_parameter,
    require_grid_sizes,
    require_numeric_sequence,
    to_grayscale,
)


class ClaheVariantGenerator:
    name = "clahe"

    def __init__(
        self,
        *,
        clip_limits: Sequence[float] = (2.0,),
        tile_grid_sizes: Sequence[tuple[int, int]] = ((8, 8),),
    ) -> None:
        self.clip_limits = tuple(clip_limits)
        self.tile_grid_sizes = tuple(tile_grid_sizes)

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        clip_limits = require_numeric_sequence(
            get_parameter(parameters, "clip_limits", self.clip_limits),
            "clip_limits",
            positive=True,
        )
        tile_grid_sizes = require_grid_sizes(
            get_parameter(parameters, "tile_grid_sizes", self.tile_grid_sizes),
            "tile_grid_sizes",
        )

        gray_image = to_grayscale(image)
        variants = []

        for clip_limit in clip_limits:
            for tile_grid_size in tile_grid_sizes:
                clahe = cv2.createCLAHE(
                    clipLimit=clip_limit,
                    tileGridSize=tile_grid_size,
                )
                variants.append(
                    VariantInfo(
                        name=(
                            "clahe_"
                            f"clip_{format_value(clip_limit)}_"
                            f"grid_{tile_grid_size[0]}x{tile_grid_size[1]}"
                        ),
                        image=clahe.apply(gray_image),
                        parameters={
                            "clip_limit": clip_limit,
                            "tile_grid_size": tile_grid_size,
                        },
                    )
                )

        return VariantBatch(generator=self.name, variants=tuple(variants))
