# Variant Generation Architecture

## Goal

The image processor should read an input image once with `cv2.imread(path)`, validate it, and pass the resulting OpenCV/Numpy image into a central `VariantGenerationContext`. The context coordinates multiple strategy implementations, where each strategy is responsible for generating one family of image variants.

This keeps filter logic isolated, makes generator configuration explicit, and gives downstream code a consistent strongly typed result shape.

## Core Flow

```text
path
  -> cv2.imread(path)
  -> validate_image(image)
  -> VariantGenerationContext.generate(image, parameters=...)
  -> Sequence[VariantBatch]
```

Important rules:

- `cv2.imread(path)` happens before variant generation.
- Generators receive an already-loaded image, not a file path.
- Generators must not mutate the original input image.
- Internal images use OpenCV's default BGR channel order unless a generator explicitly documents otherwise.
- Generated variants stay in memory as `np.ndarray` values; writing images to disk is outside the initial scope.

## Public Types

Use Python 3.12 typing, `dataclasses`, and `typing.Protocol`.

```python
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

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


class VariantGenerator(Protocol):
    name: str

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        ...
```

## VariantGenerationContext

`VariantGenerationContext` is the registry and orchestrator for variant strategies.

Responsibilities:

- Accept generator instances at construction time.
- Index generators by their unique `name`.
- Run all registered generators by default, or a selected subset when requested.
- Pass per-generator runtime parameters to the correct strategy.
- Return one `VariantBatch` per executed generator.
- Raise clear errors for duplicate generator names, unknown selected names, and invalid parameter payloads.

Target usage:

```python
context = VariantGenerationContext(
    generators=[
        OriginalVariantGenerator(),
        GammaVariantGenerator(gamma_values=(0.8, 1.0, 1.2)),
    ]
)

batches = context.generate(
    image,
    parameters={
        "gamma": {"gamma_values": [0.7, 1.2]},
    },
)
```

## Strategy Implementations

Each concrete generator owns one image transformation strategy. Generators may accept constructor defaults and may also accept runtime overrides through `parameters`.

Example:

```python
GammaVariantGenerator(
    gamma_values=(0.8, 1.0, 1.2)
)
```

When executed, this generator returns:

```python
VariantBatch(
    generator="gamma",
    variants=[
        VariantInfo(name="gamma_0.8", image=..., parameters={"gamma": 0.8}),
        VariantInfo(name="gamma_1.0", image=..., parameters={"gamma": 1.0}),
        VariantInfo(name="gamma_1.2", image=..., parameters={"gamma": 1.2}),
    ],
)
```

Runtime parameters should override constructor defaults only for that generation call:

```python
parameters={"gamma_values": [0.7, 1.2]}
```

## Expected Module Layout

```text
image_processor/
  main.py
  models.py
  validator.py
  context.py
  generators/
    __init__.py
    base.py
    original.py
    grayscale.py
    clahe.py
    gamma.py
    contrast_brightness.py
    sharpening.py
    denoising.py
    threshold.py
```

## Validation

The existing `validator.py` should be aligned with `models.py` once `Image` exists.

Validation should reject:

- `None`
- Non-`np.ndarray` values
- Empty arrays
- Non-`uint8` arrays
- Unsupported dimensions
- Unsupported channel counts

Generators should call validation at their boundary or rely on the context to validate once before dispatch. The initial implementation should validate once in the context before any generator runs.
