# Image Processor Worker

This worker will generate preprocessing variants for one image. The intended flow is:

```text
image_path -> cv2.imread(path) -> validate_image(image) -> VariantGenerationContext -> VariantBatch[]
```

The image should be read once with `cv2.imread(path)`, then passed as an in-memory OpenCV/Numpy image to a configurable set of variant generators. Each generator follows the strategy pattern and returns one strongly typed `VariantBatch` containing all variants produced by that strategy.

## Current State

- `main.py` is still a placeholder entrypoint.
- `validator.py` validates `np.ndarray` images and already expects an `Image` type from a future `models.py`.
- `pyproject.toml` does not yet declare `numpy` or `opencv-python`, which will be required for implementation.

## Design Docs

- [Variant generation architecture](docs/variant-generation-architecture.md)
- [Generator specifications](docs/generator-specifications.md)
- [Implementation roadmap](docs/implementation-roadmap.md)

## Target Usage Shape

```python
image = cv2.imread(path)
validate_image(image)

context = VariantGenerationContext(
    generators=[
        OriginalVariantGenerator(),
        GammaVariantGenerator(gamma_values=(0.8, 1.0, 1.2)),
        ThresholdVariantGenerator(methods=("otsu", "adaptive_mean")),
    ]
)

batches = context.generate(
    image,
    parameters={
        "gamma": {"gamma_values": [0.7, 1.2]},
        "threshold": {"methods": ["otsu", "adaptive_mean"]},
    },
)
```

Each item in `batches` is a `VariantBatch`, for example:

```python
VariantBatch(
    generator="gamma",
    variants=[
        VariantInfo(name="gamma_0.7", image=..., parameters={"gamma": 0.7}),
        VariantInfo(name="gamma_1.2", image=..., parameters={"gamma": 1.2}),
    ],
)
```
