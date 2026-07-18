# Image Processor Worker

This worker generates preprocessing variants for one image. The intended flow is:

```text
image_path -> cv2.imread(path) -> validate_image(image) -> VariantGenerationContext -> VariantBatch[]
```

The image is read once with `cv2.imread(path)`, then passed as an in-memory OpenCV/Numpy image to a configurable set of variant generators. Each generator follows the strategy pattern and returns one strongly typed `VariantBatch` containing all variants produced by that strategy.

## Current State

- `main.py` can generate all default variants and optionally save them to disk.
- `validator.py` validates `np.ndarray` images through the shared `Image` type alias.
- `pyproject.toml` declares `numpy` and `opencv-python`.

## CLI Usage

Generate variants and print a summary:

```powershell
python main.py "img.png"
```

Generate and save every variant:

```powershell
python main.py "img.png" --output-dir "variants"
```

Saved files are grouped by generator:

```text
variants/
  original/
    original.png
  grayscale/
    grayscale.png
  gamma/
    gamma_0.8.png
    gamma_1.0.png
    gamma_1.2.png
```

Saved images default to `.png`. Use `--extension` to choose another OpenCV-supported output format:

```powershell
python main.py "img.png" --output-dir "variants" --extension ".jpg"
```

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
