# Image Processor Worker

This legacy compatibility module generates preprocessing variants for one already loaded image and measures objective quality properties for every variant. New core development lives in [`src/kyc_engine`](../../../src/kyc_engine); this module remains while existing imports and its CLI migrate. It does not detect an identity card, correct perspective, extract fields, rank variants, or establish document authenticity.

The current flow is:

```text
image_path -> cv2.imread(path) -> validate_image(image) -> VariantGenerationContext
           -> VariantBatch[] -> VariantQualityAssessmentPipeline
           -> VariantBatchAssessment[]
```

The image is read once with `cv2.imread(path)`, then passed as an in-memory OpenCV/Numpy image to a configurable set of variant generators. Each generator follows the strategy pattern and returns one strongly typed `VariantBatch` containing all variants produced by that strategy.

## Current State

- `main.py` can generate all default variants and optionally save them to disk.
- `validator.py` validates `np.ndarray` images through the shared `Image` type alias.
- `pyproject.toml` declares `numpy` and `opencv-python`.
- The default configuration currently produces 25 variants across eight generator batches and applies five quality assessors to each variant.

Generated variants are local diagnostics and are ignored by Git. Do not use real identity-card photographs as committed examples.

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
- [Repository architecture](../../../docs/architecture.md)
- [Security and data handling](../../../docs/security-and-data-handling.md)

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
