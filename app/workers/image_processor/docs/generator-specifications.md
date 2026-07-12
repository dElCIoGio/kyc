# Generator Specifications

## Shared Contract

Every generator implements the same public contract:

```python
generate(
    image: Image,
    parameters: Mapping[str, object] | None = None,
) -> VariantBatch
```

Each generator returns exactly one `VariantBatch`. If a generator is configured with multiple scale values, it returns one `VariantInfo` per value.

Shared output requirements:

- Output images must be `np.uint8`.
- Output images must be valid according to `validate_image`.
- The source image must not be mutated.
- Variant names should be stable and include the applied parameter value when useful.
- Runtime `parameters` override constructor defaults for the current call only.

## Original

- Generator name: `original`
- Parameters: none
- Defaults: none
- Variants produced: `1`
- Output shape/channels: same as input
- OpenCV usage: none required; use `image.copy()`

Expected variant:

```python
VariantInfo(
    name="original",
    image=image.copy(),
    parameters={},
)
```

## Grayscale

- Generator name: `grayscale`
- Parameters: none for the initial implementation
- Defaults: none
- Variants produced: `1`
- Output shape/channels: 2D grayscale image
- OpenCV usage: `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` for 3-channel input

For 2D input, return a copy of the input. For 4-channel input, convert with the appropriate BGRA-to-gray conversion.

## CLAHE

- Generator name: `clahe`
- Parameters:
  - `clip_limits: Sequence[float]`
  - `tile_grid_sizes: Sequence[tuple[int, int]]`
- Defaults:
  - `clip_limits=(2.0,)`
  - `tile_grid_sizes=((8, 8),)`
- Variants produced: one per clip-limit and tile-grid-size combination
- Output shape/channels: 2D grayscale image
- OpenCV usage:
  - `cv2.cvtColor`
  - `cv2.createCLAHE(clipLimit=..., tileGridSize=...)`
  - `clahe.apply(gray_image)`

Variant names should include both values, for example `clahe_clip_2.0_grid_8x8`.

## Gamma Correction

- Generator name: `gamma`
- Parameters:
  - `gamma_values: Sequence[float]`
- Defaults:
  - `gamma_values=(0.8, 1.0, 1.2)`
- Variants produced: one per gamma value
- Output shape/channels: same as input
- OpenCV usage:
  - Lookup table with `np.array`
  - `cv2.LUT(image, table)`

Example configuration:

```python
GammaVariantGenerator(
    gamma_values=(0.8, 1.0, 1.2)
)
```

Expected batch shape:

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

Runtime override example:

```python
parameters={"gamma_values": [0.7, 1.2]}
```

## Contrast And Brightness

- Generator name: `contrast_brightness`
- Parameters:
  - `adjustments: Sequence[dict[str, float | int]]`
  - Each item contains `alpha` for contrast and `beta` for brightness
- Defaults:
  - `adjustments=({"alpha": 1.0, "beta": 0}, {"alpha": 1.2, "beta": 10})`
- Variants produced: one per adjustment
- Output shape/channels: same as input
- OpenCV usage: `cv2.convertScaleAbs(image, alpha=..., beta=...)`

Variant names should include both values, for example `contrast_1.2_brightness_10`.

## Sharpening

- Generator name: `sharpening`
- Parameters:
  - `strengths: Sequence[float]`
- Defaults:
  - `strengths=(1.0,)`
- Variants produced: one per strength
- Output shape/channels: same as input
- OpenCV usage:
  - `cv2.GaussianBlur`
  - `cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)`

Strength values must be non-negative. Variant names should include the strength, for example `sharpen_1.0`.

## Denoising

- Generator name: `denoising`
- Parameters:
  - `h_values: Sequence[float]`
- Defaults:
  - `h_values=(10.0,)`
- Variants produced: one per `h` value
- Output shape/channels: same as input
- OpenCV usage:
  - `cv2.fastNlMeansDenoising` for grayscale images
  - `cv2.fastNlMeansDenoisingColored` for BGR/BGRA images after channel handling

Variant names should include the denoising strength, for example `denoise_h_10.0`.

## Threshold

- Generator name: `threshold`
- Parameters:
  - `methods: Sequence[str]`
  - `adaptive_block_sizes: Sequence[int]`
  - `adaptive_c_values: Sequence[int | float]`
- Defaults:
  - `methods=("otsu", "adaptive_mean", "adaptive_gaussian")`
  - `adaptive_block_sizes=(11,)`
  - `adaptive_c_values=(2,)`
- Variants produced:
  - one for `otsu`
  - one per adaptive method, block size, and `C` value combination
- Output shape/channels: 2D binary grayscale image
- OpenCV usage:
  - `cv2.cvtColor`
  - `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`
  - `cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_MEAN_C, ...)`
  - `cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, ...)`

Supported method names:

- `otsu`
- `adaptive_mean`
- `adaptive_gaussian`

Adaptive block sizes must be odd integers greater than `1`. Variant names should be stable, for example:

- `threshold_otsu`
- `threshold_adaptive_mean_block_11_c_2`
- `threshold_adaptive_gaussian_block_11_c_2`
