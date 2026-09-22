# Library Usage

## Supported Entry Point

Use `build_paddle_document_coordinator()` for normal local document extraction.
It constructs independent front and back pipelines from one checksum-verified
local PaddleOCR manifest.

```python
from pathlib import Path

from kyc_engine import build_paddle_document_coordinator

coordinator = build_paddle_document_coordinator(
    model_manifest=Path("private-models/paddleocr/ao-id-front-v5-mobile/manifest.json"),
    device="cpu",
)
result = coordinator.process(front=front_image, back=back_image)
```

`front_image` and `back_image` may be a `Path`, string path, encoded JPEG/PNG
bytes, or a `uint8` NumPy array. A side may be omitted; the result will then be
`partial` with a safe missing-side issue. `result.to_dict()` is the serializable
form and deliberately omits image pixels, paths, EXIF, and stack traces.

The default OpenCV detector is suitable for local development and profile
calibration only. Pass qualified `front_detector` and `back_detector` instances
to the factory when production detection models are available. `intake_limits`
can apply stricter source-image limits for a specific integration.

## Result Handling

`DocumentExtractionResult.status` is `success`, `partial`, or `failed`. Its
`front` and `back` values are preserved `KycExtractionResult` objects, including
side-specific fields, QR output, issues, and timings. The library does not
flatten or compare values across sides.

## Advanced Composition

`build_paddle_pipeline()` builds one labelled side pipeline. `build_balanced_pipeline()`
accepts a custom `TextRecognizer` and is intended for tests or controlled
internal composition. Concrete stages and adapters remain importable for
specialized integrations, but the coordinator factory is the supported default.
