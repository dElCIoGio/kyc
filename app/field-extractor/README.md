# Field Extractor Prototype

This experimental module crops configured field regions from image variants and runs a pluggable OCR engine over each region. It assumes every input variant is already a perspective-corrected image of the same document with identical dimensions and coordinates.

This is the legacy field-extractor compatibility surface. New core development lives in [`src/kyc_engine`](../../src/kyc_engine), which now owns profile-based localization, OCR candidates, reconciliation, and validation. This module remains while existing consumers migrate and does not establish document authenticity or persist results.

## Current Components

- `FieldDefinition` and `BoundingBox` describe fixed field regions.
- `FieldExtractor` applies those regions to each supplied variant.
- `OCREngine` separates extraction orchestration from OCR implementation.
- `PaddleOCREngine` is the current local OCR adapter.
- `fields.json` contains provisional Angolan ID front coordinates and is not yet an authoritative versioned profile.

The module currently accepts paths, bytes, NumPy arrays, and Pillow images. Per-field failures are represented in extraction results so one failed field does not discard other successful reads.

## Local Use

```powershell
python main.py --images-dir .\variants --fields-config .\fields.json
```

Write OCR results only when explicitly needed:

```powershell
python main.py --images-dir .\variants --fields-config .\fields.json --output .\results.json
```

`results.json`, real document images, and generated variants are ignored by Git and must not be used as committed fixtures.

Run the fake-engine smoke tests without live OCR inference:

```powershell
python extractor_test.py
```

## Project Documentation

- [Architecture](../../docs/architecture.md)
- [Stage contracts](../../docs/contracts.md)
- [Security and data handling](../../docs/security-and-data-handling.md)
- [Development roadmap](../../docs/development-roadmap.md)
- [Testing strategy](../../docs/testing-strategy.md)
