# Angolan ID Core Engine

This repository contains an offline Python 3.12 core for detecting one front-side Angolan identity card, normalizing its perspective, reading configured text fields, and returning a structured result with complete candidate provenance.

```text
secure intake -> card detection -> perspective normalization -> variants
-> quality gates -> field crops -> OCR candidates -> reconciliation
-> normalization and validation -> structured result
```

The engine does not determine authenticity, perform liveness or biometric matching, persist identity data, or expose an HTTP API or frontend.

## Current State

The canonical implementation is in `src/kyc_engine` and provides:

- bounded JPEG/PNG and NumPy intake with metadata removal;
- a development OpenCV quadrilateral detector;
- a checksum-verified two-model ONNX detector boundary;
- perspective normalization with forward and inverse transforms;
- a balanced ten-variant OCR policy and five quality assessors;
- profile-driven field crops and a local PaddleOCR adapter;
- deterministic reconciliation, conservative normalization, and validation;
- a synchronous `KycPipeline` returning immutable, serializable contracts.

The checked-in `ao_id_card/front/v1` profile is **provisional**. It currently targets the visible front-side textual fields in the available card sample, but its geometry must still be adjusted and signed off against authoritative layout information before production use. ML weights and private accuracy data are deliberately outside Git. The older modules under `app/` remain available while consumers migrate to the root package.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Install optional local inference dependencies only where needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ocr,detection]"
```

The OCR and ONNX adapters require explicit local model paths and validate configured checksums. Processing never downloads a model.

## Python Use

```python
from pathlib import Path

from kyc_engine.defaults import build_paddle_pipeline

pipeline = build_paddle_pipeline(
    model_manifest=Path("private-models/paddleocr/ao-id-front-v5-mobile/manifest.json"),
)
result = pipeline.process(Path("private-input/card.png"))
payload = result.to_dict()
```

`build_balanced_pipeline()` accepts a fake or alternative `TextRecognizer` for tests. `build_paddle_pipeline()` validates a local OCR artifact manifest before initializing PaddleOCR and never downloads models. An `OnnxDocumentDetector` can be passed explicitly after loading its manifest. The OpenCV detector is the development default and must not be treated as production-qualified.

## Data Safety

Only synthetic or irreversibly redacted identity-document fixtures may enter Git. Real cards, OCR output, model weights, generated variants, logs containing field values, and private evaluation datasets must stay outside the repository. Serialized results exclude source pixels, crops, paths, EXIF, and stack traces.

## Documentation

- [Architecture](docs/architecture.md)
- [Stage contracts](docs/contracts.md)
- [Security and data handling](docs/security-and-data-handling.md)
- [Development roadmap](docs/development-roadmap.md)
- [Testing strategy](docs/testing-strategy.md)
- [Private PaddleOCR calibration](docs/ocr-calibration.md)
- [ONNX detector contract](docs/onnx-detector-contract.md)
- [Architecture decisions](docs/decisions/README.md)
