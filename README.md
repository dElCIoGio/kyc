# Angolan ID Core Engine

This repository contains an offline Python 3.12 core for detecting one Angolan identity-card side, normalizing its perspective, reading configured text fields, and returning a structured result with complete candidate provenance.

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
- local QR extraction and structured back-side QR payload parsing;
- deterministic reconciliation, conservative normalization, and validation;
- a synchronous `KycPipeline` returning immutable, serializable contracts.

The checked-in `ao_id_card/front/v1` and `ao_id_card/back/v1` profiles are **provisional**. Their geometry is calibrated only against private samples and must be reviewed against authoritative layout information before production use. ML weights and private accuracy data are deliberately outside Git. The installable library under `src/kyc_engine` is the sole supported implementation.

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

## Python Library

```python
from pathlib import Path

from kyc_engine import build_paddle_document_coordinator

coordinator = build_paddle_document_coordinator(
    model_manifest=Path("private-models/paddleocr/ao-id-front-v5-mobile/manifest.json"),
    device="cpu",
)
result = coordinator.process(
    front=Path("private-input/front.jpeg"),
    back=Path("private-input/back.jpeg"),
)
payload = result.to_dict()
```

`build_paddle_document_coordinator()` is the supported consumer entry point. It accepts paths, encoded bytes, or `uint8` NumPy arrays for either labelled side and returns a nested `DocumentExtractionResult`.

The default OpenCV detector is development-grade. Supply qualified front and back detector instances explicitly before treating the result as production-grade. The lower-level `build_balanced_pipeline()` and `build_paddle_pipeline()` functions remain available for advanced composition and tests; they validate local OCR manifests and never download models.

To process both private card images with the configured document coordinator, place
the front image at `private-data/ao-id-front/front.jpeg` and the back image at
`private-data/ao-id-back/back.jpeg`, then run:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_document_coordinator.py
```

The structured result is written to the ignored
`private-data/document-coordinator-result.json`. Override the defaults with
`--front`, `--back`, `--model-manifest`, or `--output` when needed.

## FastAPI Service

The separate [`api`](api/README.md) project installs this library and exposes a
session-based HTTP workflow for uploading labelled card sides, starting an
in-memory extraction job, polling status, and retrieving the nested coordinator
result. It uses API-key authentication and does not persist uploads or results.

```powershell
python -m pip install -e ".[ocr]"
python -m pip install -e "api[dev]"
$env:KYC_API_KEY = "replace-with-a-long-random-secret"
$env:KYC_OCR_MODEL_MANIFEST = "private-models\paddleocr\ao-id-front-v5-mobile\manifest.json"
uvicorn kyc_api.main:app --app-dir api/src --host 127.0.0.1 --port 8000 --no-access-log
```

## Data Safety

Only synthetic or irreversibly redacted identity-document fixtures may enter Git. Real cards, OCR output, model weights, generated variants, logs containing field values, and private evaluation datasets must stay outside the repository. Serialized results exclude source pixels, crops, paths, EXIF, and stack traces.

## Documentation

- [Architecture](docs/architecture.md)
- [Library usage](docs/library-usage.md)
- [Stage contracts](docs/contracts.md)
- [Security and data handling](docs/security-and-data-handling.md)
- [Development roadmap](docs/development-roadmap.md)
- [Testing strategy](docs/testing-strategy.md)
- [Private PaddleOCR calibration](docs/ocr-calibration.md)
- [ONNX detector contract](docs/onnx-detector-contract.md)
- [Architecture decisions](docs/decisions/README.md)
