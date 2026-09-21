# Private PaddleOCR Calibration

The first OCR baseline uses local PaddleOCR v5 mobile artifacts: `PP-OCRv5_mobile_det` for text detection and `latin_PP-OCRv5_mobile_rec` for Portuguese-compatible Latin recognition. Download and review those artifacts manually from the official PaddleOCR model sources. Do not run the engine with a missing model directory and do not let PaddleOCR fetch a default model at runtime.

Keep the models and this manifest outside Git:

```text
private-models/
  paddleocr/
    ao-id-front-v5-mobile/
      manifest.json
      det/
      rec/
```

`manifest.json` must use model-directory paths relative to itself:

```json
{
  "schema_version": "1",
  "model_version": "paddleocr-v5-mobile-pt-local-1",
  "language": "pt",
  "device": "cpu",
  "detection_model": {
    "path": "det",
    "model_name": "PP-OCRv5_mobile_det",
    "sha256": "<64-character directory SHA-256>"
  },
  "recognition_model": {
    "path": "rec",
    "model_name": "latin_PP-OCRv5_mobile_rec",
    "sha256": "<64-character directory SHA-256>"
  }
}
```

Calculate each digest after the directory has been finalized:

```powershell
$env:PYTHONPATH = "src"
python -c "from pathlib import Path; from kyc_engine.ocr import hash_model_directory; print(hash_model_directory(Path('private-models/paddleocr/ao-id-front-v5-mobile/det'))); print(hash_model_directory(Path('private-models/paddleocr/ao-id-front-v5-mobile/rec')))"
```

Run one private calibration image with an explicit private output path:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_ocr_calibration.py private-data/ao-id-front/front.jpeg --model-manifest private-models/paddleocr/ao-id-front-v5-mobile/manifest.json --output private-data/ao-id-front/ocr-calibration-result.json
```

The command prints only processing status, aggregate field-status counts, safe issue codes, and timings. The JSON contains sensitive recognized values and complete OCR candidate provenance, so keep it under `private-data/`, do not attach it to tickets or chat, and delete it according to the local retention policy.

On the current Windows/PaddlePaddle CPU runtime, the adapter disables oneDNN for the v5 detector because that backend raises a runtime `NotImplementedError`. This favors a working local baseline over oneDNN acceleration; benchmark the configured runtime before treating its latency as a release measurement.
