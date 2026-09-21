from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from kyc_engine.contracts import (
    BoundingBox,
    ExtractedField,
    FieldStatus,
    KycExtractionResult,
    OCRCandidate,
    ProcessingStatus,
)
from kyc_engine.ocr import PaddleOcrModelManifest, hash_model_directory
from scripts import run_ocr_calibration


class OcrManifestTests(unittest.TestCase):
    def test_loads_valid_relative_model_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detection = self._model_dir(root, "det", b"detection")
            recognition = self._model_dir(root, "rec", b"recognition")
            manifest_path = self._write_manifest(root, detection, recognition)

            manifest = PaddleOcrModelManifest.load(manifest_path)

            self.assertEqual(detection.resolve(), manifest.detection_model_dir)
            self.assertEqual(recognition.resolve(), manifest.recognition_model_dir)
            self.assertEqual("cpu", manifest.device)

    def test_rejects_malformed_or_invalid_manifest_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "readable JSON"):
                PaddleOcrModelManifest.load(manifest_path)

            detection = self._model_dir(root, "det", b"detection")
            recognition = self._model_dir(root, "rec", b"recognition")
            invalid_path = self._write_manifest(
                root,
                detection,
                recognition,
                device="tpu",
            )
            with self.assertRaisesRegex(ValueError, "device"):
                PaddleOcrModelManifest.load(invalid_path)

    def test_rejects_missing_models_bad_checksums_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detection = self._model_dir(root, "det", b"detection")
            recognition = self._model_dir(root, "rec", b"recognition")

            mismatch = self._write_manifest(
                root,
                detection,
                recognition,
                detection_sha256="0" * 64,
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                PaddleOcrModelManifest.load(mismatch)

            malformed = self._write_manifest(
                root,
                detection,
                recognition,
                recognition_sha256="not-a-sha",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                PaddleOcrModelManifest.load(malformed)

            escaped = self._write_manifest(
                root,
                detection,
                recognition,
                detection_path="../det",
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                PaddleOcrModelManifest.load(escaped)

            missing = self._write_manifest(
                root,
                detection,
                recognition,
                recognition_path="missing",
            )
            with self.assertRaisesRegex(ValueError, "directory"):
                PaddleOcrModelManifest.load(missing)

    @staticmethod
    def _model_dir(root: Path, name: str, content: bytes) -> Path:
        directory = root / name
        directory.mkdir()
        (directory / "model.bin").write_bytes(content)
        return directory

    @staticmethod
    def _write_manifest(
        root: Path,
        detection: Path,
        recognition: Path,
        *,
        detection_path: str = "det",
        recognition_path: str = "rec",
        detection_sha256: str | None = None,
        recognition_sha256: str | None = None,
        device: str = "cpu",
    ) -> Path:
        payload = {
            "schema_version": "1",
            "model_version": "test-v1",
            "language": "pt",
            "device": device,
            "detection_model": {
                "path": detection_path,
                "model_name": "PP-OCRv5_mobile_det",
                "sha256": detection_sha256 or hash_model_directory(detection),
            },
            "recognition_model": {
                "path": recognition_path,
                "model_name": "latin_PP-OCRv5_mobile_rec",
                "sha256": recognition_sha256 or hash_model_directory(recognition),
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path


class OcrCalibrationRunnerTests(unittest.TestCase):
    def test_writes_explicit_private_json_without_printing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private_root = root / "private-data"
            output = private_root / "result.json"
            stream = io.StringIO()

            with redirect_stdout(stream):
                exit_code = run_ocr_calibration.main(
                    [
                        "source.jpeg",
                        "--model-manifest",
                        "models/manifest.json",
                        "--output",
                        str(output),
                    ],
                    pipeline_factory=lambda **_: _FakePipeline(emit_backend_text=True),
                    private_root=private_root,
                )

            self.assertEqual(0, exit_code)
            output_text = output.read_text(encoding="utf-8")
            self.assertIn("PRIVATE OCR VALUE", output_text)
            self.assertNotIn("PRIVATE OCR VALUE", stream.getvalue())
            self.assertNotIn("BACKEND CHATTER", stream.getvalue())
            self.assertIn("status=success", stream.getvalue())

    def test_rejects_output_outside_private_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stream = io.StringIO()
            with redirect_stdout(stream):
                exit_code = run_ocr_calibration.main(
                    [
                        "source.jpeg",
                        "--model-manifest",
                        "models/manifest.json",
                        "--output",
                        str(root / "result.json"),
                    ],
                    pipeline_factory=lambda **_: _FakePipeline(),
                    private_root=root / "private-data",
                )
            self.assertEqual(2, exit_code)
            self.assertEqual("calibration_failed=INVALID_PRIVATE_OUTPUT\n", stream.getvalue())


class _FakePipeline:
    def __init__(self, *, emit_backend_text: bool = False) -> None:
        self.emit_backend_text = emit_backend_text

    def process(self, source: Path) -> KycExtractionResult:
        if self.emit_backend_text:
            print("BACKEND CHATTER")
        candidate = OCRCandidate(
            field_name="full_name",
            raw_value="PRIVATE OCR VALUE",
            confidence=0.9,
            generator="original",
            variant_name="original",
            bounding_box=BoundingBox(1, 1, 10, 10),
            ocr_engine="fake",
            ocr_model_version="1",
        )
        field = ExtractedField(
            name="full_name",
            status=FieldStatus.VALID,
            raw_value=candidate.raw_value,
            normalized_value=candidate.raw_value,
            confidence=candidate.confidence,
            selected_candidate=candidate,
        )
        return KycExtractionResult(
            schema_version="1.0",
            processing_id="test",
            status=ProcessingStatus.SUCCESS,
            document_type="ao_id_card",
            side="front",
            profile_id="ao_id_card/front/v1",
            detection=None,
            fields={"full_name": field},
            issues=(),
            timings_ms={"ocr": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
