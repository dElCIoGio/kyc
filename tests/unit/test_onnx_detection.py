import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kyc_engine.detection import DetectionError
from kyc_engine.onnx_detection import OnnxDetectorManifest, OnnxDocumentDetector


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_inputs(self):
        return (SimpleNamespace(name="images"),)

    def run(self, output_names, inputs):
        self.last_input = inputs["images"]
        return (self.output,)


class OnnxDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.box_path = self.root / "box.onnx"
        self.corner_path = self.root / "corner.onnx"
        self.box_path.write_bytes(b"box-model")
        self.corner_path.write_bytes(b"corner-model")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loads_manifest_and_verifies_models(self) -> None:
        manifest_path = self._write_manifest()
        manifest = OnnxDetectorManifest.from_json(manifest_path)
        manifest.verify_files()
        self.assertEqual((200, 200), manifest.box_input_size)

        self.box_path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "checksum"):
            manifest.verify_files()

    def test_detects_one_box_and_maps_semantic_corners(self) -> None:
        manifest = OnnxDetectorManifest.from_json(self._write_manifest())
        sessions = {
            self.box_path.resolve(): FakeSession(
                np.asarray([[[20, 60, 180, 140, 0.92]]], dtype=np.float32)
            ),
            self.corner_path.resolve(): FakeSession(
                np.asarray(
                    [[[0.0, 0.0, 0.95], [1.0, 0.0, 0.96], [1.0, 1.0, 0.94], [0.0, 1.0, 0.93]]],
                    dtype=np.float32,
                )
            ),
        }
        detector = OnnxDocumentDetector(
            manifest,
            session_factory=lambda path, providers: sessions[path],
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = detector.detect(image)
        coordinates = tuple((round(point.x), round(point.y)) for point in result.corners.points)
        self.assertEqual(((20, 10), (179, 10), (179, 89), (20, 89)), coordinates)
        self.assertEqual("test-model-1", result.detector_version)
        self.assertAlmostEqual(0.92, result.confidence, places=5)

    def test_rejects_multiple_plausible_boxes(self) -> None:
        manifest = OnnxDetectorManifest.from_json(self._write_manifest())
        sessions = {
            self.box_path.resolve(): FakeSession(
                np.asarray(
                    [[[10, 55, 90, 145, 0.9], [110, 55, 190, 145, 0.85]]],
                    dtype=np.float32,
                )
            ),
            self.corner_path.resolve(): FakeSession(np.zeros((1, 4, 3), dtype=np.float32)),
        }
        detector = OnnxDocumentDetector(
            manifest,
            session_factory=lambda path, providers: sessions[path],
        )
        with self.assertRaises(DetectionError) as raised:
            detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        self.assertEqual("MULTIPLE_DOCUMENTS", raised.exception.code)

    def _write_manifest(self) -> Path:
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "version": "test-model-1",
                    "document_type": "ao_id_card",
                    "side": "front",
                    "box_model": {
                        "path": self.box_path.name,
                        "sha256": hashlib.sha256(self.box_path.read_bytes()).hexdigest(),
                        "input_size": [200, 200],
                    },
                    "corner_model": {
                        "path": self.corner_path.name,
                        "sha256": hashlib.sha256(self.corner_path.read_bytes()).hexdigest(),
                        "input_size": [128, 128],
                    },
                    "thresholds": {
                        "box_score": 0.5,
                        "nms_iou": 0.4,
                        "corner_score": 0.5,
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest_path


if __name__ == "__main__":
    unittest.main()
