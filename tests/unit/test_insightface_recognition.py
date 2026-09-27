from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np

from kyc_engine import (
    FaceRecognitionInferenceError,
    FaceRecognitionMultipleFacesError,
    FaceRecognitionNoFaceError,
    InsightFaceRecognizer,
)


class _Face:
    def __init__(self, embedding: object | None = None, score: float = 0.8) -> None:
        self.embedding = embedding
        self.det_score = score


class _Analysis:
    def __init__(self, faces: list[_Face]) -> None:
        self.models = {"detection": object(), "recognition": object()}
        self.faces = faces
        self.prepared = False
        self.images: list[np.ndarray] = []

    def prepare(self, *, ctx_id: int) -> None:
        self.prepared = ctx_id == 0

    def get(self, image: np.ndarray) -> list[_Face]:
        self.images.append(image)
        return self.faces


class InsightFaceRecognizerTests(unittest.TestCase):
    def _package(self, temporary: str, model_id: str = "development") -> Path:
        model_dir = Path(temporary) / "models" / model_id
        model_dir.mkdir(parents=True)
        (model_dir / "det.onnx").write_bytes(b"model")
        return Path(temporary)

    def test_uses_one_faceanalysis_result_for_detection_and_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            analysis = _Analysis([_Face(np.array([1.0, 2.0]))])
            recognizer = InsightFaceRecognizer(
                self._package(temporary),
                "development",
                _analysis_factory=lambda **kwargs: self._assert_cpu_factory_args(kwargs, analysis),
            )
            source = np.zeros((8, 8, 3), dtype=np.uint8)
            source.setflags(write=False)

            result = recognizer.encode(source)

        self.assertTrue(analysis.prepared)
        self.assertEqual(1, result.face_count)
        self.assertEqual(0.8, result.detection_confidence)
        np.testing.assert_array_equal(np.array([1.0, 2.0]), result.embedding.vector)
        self.assertEqual(1, len(analysis.images))
        self.assertTrue(analysis.images[0].flags.writeable)
        self.assertIsNot(analysis.images[0], source)

    def test_zero_or_multiple_faces_are_rejected_without_embedding_selection(self) -> None:
        for faces, error in (([], FaceRecognitionNoFaceError), ([_Face([1]), _Face([2])], FaceRecognitionMultipleFacesError)):
            with self.subTest(face_count=len(faces)), tempfile.TemporaryDirectory() as temporary:
                recognizer = InsightFaceRecognizer(
                    self._package(temporary),
                    "development",
                    _analysis_factory=lambda **_kwargs: _Analysis(faces),
                )
                with self.assertRaises(error):
                    recognizer.encode(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_missing_embedding_is_an_inference_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recognizer = InsightFaceRecognizer(
                self._package(temporary),
                "development",
                _analysis_factory=lambda **_kwargs: _Analysis([_Face()]),
            )
            with self.assertRaises(FaceRecognitionInferenceError):
                recognizer.encode(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_missing_package_fails_before_loader_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(Exception, "model package"):
                InsightFaceRecognizer(
                    Path(temporary), "development", _analysis_factory=lambda **_kwargs: self.fail("loader called")
                )

    def test_default_loader_disables_automatic_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._package(temporary)
            module = ModuleType("face_analysis")
            download_calls: list[object] = []

            def original_download(*_args, **_kwargs):
                download_calls.append("download")
                raise AssertionError("automatic download attempted")

            module.ensure_available = original_download
            analysis = _Analysis([_Face([1.0])])

            def factory(**kwargs):
                self.assertEqual(str(root / "models" / "development"), module.ensure_available("models", "development", root=kwargs["root"]))
                return analysis

            from unittest.mock import patch

            with patch("kyc_engine.insightface_recognition._load_face_analysis", return_value=(factory, module)):
                recognizer = InsightFaceRecognizer(root, "development")

        self.assertIsNotNone(recognizer)
        self.assertEqual([], download_calls)

    def _assert_cpu_factory_args(self, kwargs: dict[str, object], analysis: _Analysis) -> _Analysis:
        self.assertEqual(("detection", "recognition"), kwargs["allowed_modules"])
        self.assertEqual((), kwargs["addons"])
        self.assertEqual(["CPUExecutionProvider"], kwargs["providers"])
        return analysis


if __name__ == "__main__":
    unittest.main()
