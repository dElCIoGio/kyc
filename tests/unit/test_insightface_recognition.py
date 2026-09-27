from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kyc_engine import (
    FaceRecognitionInferenceError,
    FaceRecognitionInitializationError,
    FaceRecognitionMultipleFacesError,
    FaceRecognitionNoFaceError,
    InsightFaceRecognizer,
)


class _Face:
    def __init__(self, **attributes: object) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


class _DetectionModel:
    taskname = "detection"

    def __init__(self, rows: np.ndarray | None = None, landmarks: np.ndarray | None = None) -> None:
        self.rows = np.asarray(rows) if rows is not None else np.array([[1, 2, 7, 8, 0.8]], dtype=np.float32)
        self.landmarks = (
            np.asarray(landmarks)
            if landmarks is not None
            else np.array([[[2, 3], [6, 3], [4, 5], [3, 7], [5, 7]]], dtype=np.float32)
        )
        self.prepare_calls: list[tuple[int, tuple[int, int] | None]] = []
        self.images: list[np.ndarray] = []

    def prepare(self, *, ctx_id: int, input_size: tuple[int, int] | None = None) -> None:
        self.prepare_calls.append((ctx_id, input_size))

    def detect(self, image: np.ndarray, *, max_num: int) -> tuple[np.ndarray, np.ndarray]:
        self.images.append(image)
        assert max_num == 0
        return self.rows, self.landmarks


class _RecognitionModel:
    taskname = "recognition"

    def __init__(self, embedding: object | None = None) -> None:
        self.embedding = embedding if embedding is not None else np.array([1.0, 2.0])
        self.prepare_calls: list[tuple[int, tuple[int, int] | None]] = []
        self.faces: list[_Face] = []
        self.images: list[np.ndarray] = []

    def prepare(self, *, ctx_id: int, input_size: tuple[int, int] | None = None) -> None:
        self.prepare_calls.append((ctx_id, input_size))

    def get(self, image: np.ndarray, face: _Face) -> None:
        self.images.append(image)
        self.faces.append(face)
        if self.embedding is not None:
            face.embedding = self.embedding


class _OtherModel:
    taskname = "landmark_3d_68"


class InsightFaceRecognizerTests(unittest.TestCase):
    def _package(self, temporary: str, files: dict[str, str]) -> Path:
        model_dir = Path(temporary) / "models" / "development"
        model_dir.mkdir(parents=True)
        for name, role in files.items():
            (model_dir / name).write_text(role, encoding="ascii")
        return Path(temporary)

    def _loader(self, models: dict[str, object], calls: list[tuple[str, object]]):
        def load(path: str, *, providers: object):
            calls.append((path, providers))
            role = Path(path).read_text(encoding="ascii")
            if role == "broken":
                raise RuntimeError("unsupported unrelated model")
            return models[role]

        return load

    def test_discovers_tasks_from_local_models_and_preserves_detector_face_for_recognition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._package(
                temporary,
                {"not-a-role-name.onnx": "det", "anything.onnx": "rec", "extra.onnx": "other"},
            )
            detector = _DetectionModel()
            recognition = _RecognitionModel()
            calls: list[tuple[str, object]] = []
            recognizer = InsightFaceRecognizer(
                root,
                "development",
                _model_loader=self._loader(
                    {"det": detector, "rec": recognition, "other": _OtherModel()}, calls
                ),
                _face_factory=_Face,
            )
            source = np.zeros((8, 8, 3), dtype=np.uint8)
            source.setflags(write=False)

            result = recognizer.encode(source)

        self.assertEqual(3, len(calls))
        self.assertTrue(all(providers == ["CPUExecutionProvider"] for _, providers in calls))
        self.assertEqual([(-1, (640, 640))], detector.prepare_calls)
        self.assertEqual([(-1, None)], recognition.prepare_calls)
        self.assertEqual(1, result.face_count)
        self.assertAlmostEqual(0.8, result.detection_confidence)
        np.testing.assert_array_equal(np.array([1.0, 2.0]), result.embedding.vector)
        self.assertEqual(1, len(detector.images))
        self.assertIs(detector.images[0], recognition.images[0])
        self.assertTrue(detector.images[0].flags.writeable)
        self.assertIsNot(detector.images[0], source)
        self.assertEqual(1, len(recognition.faces))
        np.testing.assert_array_equal(np.array([1, 2, 7, 8], dtype=np.float32), recognition.faces[0].bbox)
        np.testing.assert_array_equal(detector.landmarks[0], recognition.faces[0].kps)

    def test_unrelated_or_unloadable_models_do_not_block_required_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._package(
                temporary,
                {"a.onnx": "det", "b.onnx": "rec", "c.onnx": "other", "d.onnx": "broken"},
            )
            recognizer = InsightFaceRecognizer(
                root,
                "development",
                _model_loader=self._loader(
                    {"det": _DetectionModel(), "rec": _RecognitionModel(), "other": _OtherModel()}, []
                ),
                _face_factory=_Face,
            )

        self.assertIsNotNone(recognizer)

    def test_missing_or_ambiguous_required_roles_fail_startup(self) -> None:
        cases = (
            {"detector.onnx": "det", "other.onnx": "other"},
            {"one.onnx": "det", "two.onnx": "det", "rec.onnx": "rec"},
            {"det.onnx": "det", "one.onnx": "rec", "two.onnx": "rec"},
        )
        for files in cases:
            with self.subTest(files=files), tempfile.TemporaryDirectory() as temporary:
                root = self._package(temporary, files)
                with self.assertRaisesRegex(FaceRecognitionInitializationError, "exactly one usable"):
                    InsightFaceRecognizer(
                        root,
                        "development",
                        _model_loader=self._loader(
                            {"det": _DetectionModel(), "rec": _RecognitionModel(), "other": _OtherModel()}, []
                        ),
                        _face_factory=_Face,
                    )

    def test_zero_or_multiple_faces_are_rejected_before_recognition(self) -> None:
        cases = (
            (np.empty((0, 5), dtype=np.float32), FaceRecognitionNoFaceError),
            (np.array([[1, 2, 7, 8, 0.8], [2, 3, 8, 9, 0.9]], dtype=np.float32), FaceRecognitionMultipleFacesError),
        )
        for rows, error in cases:
            with self.subTest(face_count=len(rows)), tempfile.TemporaryDirectory() as temporary:
                root = self._package(temporary, {"detector.onnx": "det", "recognizer.onnx": "rec"})
                recognition = _RecognitionModel()
                recognizer = InsightFaceRecognizer(
                    root,
                    "development",
                    _model_loader=self._loader(
                        {"det": _DetectionModel(rows=rows), "rec": recognition}, []
                    ),
                    _face_factory=_Face,
                )
                with self.assertRaises(error):
                    recognizer.encode(np.zeros((8, 8, 3), dtype=np.uint8))
                self.assertEqual([], recognition.faces)

    def test_missing_landmarks_or_embedding_is_an_inference_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._package(temporary, {"detector.onnx": "det", "recognizer.onnx": "rec"})
            missing_landmarks = InsightFaceRecognizer(
                root,
                "development",
                _model_loader=self._loader(
                    {"det": _DetectionModel(landmarks=None), "rec": _RecognitionModel()}, []
                ),
                _face_factory=_Face,
            )
            missing_landmarks._detector.landmarks = None
            with self.assertRaises(FaceRecognitionInferenceError):
                missing_landmarks.encode(np.zeros((8, 8, 3), dtype=np.uint8))

            recognition_without_embedding = _RecognitionModel()
            recognition_without_embedding.embedding = None
            missing_embedding = InsightFaceRecognizer(
                root,
                "development",
                _model_loader=self._loader(
                    {"det": _DetectionModel(), "rec": recognition_without_embedding}, []
                ),
                _face_factory=_Face,
            )
            with self.assertRaises(FaceRecognitionInferenceError):
                missing_embedding.encode(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_default_loader_uses_direct_model_zoo_api_without_download_hook(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._package(temporary, {"detector.onnx": "det", "recognizer.onnx": "rec"})
            calls: list[tuple[str, object]] = []
            with patch(
                "kyc_engine.insightface_recognition._load_insightface_apis",
                return_value=(
                    self._loader({"det": _DetectionModel(), "rec": _RecognitionModel()}, calls),
                    _Face,
                ),
            ) as local_apis:
                recognizer = InsightFaceRecognizer(root, "development")

        self.assertIsNotNone(recognizer)
        local_apis.assert_called_once_with()
        self.assertEqual(2, len(calls))

    def test_missing_package_fails_before_model_api_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FaceRecognitionInitializationError, "model package"):
                InsightFaceRecognizer(
                    Path(temporary), "development", _model_loader=lambda **_kwargs: self.fail("loader called"), _face_factory=_Face
                )


if __name__ == "__main__":
    unittest.main()
