from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from kyc_engine.minifasnet import (
    MiniFASNetAntiSpoofDetector,
    MiniFASNetInitializationError,
    MiniFASNetInputError,
    MiniFASNetNoFaceError,
    discover_models,
    parse_model_filename,
)
from kyc_engine.minifasnet.preprocessing import _scaled_box, crop_frame, tensor_values, validate_frame


class _FaceDetector:
    def __init__(self, box=(2, 1, 4, 4)) -> None:
        self.box = box
        self.calls = 0

    def detect(self, frame: np.ndarray):
        self.calls += 1
        return self.box


class _Predictor:
    def __init__(self, prediction: list[float]) -> None:
        self.prediction = np.asarray(prediction, dtype=np.float64)
        self.inputs: list[np.ndarray] = []

    def predict(self, frame: np.ndarray) -> np.ndarray:
        self.inputs.append(frame)
        return self.prediction


class MiniFASNetMetadataTests(unittest.TestCase):
    def test_parse_upstream_model_names(self) -> None:
        self.assertEqual((80, 80, "MiniFASNetV2", 2.7), parse_model_filename("2.7_80x80_MiniFASNetV2.pth"))
        self.assertEqual((80, 80, "MiniFASNetV1SE", 4.0), parse_model_filename("4_0_0_80x80_MiniFASNetV1SE.pth"))
        self.assertEqual((80, 80, "MiniFASNetV2", None), parse_model_filename("org_80x80_MiniFASNetV2.pth"))

    def test_unsupported_names_fail_safely(self) -> None:
        for name in ("bad.pth", "2_80x80_MiniFASNetV1.pth", "2_0x80_MiniFASNetV2.pth", "2_80x80_MiniFASNetV2.bin"):
            with self.subTest(name=name), self.assertRaises(MiniFASNetInitializationError):
                parse_model_filename(name)

    def test_discovery_is_sorted_and_requires_readable_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "4_0_0_80x80_MiniFASNetV1SE.pth").write_bytes(b"x")
            (directory / "2.7_80x80_MiniFASNetV2.pth").write_bytes(b"x")
            self.assertEqual(
                ["2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth"],
                [spec.path.name for spec in discover_models(directory)],
            )
        with self.assertRaises(MiniFASNetInitializationError):
            discover_models(Path("does-not-exist"))


class MiniFASNetPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)

    def test_scaled_crop_preserves_upstream_inclusive_boundaries(self) -> None:
        self.assertEqual((0, 0, 8, 8), _scaled_box(12, 10, (0, 0, 4, 4), 2.0))
        crop = crop_frame(self.frame, (0, 0, 4, 4), scale=2.0, output_width=8, output_height=8)
        self.assertEqual((8, 8, 3), crop.shape)
        self.assertTrue(np.array_equal(cv2.resize(self.frame[:9, :9], (8, 8)), crop))

    def test_original_crop_and_tensor_keep_bgr_raw_range(self) -> None:
        crop = crop_frame(self.frame, (2, 2, 4, 4), scale=None, output_width=4, output_height=3)
        self.assertEqual((3, 4, 3), crop.shape)
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)
        values = tensor_values(image)
        self.assertEqual((3, 1, 1), values.shape)
        self.assertEqual(np.float32, values.dtype)
        self.assertEqual([10.0, 20.0, 30.0], values[:, 0, 0].tolist())

    def test_invalid_frames_and_boxes_are_typed_errors(self) -> None:
        for invalid in (np.empty((0, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.float32)):
            with self.subTest(shape=invalid.shape), self.assertRaises(MiniFASNetInputError):
                validate_frame(invalid)
        with self.assertRaises(MiniFASNetNoFaceError):
            crop_frame(self.frame, (1, 1, 0, 3), scale=2.0, output_width=80, output_height=80)


class MiniFASNetAdapterTests(unittest.TestCase):
    def _model_root(self, temporary: str) -> Path:
        root = Path(temporary)
        models = root / "anti_spoof_models"
        detector = root / "detection_model"
        models.mkdir(); detector.mkdir()
        for name in ("2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth", "5_80x80_MiniFASNetV2.pth"):
            (models / name).write_bytes(b"model")
        (detector / "deploy.prototxt").write_text("model")
        (detector / "Widerface-RetinaFace.caffemodel").write_bytes(b"model")
        return root

    def test_models_load_once_and_average_actual_model_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            predictor_outputs = iter(([0.2, 0.1, 0.7], [0.1, 0.4, 0.5], [0.1, 0.2, 0.7]))
            loaded: list[tuple[str, _Predictor]] = []

            def loader(spec, device):
                self.assertEqual("cpu", device)
                predictor = _Predictor(next(predictor_outputs))
                loaded.append((spec.path.name, predictor))
                return predictor

            detector = MiniFASNetAntiSpoofDetector(self._model_root(temporary), _face_detector=_FaceDetector(), _model_loader=loader)
            frame = np.full((12, 12, 3), 123, dtype=np.uint8)
            first = detector.detect(frame)
            second = detector.detect(frame)

            self.assertEqual(3, len(loaded))
            self.assertEqual(["2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth", "5_80x80_MiniFASNetV2.pth"], [name for name, _ in loaded])
            self.assertFalse(first.is_real)
            self.assertAlmostEqual(7 / 30, first.score)
            self.assertEqual(first, second)
            self.assertTrue(all(len(predictor.inputs) == 2 for _, predictor in loaded))

    def test_invalid_resources_and_model_predictions_fail_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(MiniFASNetInitializationError):
                MiniFASNetAntiSpoofDetector(Path(temporary))
            root = self._model_root(temporary)
            with self.assertRaises(MiniFASNetInputError):
                MiniFASNetAntiSpoofDetector(
                    root,
                    _face_detector=_FaceDetector(),
                    _model_loader=lambda spec, device: _Predictor([0.5, 0.5]),
                ).detect(np.ones((12, 12, 3), dtype=np.uint8))

    def test_model_loader_failure_does_not_leave_a_partial_detector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(MiniFASNetInitializationError):
                MiniFASNetAntiSpoofDetector(
                    self._model_root(temporary),
                    _face_detector=_FaceDetector(),
                    _model_loader=lambda spec, device: (_ for _ in ()).throw(
                        MiniFASNetInitializationError("invalid checkpoint")
                    ),
                )


if __name__ == "__main__":
    unittest.main()
