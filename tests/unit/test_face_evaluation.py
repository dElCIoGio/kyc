from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kyc_engine import FaceEmbedding, FaceRecognitionResult
from kyc_engine.face_evaluation import (
    CsvFaceEvaluationDatasetLoader,
    FaceEvaluationDatasetError,
    FaceEvaluationSample,
    FaceEvaluationService,
)


def _image(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


class _Recognizer:
    provider_name = "synthetic"
    model_id = "synthetic-development"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, image: np.ndarray) -> FaceRecognitionResult:
        self.calls += 1
        vectors = {
            1: [1.0, 0.0],
            2: [1.0, 0.0],
            3: [0.0, 1.0],
            4: [0.0, 1.0],
        }
        return FaceRecognitionResult(FaceEmbedding(np.asarray(vectors[int(image[0, 0, 0])])), 1)


class _Intake:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def load(self, path: Path):
        self.paths.append(path)
        return SimpleNamespace(image=_image(len(self.paths)))


class FaceEvaluationTests(unittest.TestCase):
    def test_evaluation_reuses_runtime_recognizer_and_raw_cosine_primitive(self) -> None:
        recognizer = _Recognizer()
        report = FaceEvaluationService(recognizer).evaluate_samples(
            (
                FaceEvaluationSample("subject-a", _image(1), _image(2)),
                FaceEvaluationSample("subject-b", _image(3), _image(4)),
            )
        )

        self.assertEqual(4, recognizer.calls)
        self.assertEqual(2, report.genuine.count)
        self.assertEqual(2, report.impostor.count)
        self.assertEqual(1.0, dict(report.genuine.quantiles)[0.5])
        self.assertEqual(0.0, dict(report.impostor.quantiles)[0.5])
        payload = report.to_dict()
        self.assertEqual({"genuine", "impostor", "error_rates"}, set(payload))
        self.assertNotIn("embedding", repr(payload).lower())
        self.assertNotIn("subject", repr(payload).lower())
        self.assertNotIn("review", repr(payload).lower())
        self.assertNotIn("no_match", repr(payload).lower())

    def test_dataset_loader_reads_local_manifest_without_session_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.csv").write_text(
                "subject_id,reference_path,probe_path\nopaque-a,reference/a.jpg,probe/a.jpg\n",
                encoding="utf-8",
            )
            intake = _Intake()

            samples = CsvFaceEvaluationDatasetLoader(root, intake=intake).load()

        self.assertEqual(1, len(samples))
        self.assertEqual("opaque-a", samples[0].subject_label)
        self.assertEqual(2, len(intake.paths))
        self.assertTrue(all(path.is_relative_to(root) for path in intake.paths))

    def test_dataset_loader_rejects_missing_labels_and_path_escape(self) -> None:
        cases = (
            "subject_id,reference_path,probe_path\n,reference/a.jpg,probe/a.jpg\n",
            "subject_id,reference_path,probe_path\na,../reference/a.jpg,probe/a.jpg\n",
        )
        for manifest in cases:
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "manifest.csv").write_text(manifest, encoding="utf-8")
                with self.assertRaises(FaceEvaluationDatasetError):
                    CsvFaceEvaluationDatasetLoader(root, intake=_Intake()).load()

    def test_evaluation_requires_two_subjects(self) -> None:
        with self.assertRaises(FaceEvaluationDatasetError):
            FaceEvaluationService(_Recognizer()).evaluate_samples(
                (FaceEvaluationSample("only", _image(1), _image(2)),)
            )


if __name__ == "__main__":
    unittest.main()
