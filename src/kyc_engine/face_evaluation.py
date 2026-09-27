"""Offline-only exploratory face-similarity evaluation.

This module intentionally has no dependency on the API, SessionStore, KYC
state, or artifact store. It consumes an explicitly labelled local dataset and
returns aggregate distributions only; embeddings and per-pair scores stay in
memory for the duration of one evaluation call.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .contracts import FaceRecognizer, Image
from .face_comparison import FaceComparisonError, compare_face_embeddings
from .face_recognition import FaceRecognitionError
from .intake import ImageIntake


_QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
_HISTOGRAM_EDGES = tuple(float(value) for value in np.linspace(-1.0, 1.0, 21))
_CURVE_THRESHOLDS = tuple(float(value) for value in np.linspace(-1.0, 1.0, 41))


class FaceEvaluationError(RuntimeError):
    """Base error for offline face-evaluation inputs and inference."""


class FaceEvaluationDatasetError(FaceEvaluationError):
    """The local, consented evaluation dataset is malformed or insufficient."""


class FaceEvaluationInferenceError(FaceEvaluationError):
    """A recognizer could not produce one safe embedding for evaluation."""


@dataclass(frozen=True)
class FaceEvaluationSample:
    """One labelled reference/probe pair loaded only for offline evaluation."""

    subject_label: str
    reference: Image
    probe: Image

    def __post_init__(self) -> None:
        if not isinstance(self.subject_label, str) or not self.subject_label.strip():
            raise ValueError("Face evaluation subject label is required")


class FaceEvaluationDatasetLoader(Protocol):
    """Loads consented, identity-labelled inputs without creating KYC sessions."""

    def load(self) -> tuple[FaceEvaluationSample, ...]: ...


class CsvFaceEvaluationDatasetLoader:
    """Load ``manifest.csv`` and image files located beneath one local directory.

    The CSV must have ``subject_id``, ``reference_path``, and ``probe_path``
    columns. Paths are relative to ``dataset_root`` and cannot escape it.
    Subject labels are used only to form genuine/impostor groups and never occur
    in the resulting aggregate report.
    """

    _REQUIRED_COLUMNS = frozenset({"subject_id", "reference_path", "probe_path"})

    def __init__(self, dataset_root: Path, *, intake: ImageIntake | None = None) -> None:
        self._dataset_root = Path(dataset_root).resolve()
        self._intake = intake or ImageIntake()

    def load(self) -> tuple[FaceEvaluationSample, ...]:
        manifest = self._dataset_root / "manifest.csv"
        if not self._dataset_root.is_dir() or not manifest.is_file():
            raise FaceEvaluationDatasetError("Face evaluation dataset manifest is unavailable")
        try:
            with manifest.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None or not self._REQUIRED_COLUMNS.issubset(reader.fieldnames):
                    raise FaceEvaluationDatasetError("Face evaluation dataset manifest is invalid")
                samples = tuple(self._read_row(row) for row in reader)
        except FaceEvaluationDatasetError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise FaceEvaluationDatasetError("Face evaluation dataset could not be loaded") from exc
        if not samples:
            raise FaceEvaluationDatasetError("Face evaluation dataset is empty")
        return samples

    def _read_row(self, row: dict[str, str | None]) -> FaceEvaluationSample:
        subject_label = (row.get("subject_id") or "").strip()
        reference_path = self._resolve_dataset_path(row.get("reference_path"))
        probe_path = self._resolve_dataset_path(row.get("probe_path"))
        if not subject_label:
            raise FaceEvaluationDatasetError("Face evaluation dataset manifest is invalid")
        try:
            return FaceEvaluationSample(
                subject_label=subject_label,
                reference=self._intake.load(reference_path).image,
                probe=self._intake.load(probe_path).image,
            )
        except Exception as exc:
            raise FaceEvaluationDatasetError("Face evaluation dataset image is invalid") from exc

    def _resolve_dataset_path(self, value: str | None) -> Path:
        if not value or Path(value).is_absolute():
            raise FaceEvaluationDatasetError("Face evaluation dataset manifest is invalid")
        path = (self._dataset_root / value).resolve()
        if not path.is_relative_to(self._dataset_root):
            raise FaceEvaluationDatasetError("Face evaluation dataset manifest is invalid")
        return path


@dataclass(frozen=True)
class FaceEvaluationDistribution:
    """Aggregate score distribution with no subject, image, or embedding data."""

    count: int
    quantiles: tuple[tuple[float, float], ...]
    histogram_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "quantiles": {str(probability): value for probability, value in self.quantiles},
            "histogram": {
                "edges": list(self.histogram_edges),
                "counts": list(self.histogram_counts),
            },
        }


@dataclass(frozen=True)
class FaceEvaluationErrorRate:
    """Exploratory error rates at one raw-similarity value, not a decision rule."""

    similarity_value: float
    false_match_rate: float
    false_non_match_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "similarity_value": self.similarity_value,
            "false_match_rate": self.false_match_rate,
            "false_non_match_rate": self.false_non_match_rate,
        }


@dataclass(frozen=True)
class FaceEvaluationReport:
    """Aggregate exploratory report; it carries no pair-level biometric data."""

    genuine: FaceEvaluationDistribution
    impostor: FaceEvaluationDistribution
    error_rates: tuple[FaceEvaluationErrorRate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "genuine": self.genuine.to_dict(),
            "impostor": self.impostor.to_dict(),
            "error_rates": [item.to_dict() for item in self.error_rates],
        }


class FaceEvaluationService:
    """Compare an offline labelled dataset using the runtime recognizer/primitives."""

    def __init__(self, recognizer: FaceRecognizer) -> None:
        self._recognizer = recognizer

    def evaluate(self, loader: FaceEvaluationDatasetLoader) -> FaceEvaluationReport:
        return self.evaluate_samples(loader.load())

    def evaluate_samples(self, samples: tuple[FaceEvaluationSample, ...]) -> FaceEvaluationReport:
        samples = tuple(samples)
        if len({sample.subject_label for sample in samples}) < 2:
            raise FaceEvaluationDatasetError(
                "Face evaluation requires at least two labelled subjects for impostor comparisons"
            )

        encoded = tuple(
            (sample.subject_label, self._encode(sample.reference), self._encode(sample.probe))
            for sample in samples
        )
        genuine_scores: list[float] = []
        impostor_scores: list[float] = []
        for reference_label, reference_embedding, _ in encoded:
            for probe_label, _, probe_embedding in encoded:
                try:
                    similarity = compare_face_embeddings(reference_embedding, probe_embedding).similarity
                except FaceComparisonError as exc:
                    raise FaceEvaluationInferenceError("Face evaluation embedding comparison failed") from exc
                if reference_label == probe_label:
                    genuine_scores.append(similarity)
                else:
                    impostor_scores.append(similarity)
        if not genuine_scores or not impostor_scores:
            raise FaceEvaluationDatasetError("Face evaluation did not produce both score populations")
        return FaceEvaluationReport(
            genuine=_distribution(genuine_scores),
            impostor=_distribution(impostor_scores),
            error_rates=_error_rates(genuine_scores, impostor_scores),
        )

    def _encode(self, image: Image):
        try:
            return self._recognizer.encode(image).embedding
        except FaceRecognitionError as exc:
            raise FaceEvaluationInferenceError("Face evaluation recognition failed") from exc
        except Exception as exc:
            raise FaceEvaluationInferenceError("Face evaluation recognition failed") from exc


def _distribution(scores: list[float]) -> FaceEvaluationDistribution:
    values = np.asarray(scores, dtype=np.float64)
    quantile_values = np.quantile(values, _QUANTILES)
    histogram_counts, histogram_edges = np.histogram(values, bins=_HISTOGRAM_EDGES)
    return FaceEvaluationDistribution(
        count=int(values.size),
        quantiles=tuple((probability, float(value)) for probability, value in zip(_QUANTILES, quantile_values)),
        histogram_edges=tuple(float(value) for value in histogram_edges),
        histogram_counts=tuple(int(value) for value in histogram_counts),
    )


def _error_rates(
    genuine_scores: list[float], impostor_scores: list[float]
) -> tuple[FaceEvaluationErrorRate, ...]:
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    return tuple(
        FaceEvaluationErrorRate(
            similarity_value=value,
            false_match_rate=float(np.mean(impostor >= value)),
            false_non_match_rate=float(np.mean(genuine < value)),
        )
        for value in _CURVE_THRESHOLDS
    )
