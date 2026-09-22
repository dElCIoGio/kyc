from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cv2
import numpy as np

from .contracts import FieldCrop, OCRCandidate


logger = logging.getLogger(__name__)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_DEVICE_PATTERN = re.compile(r"^(cpu|gpu(?::[0-9]+)?)$")
_MANIFEST_VERSION = "1"
_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
_RECOGNITION_MODEL_NAME = "latin_PP-OCRv5_mobile_rec"


class TextRecognizer(Protocol):
    name: str
    model_version: str

    def recognize_batch(self, crops: Sequence[FieldCrop]) -> tuple[OCRCandidate, ...]:
        ...


@dataclass(frozen=True)
class PaddleOcrModelManifest:
    """Verified local artifact metadata for the supported PaddleOCR baseline."""

    model_version: str
    language: str
    device: str
    detection_model_dir: Path
    detection_model_sha256: str
    detection_model_name: str
    recognition_model_dir: Path
    recognition_model_sha256: str
    recognition_model_name: str
    schema_version: str = _MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _MANIFEST_VERSION:
            raise ValueError("Unsupported PaddleOCR manifest schema version")
        if not self.model_version.strip():
            raise ValueError("PaddleOCR model version must not be empty")
        if self.language != "pt":
            raise ValueError("PaddleOCR calibration supports Portuguese language only")
        _validate_device(self.device)
        if self.detection_model_name != _DETECTION_MODEL_NAME:
            raise ValueError("Unsupported PaddleOCR detection model")
        if self.recognition_model_name != _RECOGNITION_MODEL_NAME:
            raise ValueError("Unsupported PaddleOCR recognition model")
        _validate_sha256(self.detection_model_sha256, "detection model")
        _validate_sha256(self.recognition_model_sha256, "recognition model")

    @classmethod
    def load(cls, path: str | Path) -> PaddleOcrModelManifest:
        manifest_path = Path(path).resolve()
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError("PaddleOCR model manifest does not exist") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("PaddleOCR model manifest is not readable JSON") from exc
        if not isinstance(raw, Mapping):
            raise TypeError("PaddleOCR model manifest root must be an object")

        detection_dir, detection_sha256, detection_name = _load_model_entry(
            raw,
            "detection_model",
            manifest_path.parent,
        )
        recognition_dir, recognition_sha256, recognition_name = _load_model_entry(
            raw,
            "recognition_model",
            manifest_path.parent,
        )
        try:
            manifest = cls(
                schema_version=str(raw["schema_version"]),
                model_version=str(raw["model_version"]),
                language=str(raw.get("language", "pt")),
                device=str(raw.get("device", "cpu")),
                detection_model_dir=detection_dir,
                detection_model_sha256=detection_sha256,
                detection_model_name=detection_name,
                recognition_model_dir=recognition_dir,
                recognition_model_sha256=recognition_sha256,
                recognition_model_name=recognition_name,
            )
        except KeyError as exc:
            raise ValueError(f"PaddleOCR model manifest is missing '{exc.args[0]}'") from exc
        manifest.verify_files()
        return manifest

    def with_device(self, device: str | None) -> PaddleOcrModelManifest:
        if device is None:
            return self
        return PaddleOcrModelManifest(
            model_version=self.model_version,
            language=self.language,
            device=device,
            detection_model_dir=self.detection_model_dir,
            detection_model_sha256=self.detection_model_sha256,
            detection_model_name=self.detection_model_name,
            recognition_model_dir=self.recognition_model_dir,
            recognition_model_sha256=self.recognition_model_sha256,
            recognition_model_name=self.recognition_model_name,
            schema_version=self.schema_version,
        )

    def verify_files(self) -> None:
        _verify_model_directory(
            self.detection_model_dir,
            self.detection_model_sha256,
            "detection model",
        )
        _verify_model_directory(
            self.recognition_model_dir,
            self.recognition_model_sha256,
            "recognition model",
        )


class PaddleOCRTextRecognizer:
    name = "paddleocr"

    def __init__(
        self,
        *,
        detection_model_dir: str | Path,
        recognition_model_dir: str | Path,
        model_version: str,
        model_checksum: str,
        device: str = "cpu",
        language: str = "pt",
        detection_model_name: str = _DETECTION_MODEL_NAME,
        recognition_model_name: str = _RECOGNITION_MODEL_NAME,
    ) -> None:
        """Construct directly for isolated integration tests.

        Production and calibration callers should use :meth:`from_manifest` so
        each model directory is checked independently.
        """
        detection_path = Path(detection_model_dir)
        recognition_path = Path(recognition_model_dir)
        _validate_device(device)
        if language != "pt":
            raise ValueError("PaddleOCR calibration supports Portuguese language only")
        if detection_model_name != _DETECTION_MODEL_NAME:
            raise ValueError("Unsupported PaddleOCR detection model")
        if recognition_model_name != _RECOGNITION_MODEL_NAME:
            raise ValueError("Unsupported PaddleOCR recognition model")
        if not model_version.strip():
            raise ValueError("PaddleOCR model version must not be empty")
        _validate_sha256(model_checksum, "PaddleOCR model checksum")
        if not detection_path.is_dir() or not recognition_path.is_dir():
            raise ValueError("PaddleOCR model directories must exist locally")
        actual = hash_model_directories((detection_path, recognition_path))
        if actual.lower() != model_checksum.lower():
            raise ValueError("PaddleOCR model checksum does not match")
        self._initialize(
            detection_model_dir=detection_path,
            recognition_model_dir=recognition_path,
            model_version=model_version,
            device=device,
            language=language,
            detection_model_name=detection_model_name,
            recognition_model_name=recognition_model_name,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: PaddleOcrModelManifest,
        *,
        device: str | None = None,
    ) -> PaddleOCRTextRecognizer:
        configured = manifest.with_device(device)
        configured.verify_files()
        recognizer = cls.__new__(cls)
        recognizer._initialize(
            detection_model_dir=configured.detection_model_dir,
            recognition_model_dir=configured.recognition_model_dir,
            model_version=configured.model_version,
            device=configured.device,
            language=configured.language,
            detection_model_name=configured.detection_model_name,
            recognition_model_name=configured.recognition_model_name,
        )
        return recognizer

    def _initialize(
        self,
        *,
        detection_model_dir: Path,
        recognition_model_dir: Path,
        model_version: str,
        device: str,
        language: str,
        detection_model_name: str,
        recognition_model_name: str,
    ) -> None:
        try:
            from paddleocr import PaddleOCR, TextRecognition
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR support requires the project 'ocr' dependency extra"
            ) from exc

        self.model_version = model_version
        self._recognition = TextRecognition(
            model_name=recognition_model_name,
            model_dir=str(recognition_model_dir),
            device=device,
        )
        self._ocr = PaddleOCR(
            lang=language,
            device=device,
            text_detection_model_name=detection_model_name,
            text_detection_model_dir=str(detection_model_dir),
            text_recognition_model_name=recognition_model_name,
            text_recognition_model_dir=str(recognition_model_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # PaddlePaddle's oneDNN path cannot execute this v5 detector on Windows.
            enable_mkldnn=False,
        )

    def recognize_batch(self, crops: Sequence[FieldCrop]) -> tuple[OCRCandidate, ...]:
        candidates: list[OCRCandidate | None] = [None] * len(crops)
        single_line = [
            (index, crop) for index, crop in enumerate(crops) if crop.ocr_mode == "single_line"
        ]
        multiline = [
            (index, crop) for index, crop in enumerate(crops) if crop.ocr_mode == "multiline"
        ]
        unsupported = [
            (index, crop)
            for index, crop in enumerate(crops)
            if crop.ocr_mode not in {"single_line", "multiline"}
        ]

        self._recognize_with_retry(
            single_line,
            candidates,
            self._recognition.predict,
            _combine_recognition_result,
        )
        self._recognize_with_retry(
            multiline,
            candidates,
            self._ocr.predict,
            lambda page: _combine_pages((page,), multiline=True),
        )
        for index, crop in unsupported:
            candidates[index] = _candidate(
                crop,
                "",
                0.0,
                self.name,
                self.model_version,
                error_code="UNSUPPORTED_OCR_MODE",
            )
        if any(candidate is None for candidate in candidates):
            raise RuntimeError("PaddleOCR did not produce a candidate for every field crop")
        return tuple(candidate for candidate in candidates if candidate is not None)

    def _recognize_with_retry(
        self,
        indexed_crops: Sequence[tuple[int, FieldCrop]],
        candidates: list[OCRCandidate | None],
        predict: Callable[[list[Any]], Any],
        combine: Callable[[Any], tuple[str, float]],
    ) -> None:
        if not indexed_crops:
            return
        try:
            parsed = self._predict_and_combine(indexed_crops, predict, combine)
        except Exception as exc:
            logger.exception(
                "OCR batch prediction failed; retrying fields individually",
                extra={
                    "event": "ocr_batch_prediction_failed",
                    "stage": "ocr",
                    "exception_type": type(exc).__name__,
                },
            )
            for index, crop in indexed_crops:
                try:
                    text, confidence = self._predict_and_combine(
                        ((index, crop),), predict, combine
                    )[0]
                    candidates[index] = _candidate(
                        crop, text, confidence, self.name, self.model_version
                    )
                except Exception as exc:
                    logger.exception(
                        "OCR field prediction failed",
                        extra={
                            "event": "ocr_field_prediction_failed",
                            "stage": "ocr",
                            "field": crop.field_name,
                            "exception_type": type(exc).__name__,
                        },
                    )
                    candidates[index] = _failed_candidate(crop, self.name, self.model_version)
            return

        for (index, crop), (text, confidence) in zip(indexed_crops, parsed, strict=True):
            candidates[index] = _candidate(crop, text, confidence, self.name, self.model_version)

    @staticmethod
    def _predict_and_combine(
        indexed_crops: Sequence[tuple[int, FieldCrop]],
        predict: Callable[[list[Any]], Any],
        combine: Callable[[Any], tuple[str, float]],
    ) -> tuple[tuple[str, float], ...]:
        results = tuple(predict([_paddle_input(crop.image) for _, crop in indexed_crops]))
        if len(results) != len(indexed_crops):
            raise RuntimeError("PaddleOCR returned an unexpected result count")
        return tuple(combine(result) for result in results)


def hash_model_directory(path: Path) -> str:
    if not path.is_dir():
        raise ValueError("PaddleOCR model directory must exist locally")
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def hash_model_directories(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for root in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(root.name.encode("utf-8"))
        digest.update(hash_model_directory(root).encode("ascii"))
    return digest.hexdigest()


def _load_model_entry(
    raw: Mapping[str, Any],
    name: str,
    manifest_root: Path,
) -> tuple[Path, str, str]:
    entry = raw.get(name)
    if not isinstance(entry, Mapping):
        raise TypeError(f"PaddleOCR model manifest '{name}' must be an object")
    relative_path = entry.get("path")
    checksum = entry.get("sha256")
    model_name = entry.get("model_name")
    if not isinstance(relative_path, str) or not isinstance(checksum, str) or not isinstance(model_name, str):
        raise TypeError(
            f"PaddleOCR model manifest '{name}' path, sha256, and model_name must be strings"
        )
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        raise ValueError(f"PaddleOCR model manifest '{name}' path must be relative")
    resolved = (manifest_root / raw_path).resolve()
    if not resolved.is_relative_to(manifest_root.resolve()):
        raise ValueError(f"PaddleOCR model manifest '{name}' path escapes the manifest directory")
    return resolved, checksum, model_name


def _verify_model_directory(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_dir():
        raise ValueError(f"PaddleOCR {label} directory must exist locally")
    actual = hash_model_directory(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"PaddleOCR {label} checksum does not match manifest")


def _validate_sha256(value: str, label: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label.capitalize()} SHA-256 must be 64 hexadecimal characters")


def _validate_device(device: str) -> None:
    if not _DEVICE_PATTERN.fullmatch(device):
        raise ValueError("PaddleOCR device must be 'cpu', 'gpu', or 'gpu:N'")


def _combine_pages(pages: Sequence[Any], *, multiline: bool) -> tuple[str, float]:
    if not pages:
        return "", 0.0
    page = pages[0]
    texts = list(_result_field(page, "rec_texts") or [])
    scores = list(_result_field(page, "rec_scores") or [])
    lines = [
        (str(text).strip(), float(score))
        for text, score in zip(texts, scores, strict=False)
        if str(text).strip()
    ]
    if not lines:
        return "", 0.0
    separator = "\n" if multiline else " "
    return separator.join(text for text, _ in lines), min(score for _, score in lines)


def _combine_recognition_result(result: Any) -> tuple[str, float]:
    text = _result_field(result, "rec_text")
    confidence = _result_field(result, "rec_score")
    if isinstance(text, (tuple, list)):
        text = text[0] if text else ""
    value = str(text or "").strip()
    return value, float(confidence or 0.0) if value else 0.0


def _result_field(result: Any, key: str) -> Any:
    try:
        return result[key]
    except (KeyError, TypeError, IndexError):
        return getattr(result, key, None)


def _paddle_input(image: np.ndarray) -> np.ndarray:
    """Return an isolated contiguous BGR image for PaddleOCR's image reader."""
    if image.ndim == 2:
        converted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        converted = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        converted = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        converted = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError("OCR crop must be grayscale, BGR, or BGRA")
    return np.ascontiguousarray(converted)


def _candidate(
    crop: FieldCrop,
    value: str,
    confidence: float,
    engine: str,
    version: str,
    *,
    error_code: str | None = None,
) -> OCRCandidate:
    return OCRCandidate(
        field_name=crop.field_name,
        raw_value=value,
        confidence=max(0.0, min(1.0, confidence)),
        generator=crop.generator,
        variant_name=crop.variant_name,
        bounding_box=crop.bounding_box,
        ocr_engine=engine,
        ocr_model_version=version,
        error_code=error_code,
    )


def _failed_candidate(crop: FieldCrop, engine: str, version: str) -> OCRCandidate:
    return _candidate(
        crop,
        "",
        0.0,
        engine,
        version,
        error_code="OCR_FAILED",
    )
