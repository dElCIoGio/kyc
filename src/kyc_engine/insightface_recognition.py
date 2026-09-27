"""Local, offline-only InsightFace implementation of :class:`FaceRecognizer`."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from re import fullmatch
from types import ModuleType
from typing import Any

import numpy as np

from .contracts import FaceEmbedding, FaceRecognitionResult, Image
from .face_recognition import (
    FaceRecognitionInferenceError,
    FaceRecognitionInitializationError,
    FaceRecognitionMultipleFacesError,
    FaceRecognitionNoFaceError,
)


_MODEL_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
_AnalysisFactory = Callable[..., Any]


class InsightFaceRecognizer:
    """Use one pre-provisioned InsightFace model package without network access.

    ``FaceAnalysis.get`` owns face detection, landmarks, alignment, and
    recognition. This adapter only validates that exactly one returned Face is
    present, then reads that same Face object's embedding.
    """

    provider_name = "insightface"

    def __init__(
        self,
        model_root: Path,
        model_id: str,
        *,
        _analysis_factory: _AnalysisFactory | None = None,
    ) -> None:
        self.model_id = _validate_model_id(model_id)
        self._model_root = Path(model_root)
        self._model_dir = _validate_model_package(self._model_root, self.model_id)
        try:
            if _analysis_factory is None:
                analysis_factory, analysis_module = _load_face_analysis()
                with _offline_model_package(analysis_module, self._model_root, self.model_id, self._model_dir):
                    analysis = analysis_factory(
                        name=self.model_id,
                        root=str(self._model_root),
                        allowed_modules=("detection", "recognition"),
                        addons=(),
                        providers=["CPUExecutionProvider"],
                    )
            else:
                analysis = _analysis_factory(
                    name=self.model_id,
                    root=str(self._model_root),
                    allowed_modules=("detection", "recognition"),
                    addons=(),
                    providers=["CPUExecutionProvider"],
                )
            analysis.prepare(ctx_id=0)
            modules = getattr(analysis, "models", {})
            if not {"detection", "recognition"}.issubset(modules):
                raise ValueError("Configured model package must provide detection and recognition")
        except FaceRecognitionInitializationError:
            raise
        except Exception as exc:
            raise FaceRecognitionInitializationError(
                "Configured local face-recognition model could not be initialized"
            ) from exc
        self._analysis = analysis

    def encode(self, image: Image) -> FaceRecognitionResult:
        try:
            # InsightFace receives an ephemeral writable BGR copy; immutable session
            # artifacts are never handed to a third-party model implementation.
            faces = tuple(self._analysis.get(np.array(image, dtype=np.uint8, copy=True)))
        except Exception as exc:
            raise FaceRecognitionInferenceError("Face-recognition inference failed") from exc
        face_count = len(faces)
        if face_count == 0:
            raise FaceRecognitionNoFaceError("Recognition found no faces")
        if face_count != 1:
            raise FaceRecognitionMultipleFacesError(face_count)

        face = faces[0]
        try:
            # ``FaceAnalysis.get`` runs recognition against this Face's own detected
            # landmarks/alignment. Do not perform a separate crop or face selection.
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                raise ValueError("Recognition did not provide an embedding")
            confidence = _detection_confidence(getattr(face, "det_score", None))
            return FaceRecognitionResult(
                embedding=FaceEmbedding(np.asarray(embedding)),
                face_count=1,
                detection_confidence=confidence,
            )
        except Exception as exc:
            raise FaceRecognitionInferenceError("Face-recognition embedding was unavailable") from exc


def _validate_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or fullmatch(_MODEL_ID_PATTERN, model_id) is None:
        raise FaceRecognitionInitializationError("Face-recognition model identifier is invalid")
    return model_id


def _validate_model_package(model_root: Path, model_id: str) -> Path:
    if not model_root.is_dir():
        raise FaceRecognitionInitializationError("Face-recognition model root is not a directory")
    model_dir = model_root / "models" / model_id
    onnx_files = tuple(model_dir.glob("*.onnx")) if model_dir.is_dir() else ()
    if not onnx_files or not all(path.is_file() and path.stat().st_size > 0 for path in onnx_files):
        raise FaceRecognitionInitializationError("Face-recognition model package is missing or invalid")
    return model_dir


def _load_face_analysis() -> tuple[_AnalysisFactory, ModuleType]:
    try:
        from insightface.app import FaceAnalysis
        import insightface.app.face_analysis as face_analysis_module
    except Exception as exc:
        raise FaceRecognitionInitializationError(
            "InsightFace recognition dependency is unavailable"
        ) from exc
    return FaceAnalysis, face_analysis_module


@contextmanager
def _offline_model_package(
    module: ModuleType, model_root: Path, model_id: str, model_dir: Path
):
    """Prevent FaceAnalysis from using its automatic package downloader at startup."""

    original = getattr(module, "ensure_available", None)
    if original is None:
        raise FaceRecognitionInitializationError("InsightFace offline model loader is unavailable")

    def local_only(category: str, name: str, *, root: str | Path | None = None, **_kwargs: object) -> str:
        if category != "models" or name != model_id or Path(root or "").resolve() != model_root.resolve():
            raise FaceRecognitionInitializationError("Automatic InsightFace model downloads are disabled")
        return str(model_dir)

    setattr(module, "ensure_available", local_only)
    try:
        yield
    finally:
        setattr(module, "ensure_available", original)


def _detection_confidence(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        confidence = float(value)
        if 0.0 <= confidence <= 1.0:
            return confidence
    return None
