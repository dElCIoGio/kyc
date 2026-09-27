"""Local, offline-only InsightFace implementation of :class:`FaceRecognizer`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from re import fullmatch
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
_CPU_PROVIDER = "CPUExecutionProvider"
_DETECTION_INPUT_SIZE = (640, 640)
_ModelLoader = Callable[..., Any]
_FaceFactory = Callable[..., Any]


class InsightFaceRecognizer:
    """Use explicit local InsightFace ONNX files without package downloads.

    Local ONNX files are loaded with InsightFace's model-zoo API and identified
    by the resulting model's ``taskname``. Detection and recognition models are
    retained once at startup. The detector's sole returned face and landmarks
    are passed to recognition unchanged, so InsightFace owns alignment and the
    expected recognition preprocessing.
    """

    provider_name = "insightface"

    def __init__(
        self,
        model_root: Path,
        model_id: str,
        *,
        _model_loader: _ModelLoader | None = None,
        _face_factory: _FaceFactory | None = None,
    ) -> None:
        self.model_id = _validate_model_id(model_id)
        self._model_root = Path(model_root)
        self._model_dir = _validate_model_package(self._model_root, self.model_id)
        try:
            if _model_loader is None or _face_factory is None:
                default_loader, default_face_factory = _load_insightface_apis()
                model_loader = _model_loader or default_loader
                face_factory = _face_factory or default_face_factory
            else:
                model_loader = _model_loader
                face_factory = _face_factory
            detector, recognition = _load_required_models(self._model_dir, model_loader)
            detector.prepare(ctx_id=-1, input_size=_DETECTION_INPUT_SIZE)
            recognition.prepare(ctx_id=-1)
        except FaceRecognitionInitializationError:
            raise
        except Exception as exc:
            raise FaceRecognitionInitializationError(
                "Configured local face-recognition model could not be initialized"
            ) from exc
        self._detector = detector
        self._recognition = recognition
        self._face_factory = face_factory

    def encode(self, image: Image) -> FaceRecognitionResult:
        # InsightFace receives an ephemeral writable BGR copy; immutable session
        # artifacts and offline-loader images are never passed through directly.
        writable_image = np.array(image, dtype=np.uint8, copy=True)
        try:
            detections, landmarks = self._detector.detect(writable_image, max_num=0)
            detection_rows = _detection_rows(detections)
        except Exception as exc:
            raise FaceRecognitionInferenceError("Face-recognition inference failed") from exc

        face_count = len(detection_rows)
        if face_count == 0:
            raise FaceRecognitionNoFaceError("Recognition found no faces")
        if face_count != 1:
            raise FaceRecognitionMultipleFacesError(face_count)

        try:
            landmark_rows = _landmark_rows(landmarks)
            if len(landmark_rows) != 1:
                raise ValueError("Detection landmarks were unavailable")
            detection = detection_rows[0]
            if detection.size < 5:
                raise ValueError("Detection confidence was unavailable")
            # This exact detection object, including its landmarks, is passed to
            # InsightFace recognition. ArcFace's ``get`` aligns from ``face.kps``.
            face = self._face_factory(
                bbox=np.asarray(detection[:4], dtype=np.float32),
                kps=np.asarray(landmark_rows[0], dtype=np.float32),
                det_score=float(detection[4]),
            )
            self._recognition.get(writable_image, face)
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                raise ValueError("Recognition did not provide an embedding")
            return FaceRecognitionResult(
                embedding=FaceEmbedding(np.asarray(embedding)),
                face_count=1,
                detection_confidence=_detection_confidence(detection[4]),
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
    if not model_dir.is_dir() or not any(path.is_file() and path.stat().st_size > 0 for path in model_dir.rglob("*.onnx")):
        raise FaceRecognitionInitializationError("Face-recognition model package is missing or invalid")
    return model_dir


def _load_insightface_apis() -> tuple[_ModelLoader, _FaceFactory]:
    try:
        from insightface.app.common import Face
        from insightface.model_zoo import get_model
    except Exception as exc:
        raise FaceRecognitionInitializationError(
            "InsightFace recognition dependency is unavailable"
        ) from exc
    return get_model, Face


def _load_required_models(model_dir: Path, model_loader: _ModelLoader) -> tuple[Any, Any]:
    discovered: dict[str, list[Any]] = {"detection": [], "recognition": []}
    for path in sorted(model_dir.rglob("*.onnx")):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        try:
            model = model_loader(str(path), providers=[_CPU_PROVIDER])
        except Exception:
            # A pack can contain unrelated or unsupported ONNX files. They are
            # irrelevant unless no usable required role can be found.
            continue
        task = getattr(model, "taskname", None)
        if task in discovered:
            discovered[task].append(model)

    if any(len(models) != 1 for models in discovered.values()):
        raise FaceRecognitionInitializationError(
            "Configured local face-recognition model must contain exactly one usable "
            "detection model and exactly one usable recognition model"
        )
    return discovered["detection"][0], discovered["recognition"][0]


def _detection_rows(detections: object) -> tuple[np.ndarray, ...]:
    if detections is None:
        return ()
    rows = np.asarray(detections)
    if rows.ndim != 2:
        raise ValueError("Detection output is invalid")
    return tuple(rows[index] for index in range(rows.shape[0]))


def _landmark_rows(landmarks: object) -> tuple[np.ndarray, ...]:
    if landmarks is None:
        return ()
    rows = np.asarray(landmarks)
    if rows.ndim != 3 or rows.shape[0] == 0:
        return ()
    return tuple(rows[index] for index in range(rows.shape[0]))


def _detection_confidence(value: object) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(confidence) and 0.0 <= confidence <= 1.0:
        return confidence
    return None
