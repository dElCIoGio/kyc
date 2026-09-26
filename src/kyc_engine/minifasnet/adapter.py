"""Concrete ``AntiSpoofDetector`` backed by upstream-compatible MiniFASNet."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from kyc_engine.contracts import AntiSpoofResult, Image

from .checkpoint import MiniFASNetModelSpec, discover_models, validate_resources
from .errors import MiniFASNetInitializationError, MiniFASNetInputError
from .model import LoadedMiniFASNetModel, load_model
from .preprocessing import CaffeFaceRegionDetector, FaceBox, crop_frame, validate_frame


class FaceRegionDetector(Protocol):
    def detect(self, frame: np.ndarray) -> FaceBox: ...


class _Predictor(Protocol):
    def predict(self, frame: np.ndarray) -> np.ndarray: ...


class MiniFASNetAntiSpoofDetector:
    """Passive anti-spoof detector using loaded MiniFASNet checkpoints.

    Model files and the RetinaFace Caffe resources live below the explicitly
    supplied private model root.  Construction loads every model exactly once;
    ``detect`` has no disk I/O and retains no input frame.
    """

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "cpu",
        _face_detector: FaceRegionDetector | None = None,
        _model_loader: Callable[[MiniFASNetModelSpec, str], _Predictor] = load_model,
    ) -> None:
        root = Path(model_dir)
        model_paths = root / "anti_spoof_models"
        detector_paths = root / "detection_model"
        validate_resources((
            detector_paths / "deploy.prototxt",
            detector_paths / "Widerface-RetinaFace.caffemodel",
        ))
        specs = discover_models(model_paths)
        try:
            face_detector = _face_detector or CaffeFaceRegionDetector(
                str(detector_paths / "deploy.prototxt"),
                str(detector_paths / "Widerface-RetinaFace.caffemodel"),
            )
            loaded = tuple((spec, _model_loader(spec, device)) for spec in specs)
        except MiniFASNetInitializationError:
            raise
        except Exception as exc:
            raise MiniFASNetInitializationError("MiniFASNet could not be initialized") from exc
        if not loaded:
            raise MiniFASNetInitializationError("MiniFASNet model files are missing")
        self._face_detector = face_detector
        self._models = loaded

    def detect(self, frame: Image) -> AntiSpoofResult:
        frame = validate_frame(frame)
        bbox = self._face_detector.detect(frame)
        predictions = []
        for spec, model in self._models:
            crop = crop_frame(
                frame,
                bbox,
                scale=spec.scale,
                output_width=spec.input_width,
                output_height=spec.input_height,
            )
            predictions.append(_validate_probability_vector(model.predict(crop)))
        probabilities = np.mean(np.stack(predictions), axis=0)
        real_score = float(probabilities[1])
        return AntiSpoofResult(is_real=int(np.argmax(probabilities)) == 1, score=real_score)


def _validate_probability_vector(prediction: np.ndarray) -> np.ndarray:
    vector = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise MiniFASNetInputError("MiniFASNet produced an invalid prediction")
    if np.any(vector < 0) or np.any(vector > 1) or not np.isclose(float(vector.sum()), 1.0, atol=1e-5):
        raise MiniFASNetInputError("MiniFASNet produced an invalid prediction")
    return vector
