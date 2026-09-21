from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .contracts import DetectionResult, Image, Point, Quadrilateral
from .detection import DetectionError


class InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[Any]:
        ...

    def run(
        self, output_names: Sequence[str] | None, inputs: Mapping[str, NDArray[np.float32]]
    ) -> Sequence[Any]:
        ...


SessionFactory = Callable[[Path, tuple[str, ...]], InferenceSession]


@dataclass(frozen=True)
class OnnxDetectorManifest:
    version: str
    box_model_path: Path
    corner_model_path: Path
    box_model_sha256: str
    corner_model_sha256: str
    box_input_size: tuple[int, int]
    corner_input_size: tuple[int, int]
    box_score_threshold: float
    nms_iou_threshold: float
    corner_score_threshold: float
    document_type: str = "ao_id_card"
    side: str = "front"

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("Detector manifest version must not be empty")
        for label, path in (
            ("box", self.box_model_path),
            ("corner", self.corner_model_path),
        ):
            if not path.is_file():
                raise ValueError(f"{label.capitalize()} model file does not exist")
        for label, digest in (
            ("box", self.box_model_sha256),
            ("corner", self.corner_model_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
                raise ValueError(f"{label.capitalize()} model SHA-256 must be 64 hexadecimal characters")
        for label, size in (
            ("box", self.box_input_size),
            ("corner", self.corner_input_size),
        ):
            if len(size) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size):
                raise ValueError(f"{label.capitalize()} input size must contain two positive integers")
        for label, value in (
            ("box score", self.box_score_threshold),
            ("NMS IoU", self.nms_iou_threshold),
            ("corner score", self.corner_score_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label.capitalize()} threshold must be between 0 and 1")

    @classmethod
    def from_json(cls, path: str | Path) -> OnnxDetectorManifest:
        manifest_path = Path(path).resolve()
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Detector manifest is not readable JSON") from exc
        if not isinstance(data, dict):
            raise TypeError("Detector manifest root must be an object")

        def model_entry(name: str) -> tuple[Path, str, tuple[int, int]]:
            entry = data.get(name)
            if not isinstance(entry, dict):
                raise TypeError(f"Detector manifest '{name}' must be an object")
            relative_path = entry.get("path")
            digest = entry.get("sha256")
            size = entry.get("input_size")
            if not isinstance(relative_path, str) or not isinstance(digest, str):
                raise TypeError(f"Detector manifest '{name}' path and sha256 must be strings")
            if not isinstance(size, list) or len(size) != 2:
                raise TypeError(f"Detector manifest '{name}' input_size must be [width, height]")
            return (
                (manifest_path.parent / relative_path).resolve(),
                digest,
                (size[0], size[1]),
            )

        box_path, box_digest, box_size = model_entry("box_model")
        corner_path, corner_digest, corner_size = model_entry("corner_model")
        thresholds = data.get("thresholds")
        if not isinstance(thresholds, dict):
            raise TypeError("Detector manifest 'thresholds' must be an object")
        try:
            return cls(
                version=str(data["version"]),
                document_type=str(data.get("document_type", "ao_id_card")),
                side=str(data.get("side", "front")),
                box_model_path=box_path,
                corner_model_path=corner_path,
                box_model_sha256=box_digest,
                corner_model_sha256=corner_digest,
                box_input_size=box_size,
                corner_input_size=corner_size,
                box_score_threshold=float(thresholds["box_score"]),
                nms_iou_threshold=float(thresholds["nms_iou"]),
                corner_score_threshold=float(thresholds["corner_score"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Detector manifest contains invalid values") from exc

    def verify_files(self) -> None:
        for label, path, expected in (
            ("box", self.box_model_path, self.box_model_sha256),
            ("corner", self.corner_model_path, self.corner_model_sha256),
        ):
            if _sha256(path).lower() != expected.lower():
                raise ValueError(f"{label.capitalize()} model checksum does not match manifest")


class OnnxDocumentDetector:
    """Runs a box detector followed by a semantic four-corner model.

    The box model output is ``N x >=5`` in input-pixel coordinates as
    ``x1, y1, x2, y2, score``. The corner model output is ``4 x >=3`` in
    semantic TL, TR, BR, BL order as normalized ``x, y, score`` values.
    """

    name = "onnx_ao_id_front"

    def __init__(
        self,
        manifest: OnnxDetectorManifest,
        *,
        device: str = "cpu",
        session_factory: SessionFactory | None = None,
    ) -> None:
        manifest.verify_files()
        if device not in {"cpu", "gpu"}:
            raise ValueError("ONNX detector device must be 'cpu' or 'gpu'")
        providers = (
            ("CUDAExecutionProvider", "CPUExecutionProvider")
            if device == "gpu"
            else ("CPUExecutionProvider",)
        )
        factory = session_factory or _default_session_factory
        self._manifest = manifest
        self._box_session = factory(manifest.box_model_path, providers)
        self._corner_session = factory(manifest.corner_model_path, providers)

    def detect(self, image: Image) -> DetectionResult:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("ONNX detector requires a BGR uint8 image")
        tensor, scale, offset = _letterbox(image, self._manifest.box_input_size)
        output = _run(self._box_session, tensor)
        boxes = _parse_boxes(output, self._manifest.box_score_threshold)
        boxes = _nms(boxes, self._manifest.nms_iou_threshold)
        if not boxes:
            raise DetectionError("NO_DOCUMENT", "No supported document was detected")
        if len(boxes) > 1:
            raise DetectionError(
                "MULTIPLE_DOCUMENTS",
                "Multiple plausible documents were detected",
            )

        x1, y1, x2, y2, box_score = _restore_box(
            boxes[0], scale, offset, image.shape[1], image.shape[0]
        )
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise DetectionError("INVALID_DETECTION", "Detected document bounds are empty")
        corner_tensor = _resize_tensor(crop, self._manifest.corner_input_size)
        corners = _parse_corners(_run(self._corner_session, corner_tensor))
        corner_scores = corners[:, 2]
        if np.any(corner_scores < self._manifest.corner_score_threshold):
            raise DetectionError(
                "LOW_CORNER_CONFIDENCE",
                "Document corners did not meet the configured confidence threshold",
            )
        width = x2 - x1
        height = y2 - y1
        corner_width = max(0, width - 1)
        corner_height = max(0, height - 1)
        points = tuple(
            Point(
                float(x1 + np.clip(corner[0], 0.0, 1.0) * corner_width),
                float(y1 + np.clip(corner[1], 0.0, 1.0) * corner_height),
            )
            for corner in corners
        )
        mean_corner_score = float(np.mean(corner_scores))
        confidence = min(float(box_score), mean_corner_score)
        return DetectionResult(
            document_type=self._manifest.document_type,
            side=self._manifest.side,
            confidence=confidence,
            corners=Quadrilateral(points),
            detector=self.name,
            detector_version=self._manifest.version,
            confidence_components={
                "box_score": float(box_score),
                "corner_score": mean_corner_score,
            },
        )


def _default_session_factory(path: Path, providers: tuple[str, ...]) -> InferenceSession:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "ONNX detection requires the project 'detection' dependency extra"
        ) from exc
    available = set(ort.get_available_providers())
    if providers[0] == "CUDAExecutionProvider" and providers[0] not in available:
        raise RuntimeError("CUDAExecutionProvider is not available")
    return ort.InferenceSession(str(path), providers=list(providers))


def _run(session: InferenceSession, tensor: NDArray[np.float32]) -> Any:
    inputs = session.get_inputs()
    if len(inputs) != 1 or not getattr(inputs[0], "name", None):
        raise RuntimeError("ONNX model must expose exactly one named input")
    outputs = session.run(None, {inputs[0].name: tensor})
    if not outputs:
        raise RuntimeError("ONNX model returned no outputs")
    return outputs[0]


def _letterbox(
    image: Image, size: tuple[int, int]
) -> tuple[NDArray[np.float32], float, tuple[int, int]]:
    target_width, target_height = size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return _to_tensor(canvas), scale, (left, top)


def _resize_tensor(image: Image, size: tuple[int, int]) -> NDArray[np.float32]:
    return _to_tensor(cv2.resize(image, size, interpolation=cv2.INTER_LINEAR))


def _to_tensor(image: Image) -> NDArray[np.float32]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def _parse_boxes(output: Any, threshold: float) -> list[tuple[float, float, float, float, float]]:
    values = np.asarray(output, dtype=np.float32).squeeze()
    if values.size == 0:
        return []
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 5:
        raise RuntimeError("Box model output must have shape N x >=5")
    return [
        tuple(float(value) for value in row[:5])
        for row in values
        if float(row[4]) >= threshold and row[2] > row[0] and row[3] > row[1]
    ]


def _nms(
    boxes: Sequence[tuple[float, float, float, float, float]], threshold: float
) -> list[tuple[float, float, float, float, float]]:
    if not boxes:
        return []
    rectangles = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in boxes]
    scores = [score for *_, score in boxes]
    indexes = cv2.dnn.NMSBoxes(rectangles, scores, 0.0, threshold)
    selected = [boxes[int(index)] for index in np.asarray(indexes).reshape(-1)]
    return sorted(selected, key=lambda item: item[4], reverse=True)


def _restore_box(
    box: tuple[float, float, float, float, float],
    scale: float,
    offset: tuple[int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int, float]:
    left, top = offset
    x1, y1, x2, y2, score = box
    restored = (
        max(0, min(image_width - 1, int(round((x1 - left) / scale)))),
        max(0, min(image_height - 1, int(round((y1 - top) / scale)))),
        max(1, min(image_width, int(round((x2 - left) / scale)))),
        max(1, min(image_height, int(round((y2 - top) / scale)))),
        score,
    )
    if restored[2] <= restored[0] or restored[3] <= restored[1]:
        raise DetectionError("INVALID_DETECTION", "Detected document bounds are invalid")
    return restored


def _parse_corners(output: Any) -> NDArray[np.float32]:
    values = np.asarray(output, dtype=np.float32).squeeze()
    if values.shape != (4, 3) or not np.all(np.isfinite(values)):
        raise RuntimeError("Corner model output must have shape 4 x 3")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
