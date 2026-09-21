from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image as PillowImage
from PIL import ImageOps, UnidentifiedImageError

from .contracts import Image, ImageSource, InputImage


class IntakeError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class IntakeLimits:
    max_encoded_bytes: int = 15 * 1024 * 1024
    max_width: int = 10_000
    max_height: int = 10_000
    max_pixels: int = 40_000_000


class ImageIntake:
    _SUPPORTED_FORMATS = {"JPEG", "PNG"}

    def __init__(self, limits: IntakeLimits | None = None) -> None:
        self.limits = limits or IntakeLimits()

    def load(
        self,
        source: ImageSource,
        *,
        processing_id: str | None = None,
    ) -> InputImage:
        identifier = processing_id or uuid4().hex
        if isinstance(source, np.ndarray):
            pixels = self._normalize_array(source)
            source_format = "array"
        elif isinstance(source, bytes):
            pixels, source_format = self._decode_bytes(source)
        elif isinstance(source, (str, Path)):
            encoded = self._read_path(Path(source))
            pixels, source_format = self._decode_bytes(encoded)
        else:
            raise TypeError(
                "Image source must be a path, encoded bytes, or uint8 NumPy array"
            )

        height, width = pixels.shape[:2]
        return InputImage(
            image=pixels,
            width=width,
            height=height,
            channels=3,
            source_format=source_format,
            processing_id=identifier,
        )

    def _read_path(self, path: Path) -> bytes:
        try:
            if not path.is_file():
                raise IntakeError("INPUT_NOT_FOUND", "Input image was not found")
            size = path.stat().st_size
            if size <= 0:
                raise IntakeError("EMPTY_INPUT", "Input image is empty")
            if size > self.limits.max_encoded_bytes:
                raise IntakeError("INPUT_TOO_LARGE", "Encoded image exceeds the size limit")
            return path.read_bytes()
        except IntakeError:
            raise
        except OSError as exc:
            raise IntakeError("INPUT_READ_FAILED", "Input image could not be read") from exc

    def _decode_bytes(self, encoded: bytes) -> tuple[Image, str]:
        if not encoded:
            raise IntakeError("EMPTY_INPUT", "Input image is empty")
        if len(encoded) > self.limits.max_encoded_bytes:
            raise IntakeError("INPUT_TOO_LARGE", "Encoded image exceeds the size limit")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", PillowImage.DecompressionBombWarning)
                with PillowImage.open(io.BytesIO(encoded)) as opened:
                    source_format = (opened.format or "").upper()
                    if source_format not in self._SUPPORTED_FORMATS:
                        raise IntakeError(
                            "UNSUPPORTED_FORMAT",
                            "Only JPEG and PNG images are supported",
                        )
                    self._validate_dimensions(opened.width, opened.height)
                    oriented = ImageOps.exif_transpose(opened)
                    rgb = oriented.convert("RGB")
                    rgb.load()
                    pixels = np.asarray(rgb, dtype=np.uint8)
        except IntakeError:
            raise
        except (UnidentifiedImageError, OSError, ValueError, PillowImage.DecompressionBombError) as exc:
            raise IntakeError("DECODE_FAILED", "Input image could not be decoded") from exc

        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr), source_format.lower()

    def _normalize_array(self, source: Image) -> Image:
        if source.dtype != np.uint8:
            raise IntakeError("INVALID_DTYPE", "Image arrays must use uint8 pixels")
        if source.size == 0:
            raise IntakeError("EMPTY_INPUT", "Input image is empty")
        if source.ndim not in (2, 3):
            raise IntakeError("INVALID_DIMENSIONS", "Image arrays must have two or three dimensions")

        height, width = source.shape[:2]
        self._validate_dimensions(width, height)
        if source.ndim == 2:
            bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        else:
            channels = source.shape[2]
            if channels == 1:
                bgr = cv2.cvtColor(source[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif channels == 3:
                bgr = source.copy()
            elif channels == 4:
                bgr = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
            else:
                raise IntakeError(
                    "INVALID_CHANNELS",
                    "Image arrays must have 1, 3, or 4 channels",
                )
        return np.ascontiguousarray(bgr)

    def _validate_dimensions(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise IntakeError("INVALID_DIMENSIONS", "Image dimensions must be positive")
        if width > self.limits.max_width or height > self.limits.max_height:
            raise IntakeError("IMAGE_TOO_LARGE", "Image dimensions exceed configured limits")
        if width * height > self.limits.max_pixels:
            raise IntakeError("IMAGE_TOO_LARGE", "Image pixel count exceeds configured limits")

