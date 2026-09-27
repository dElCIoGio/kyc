from __future__ import annotations

import cv2
import numpy as np

from .contracts import Image, VariantBatch, VariantInfo


class BalancedVariantPolicy:
    """Generate a small deterministic set of OCR-oriented preprocessing variants.

    The policy is deliberately conservative: it always preserves the original
    image first, produces at most ten variants, never mutates the caller's array,
    and keeps transformation provenance in each VariantInfo.
    """

    def generate(self, image: Image) -> tuple[VariantBatch, ...]:
        source = _validate_image(image)
        gray = _to_grayscale(source)

        batches = (
            VariantBatch(
                "original",
                (
                    VariantInfo(
                        "original",
                        source.copy(),
                        {"transform": "none"},
                    ),
                ),
            ),
            VariantBatch(
                "grayscale",
                (
                    VariantInfo(
                        "grayscale",
                        gray.copy(),
                        {"conversion": "bgr_to_gray" if source.ndim == 3 else "identity"},
                    ),
                ),
            ),
            VariantBatch(
                "clahe",
                (
                    VariantInfo(
                        "clahe",
                        _clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8)),
                        {
                            "clip_limit": 2.0,
                            "tile_grid_size": (8, 8),
                        },
                    ),
                ),
            ),
            VariantBatch(
                "gamma",
                (
                    VariantInfo(
                        "gamma_0_75",
                        _gamma(gray, 0.75),
                        {"gamma": 0.75},
                    ),
                    VariantInfo(
                        "gamma_1_25",
                        _gamma(gray, 1.25),
                        {"gamma": 1.25},
                    ),
                ),
            ),
            VariantBatch(
                "contrast",
                (
                    VariantInfo(
                        "contrast_0_85",
                        _contrast(gray, alpha=0.85),
                        {"alpha": 0.85, "beta": 0.0},
                    ),
                    VariantInfo(
                        "contrast_1_15",
                        _contrast(gray, alpha=1.15),
                        {"alpha": 1.15, "beta": 0.0},
                    ),
                ),
            ),
            VariantBatch(
                "threshold",
                (
                    VariantInfo(
                        "threshold_otsu",
                        _otsu(gray),
                        {"method": "otsu"},
                    ),
                    VariantInfo(
                        "threshold_adaptive_gaussian",
                        _adaptive_threshold(gray, block_size=31, c=11),
                        {
                            "method": "adaptive_gaussian",
                            "block_size": 31,
                            "c": 11,
                        },
                    ),
                ),
            ),
            VariantBatch(
                "sharpen",
                (
                    VariantInfo(
                        "sharpen",
                        _sharpen(gray),
                        {"kernel": "cross_5"},
                    ),
                ),
            ),
        )

        if sum(len(batch.variants) for batch in batches) > 10:
            raise RuntimeError("Balanced variant policy exceeded its ten-variant limit")
        return batches


def _validate_image(image: Image) -> Image:
    if not isinstance(image, np.ndarray):
        raise TypeError("Variant input must be a NumPy array")
    if image.dtype != np.uint8:
        raise ValueError("Variant input must use uint8 pixels")
    if image.ndim == 2:
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("Variant input dimensions must be positive")
    elif image.ndim == 3:
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("Variant input dimensions must be positive")
        if image.shape[2] not in (1, 3, 4):
            raise ValueError("Variant input must have 1, 3, or 4 channels")
    else:
        raise ValueError("Variant input must be a 2D or 3D image")
    return np.array(image, dtype=np.uint8, copy=True, order="C")


def _to_grayscale(image: Image) -> Image:
    if image.ndim == 2:
        return image.copy()
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0].copy()
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)


def _clahe(
    gray: Image,
    *,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> Image:
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(gray)


def _gamma(gray: Image, gamma: float) -> Image:
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    values = np.arange(256, dtype=np.float32) / 255.0
    table = np.clip(np.power(values, gamma) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.LUT(gray, table)


def _contrast(gray: Image, *, alpha: float) -> Image:
    if alpha <= 0.0:
        raise ValueError("contrast alpha must be positive")
    adjusted = gray.astype(np.float32) * float(alpha)
    return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)


def _otsu(gray: Image) -> Image:
    _threshold, output = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    return output


def _adaptive_threshold(gray: Image, *, block_size: int, c: int) -> Image:
    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError("adaptive threshold block_size must be odd and greater than one")
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def _sharpen(gray: Image) -> Image:
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    return cv2.filter2D(gray, ddepth=-1, kernel=kernel)
