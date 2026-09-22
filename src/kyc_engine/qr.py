from __future__ import annotations

import re
import logging
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from .contracts import (
    BoundingBox,
    QrCodeDefinition,
    QrCodeResult,
    QrCodeStatus,
    QrIdentityData,
    VariantBatch,
    VariantInfo,
)


logger = logging.getLogger(__name__)

_PREFERRED_VARIANTS = (
    "original",
    "grayscale",
    "clahe_clip_2_grid_8x8",
    "gamma_0.8",
    "gamma_1.2",
)
_DATE_PATTERN = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
_VERSION_PATTERN = re.compile(r"\bV(?P<version>\d+)\s*$", re.IGNORECASE)
_ID_PATTERN = re.compile(r"\b(?=[A-Z0-9/-]*\d)[A-Z0-9/-]{5,30}\b", re.IGNORECASE)
_SEX_PATTERN = re.compile(r"\b(?P<sex>MASCULINO|FEMININO)\b", re.IGNORECASE)


class QrCodeExtractor:
    """Decode one configured QR region without using OCR field evidence."""

    name = "zxingcpp"

    def __init__(self, decoder: Any | None = None) -> None:
        if decoder is None:
            try:
                import zxingcpp
            except ImportError as exc:
                raise ImportError("QR extraction requires the zxing-cpp dependency") from exc
            decoder = zxingcpp
        self._decoder = decoder
        self.version = str(getattr(decoder, "__version__", "unknown"))

    def extract(
        self,
        batches: Sequence[VariantBatch],
        definition: QrCodeDefinition,
    ) -> QrCodeResult:
        detected = False
        parse_failure: tuple[str, str] | None = None
        for variant in _preferred_variants(batches):
            crop = _crop(variant.image, definition)
            detected = _opencv_detects_qr(crop) or detected
            payload = self._decode(crop)
            if not payload:
                continue
            data = parse_qr_payload(payload)
            if data is not None:
                return QrCodeResult(
                    status=QrCodeStatus.DECODED,
                    raw_payload=payload,
                    data=data,
                    bounding_box=definition.bounding_box,
                    decoder=self.name,
                    decoder_version=self.version,
                    variant_name=variant.name,
                )
            if parse_failure is None:
                parse_failure = (payload, variant.name)

        if parse_failure is not None:
            payload, variant_name = parse_failure
            return QrCodeResult(
                status=QrCodeStatus.PARSE_FAILED,
                raw_payload=payload,
                data=None,
                bounding_box=definition.bounding_box,
                decoder=self.name,
                decoder_version=self.version,
                variant_name=variant_name,
                warnings=("QR payload did not match the configured record grammar",),
            )
        return QrCodeResult(
            status=QrCodeStatus.DECODE_FAILED if detected else QrCodeStatus.NOT_DETECTED,
            raw_payload=None,
            data=None,
            bounding_box=definition.bounding_box,
            decoder=self.name,
            decoder_version=self.version,
            variant_name=None,
            warnings=("QR code could not be decoded",),
        )

    def _decode(self, image: np.ndarray) -> str | None:
        try:
            barcode = self._decoder.read_barcode(
                np.ascontiguousarray(image.copy()),
                formats=self._decoder.BarcodeFormat.QRCode,
            )
        except Exception as exc:
            logger.exception(
                "QR decoder backend failed",
                extra={
                    "event": "qr_decode_backend_failed",
                    "stage": "qr",
                    "exception_type": type(exc).__name__,
                },
            )
            return None
        if barcode is None:
            return None
        value = getattr(barcode, "text", None)
        return value if isinstance(value, str) and value else None


def parse_qr_payload(payload: str) -> QrIdentityData | None:
    """Parse the fixed QR record grammar without consulting OCR output."""
    normalized = " ".join(payload.split())
    version = _VERSION_PATTERN.search(normalized)
    dates = tuple(_DATE_PATTERN.finditer(normalized))
    if version is None or len(dates) != 3 or dates[-1].end() > version.start():
        return None

    prefix = normalized[: dates[0].start()].strip()
    identifiers = tuple(_ID_PATTERN.finditer(prefix))
    if not identifiers:
        return None
    identifier = identifiers[-1]
    full_name = prefix[: identifier.start()].strip()
    birth_province = prefix[identifier.end() :].strip()

    middle = normalized[dates[0].end() : dates[1].start()].strip()
    sex = _SEX_PATTERN.search(middle)
    if sex is None:
        return None
    marital_status = middle[sex.end() :].strip()
    issuing_province = normalized[dates[2].end() : version.start()].strip()
    if not all((full_name, birth_province, marital_status, issuing_province)):
        return None

    return QrIdentityData(
        full_name=full_name,
        id_number=identifier.group(0),
        birth_province=birth_province,
        date_of_birth=dates[0].group(0),
        sex=sex.group("sex").upper(),
        marital_status=marital_status,
        issue_date=dates[1].group(0),
        expiry_date=dates[2].group(0),
        issuing_province=issuing_province,
        version=version.group("version"),
    )


def _preferred_variants(batches: Sequence[VariantBatch]) -> tuple[VariantInfo, ...]:
    variants = {
        variant.name: variant
        for batch in batches
        for variant in batch.variants
    }
    return tuple(variants[name] for name in _PREFERRED_VARIANTS if name in variants)


def _crop(image: np.ndarray, definition: QrCodeDefinition) -> np.ndarray:
    box = definition.bounding_box
    left = max(0, box.x - definition.padding)
    top = max(0, box.y - definition.padding)
    right = min(image.shape[1], box.right + definition.padding)
    bottom = min(image.shape[0], box.bottom + definition.padding)
    return image[top:bottom, left:right].copy()


def _opencv_detects_qr(image: np.ndarray) -> bool:
    try:
        detected, _ = cv2.QRCodeDetector().detect(image)
        return bool(detected)
    except cv2.error:
        return False


__all__ = ["QrCodeExtractor", "parse_qr_payload"]
