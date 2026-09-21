"""
field_extractor.py
===================

Extracts a fixed set of fields, at fixed pixel locations, from a set of
image *variants* that all represent the same underlying document and share
the same crop (and therefore the same coordinate system).

Scope
-----
This module does exactly one job: given field locations and a set of image
variants, return the OCR value + confidence for every field in every
variant. It deliberately does NOT reconcile/merge values across variants
(e.g. "pick the highest-confidence variant per field") -- that join step
happens downstream, later, in a different component.

Architecture
------------
- `BoundingBox` / `FieldDefinition`  -> where each field lives (injected config)
- `ImageVariant`                     -> one version of the source image
- `OCREngine` (ABC)                  -> pluggable OCR backend interface
- `PaddleOCREngine`                  -> the only backend implemented for now
- `FieldExtractor`                   -> orchestrates cropping + OCR + result assembly

Swapping OCR engines later (e.g. adding Tesseract) means writing a new
`OCREngine` subclass -- `FieldExtractor` never needs to change.

Assumptions
-----------
- Bounding boxes are absolute pixel coordinates. All variants of a given
  document are assumed to share the same pixel dimensions (they are "the
  same crop"), so one set of `FieldDefinition`s applies to every variant.
  If a future variant pipeline produces images at different resolutions,
  bounding boxes will need to be normalized (0-1 range) instead -- that is
  a natural extension point but is not implemented here.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "BoundingBox",
    "FieldDefinition",
    "ImageVariant",
    "OCRLine",
    "OCRReadResult",
    "OCREngine",
    "PaddleOCREngine",
    "FieldExtractionResult",
    "ImageExtractionResult",
    "FieldExtractor",
]


# ---------------------------------------------------------------------------
# Field configuration (the "external objects" injected into FieldExtractor)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned pixel region within a (pre-cropped) document image.

    Coordinates are absolute pixels with (0, 0) at the top-left corner,
    consistent with the shared crop applied to every variant of a document.

    Attributes:
        x: Left edge, in pixels.
        y: Top edge, in pixels.
        width: Region width, in pixels.
        height: Region height, in pixels.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"BoundingBox must have positive width/height, got "
                f"{self.width}x{self.height}"
            )
        if self.x < 0 or self.y < 0:
            raise ValueError(
                f"BoundingBox origin must be non-negative, got ({self.x}, {self.y})"
            )

    @property
    def x2(self) -> int:
        """Right edge, in pixels (exclusive)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge, in pixels (exclusive)."""
        return self.y + self.height

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Return (left, top, right, bottom)."""
        return self.x, self.y, self.x2, self.y2


@dataclass(frozen=True)
class FieldDefinition:
    """Describes one field to extract: a name and where to find it.

    A collection of these is supplied by the caller (injected into
    `FieldExtractor` at construction time) so the same extractor code can
    serve any document layout without modification.

    Attributes:
        name: Unique identifier for the field (e.g. "full_name", "date_of_birth").
            Used as the key in extraction results.
        bounding_box: Where the field lives, in the shared crop's pixel space.
        multi_line: Whether the field is expected to span multiple lines
            (e.g. an address). Affects how OCR lines within the region are
            joined: multi-line fields are joined with newlines, single-line
            fields are joined with spaces (guarding against a single visual
            line being split into more than one detection).
        description: Optional free-text note for humans reading the config
            (not used by the extractor itself).
    """

    name: str
    bounding_box: BoundingBox
    multi_line: bool = False
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Image variants (what gets passed into FieldExtractor's methods)
# ---------------------------------------------------------------------------


ImageSource = Union[str, Path, bytes, np.ndarray, Image.Image]
"""Anything that can be resolved into pixel data for a single image variant."""


@dataclass(frozen=True)
class ImageVariant:
    """One version of the same source document.

    Attributes:
        variant_id: Caller-assigned identifier for this variant
            (e.g. "original", "denoised", "contrast_enhanced"). Used to key
            results so the (future) join step can tell variants apart.
        image: The image itself -- a file path, raw bytes, a PIL Image, or
            a numpy array.
    """

    variant_id: str
    image: ImageSource


# ---------------------------------------------------------------------------
# OCR engine interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OCRLine:
    """A single recognized line of text within a region, in reading order.

    Attributes:
        text: Recognized text for this line.
        confidence: Engine-reported confidence in [0, 1].
    """

    text: str
    confidence: float


@dataclass(frozen=True)
class OCRReadResult:
    """Raw OCR output for one image region.

    Kept engine-agnostic on purpose: it's just recognized lines, top to
    bottom. Turning that into a single field value + confidence is
    `FieldExtractor`'s job (via `FieldDefinition.multi_line`), not the
    engine's -- that way every engine implementation stays simple and the
    combination logic stays consistent across engines.
    """

    lines: Tuple[OCRLine, ...] = ()

    @property
    def is_empty(self) -> bool:
        return len(self.lines) == 0


class OCREngine(ABC):
    """Interface every OCR backend must implement.

    This abstraction sits between `FieldExtractor` and whichever OCR
    library actually does the recognition, so the backend can be swapped
    (Tesseract, PaddleOCR, a cloud OCR API, ...) without touching
    extraction logic.
    """

    @abstractmethod
    def read_text(self, image: np.ndarray) -> OCRReadResult:
        """Run OCR on an already-cropped field region.

        Args:
            image: RGB image array of just the field region, shape
                (H, W, 3), dtype uint8.

        Returns:
            An `OCRReadResult` with recognized lines in reading order.
            Implementations should not raise for "no text found" --
            return an empty result instead. Raising is reserved for
            genuine engine/processing failures, which `FieldExtractor`
            will catch and record per-field.
        """
        raise NotImplementedError


class PaddleOCREngine(OCREngine):
    """`OCREngine` backed by PaddleOCR's detection + recognition pipeline.

    Uses the full det+rec pipeline (rather than the recognition-only
    module) because field bounding boxes are configured externally and may
    not frame text as tightly or as single-line as the underlying model
    would like -- letting PaddleOCR (re)detect text within the crop is
    more robust to that than assuming the crop is already exactly one
    text line.

    Document-level preprocessing (page orientation classification, page
    unwarping) is disabled by default since fields are pre-cropped
    sub-regions, not full pages -- that correction is expected to have
    already happened (if needed) upstream, before cropping.

    Requires the `paddleocr` package (and its `paddlepaddle` dependency)
    to be installed; the import is deferred to `__init__` so importing
    this module doesn't require paddleocr unless you actually construct
    this engine.
    """

    def __init__(
        self,
        lang: str = "en",
        use_textline_orientation: bool = False,
        confidence_strategy: "ConfidenceStrategy" = "min",
        **paddleocr_kwargs: Any,
    ) -> None:
        """
        Args:
            lang: PaddleOCR language code (e.g. "en", "ch", "fr"). See the
                PaddleOCR docs for supported values.
            use_textline_orientation: Whether to correct individual
                text-line rotation within a field crop. Leave off (default)
                if fields are always upright; turn on for documents where a
                field's text could be rotated independently of the page.
            confidence_strategy: How to combine confidences when a field's
                region yields multiple recognized lines. "min" (default) is
                conservative -- a field is only as trustworthy as its
                weakest line, which suits a compliance context where you'd
                rather under-trust than over-trust a value. "mean" averages
                instead.
            **paddleocr_kwargs: Forwarded as-is to `paddleocr.PaddleOCR(...)`
                for anything not exposed above (e.g. model directories,
                detection thresholds).
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required for PaddleOCREngine. Install it with "
                "`pip install paddleocr paddlepaddle`."
            ) from exc

        if confidence_strategy not in ("min", "mean"):
            raise ValueError(
                f"confidence_strategy must be 'min' or 'mean', got {confidence_strategy!r}"
            )
        self._confidence_strategy: "ConfidenceStrategy" = confidence_strategy

        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,
            **paddleocr_kwargs,
        )

    def read_text(self, image: np.ndarray) -> OCRReadResult:
        try:
            pages = self._ocr.predict(image)
        except Exception:
            logger.exception("PaddleOCR raised while reading a field region")
            raise

        if not pages:
            return OCRReadResult()

        page = pages[0]
        texts = list(_get_result_field(page, "rec_texts") or [])
        scores = list(_get_result_field(page, "rec_scores") or [])

        if not texts:
            return OCRReadResult()

        lines = tuple(
            OCRLine(text=str(text), confidence=float(score))
            for text, score in zip(texts, scores)
        )
        return OCRReadResult(lines=lines)


ConfidenceStrategy = Literal["min", "mean"]


def _get_result_field(page: Any, key: str) -> Any:
    """Read a value from a PaddleOCR result object.

    PaddleOCR's 3.x result objects behave like dicts for the fields we
    need (`rec_texts`, `rec_scores`), but this stays defensive against
    attribute-style access too, since exact result-object internals have
    shifted across PaddleOCR versions before.
    """
    try:
        return page[key]
    except (KeyError, TypeError, IndexError):
        return getattr(page, key, None)


# ---------------------------------------------------------------------------
# Extraction results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldExtractionResult:
    """Outcome of extracting a single field from a single image variant.

    Attributes:
        field_name: Matches `FieldDefinition.name`.
        value: Extracted text. Empty string if nothing was recognized.
        confidence: Confidence in [0, 1]. 0.0 when nothing was recognized
            or when `error` is set.
        error: None on success. Set when the field couldn't be processed
            at all (e.g. its bounding box doesn't overlap the image, or
            the OCR engine raised) -- distinct from "OCR ran but found no
            text", which is a normal, non-error, empty result.
    """

    field_name: str
    value: str
    confidence: float
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """True if extraction ran without error (regardless of whether text was found)."""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "value": self.value,
            "confidence": self.confidence,
            "error": self.error,
        }


@dataclass(frozen=True)
class ImageExtractionResult:
    """All field extraction results for one image variant.

    Attributes:
        variant_id: Matches `ImageVariant.variant_id`.
        fields: Extraction results keyed by field name.
    """

    variant_id: str
    fields: Dict[str, FieldExtractionResult] = field(default_factory=dict)

    def get(self, field_name: str) -> Optional[FieldExtractionResult]:
        """Look up a single field's result by name, or None if not present."""
        return self.fields.get(field_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "fields": {name: result.to_dict() for name, result in self.fields.items()},
        }


# ---------------------------------------------------------------------------
# Image loading / cropping helpers
# ---------------------------------------------------------------------------


def _load_image(source: ImageSource) -> np.ndarray:
    """Normalize any supported image source into an RGB uint8 numpy array."""
    if isinstance(source, np.ndarray):
        image = source
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        elif image.ndim == 3 and image.shape[-1] != 3:
            raise ValueError(f"Unsupported array image shape: {image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image

    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"))

    if isinstance(source, (str, Path)):
        with Image.open(source) as img:
            return np.array(img.convert("RGB"))

    if isinstance(source, bytes):
        with Image.open(io.BytesIO(source)) as img:
            return np.array(img.convert("RGB"))

    raise TypeError(f"Unsupported image source type: {type(source)!r}")


def _crop_to_bounding_box(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Crop `image` to `box`, clipping to image bounds.

    Raises:
        ValueError: If the clipped region has zero area, i.e. `box`
            doesn't actually overlap the image.
    """
    height, width = image.shape[:2]
    x1, y1 = max(0, box.x), max(0, box.y)
    x2, y2 = min(width, box.x2), min(height, box.y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Bounding box {box} does not overlap the image "
            f"(image size: {width}x{height})"
        )

    if (x1, y1, x2, y2) != (box.x, box.y, box.x2, box.y2):
        logger.warning(
            "Bounding box %s was clipped to image bounds (%dx%d)", box, width, height
        )

    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# FieldExtractor
# ---------------------------------------------------------------------------


class FieldExtractor:
    """Extracts a fixed set of fields, at fixed locations, from image variants.

    The set of fields (name + bounding box) is injected at construction
    time; images are supplied per call. This lets the same extractor
    instance be reused across many documents of the same layout, or a new
    instance be built per document type by injecting a different set of
    `FieldDefinition`s.

    Cross-variant reconciliation (deciding which variant's value "wins"
    per field) is intentionally out of scope here -- this class reports
    what each variant individually yields; joining happens later,
    elsewhere.
    """

    def __init__(
        self,
        field_definitions: Sequence[FieldDefinition],
        ocr_engine: OCREngine,
    ) -> None:
        """
        Args:
            field_definitions: The fields to extract and where to find
                them. Must be non-empty; field names must be unique.
            ocr_engine: The OCR backend to use for every field/variant.

        Raises:
            ValueError: If `field_definitions` is empty or contains
                duplicate field names.
        """
        if not field_definitions:
            raise ValueError("field_definitions must contain at least one FieldDefinition")

        names = [f.name for f in field_definitions]
        if len(names) != len(set(names)):
            duplicates = {name for name in names if names.count(name) > 1}
            raise ValueError(f"Duplicate field name(s) in field_definitions: {duplicates}")

        self._field_definitions: Tuple[FieldDefinition, ...] = tuple(field_definitions)
        self._ocr_engine = ocr_engine

    @property
    def field_names(self) -> Tuple[str, ...]:
        """Names of all configured fields, in definition order."""
        return tuple(f.name for f in self._field_definitions)

    def extract_fields(self, image: ImageSource, variant_id: str = "default") -> ImageExtractionResult:
        """Extract every configured field from a single image.

        Args:
            image: The image to extract from (path, bytes, PIL Image, or
                numpy array).
            variant_id: Identifier to tag this image's results with.

        Returns:
            One `ImageExtractionResult` covering every configured field.
            Per-field failures (e.g. an out-of-bounds box) are captured on
            that field's result rather than raised; a whole-image load
            failure is captured on every field's result so the call still
            returns a usable (if all-errored) result instead of raising.
        """
        try:
            pixels = _load_image(image)
        except Exception as exc:
            logger.exception("Failed to load image for variant %r", variant_id)
            error_message = f"Failed to load image: {exc}"
            fields = {
                fd.name: FieldExtractionResult(
                    field_name=fd.name, value="", confidence=0.0, error=error_message
                )
                for fd in self._field_definitions
            }
            return ImageExtractionResult(variant_id=variant_id, fields=fields)

        fields = {fd.name: self._extract_field(pixels, fd) for fd in self._field_definitions}
        return ImageExtractionResult(variant_id=variant_id, fields=fields)

    def extract_fields_batch(self, variants: Sequence[ImageVariant]) -> List[ImageExtractionResult]:
        """Extract every configured field from each of a set of image variants.

        The same fields (as configured at construction time) are extracted
        independently from every variant. No cross-variant merging happens
        here -- callers get one `ImageExtractionResult` per variant back,
        in the same order as `variants`.

        Args:
            variants: The image variants to process (e.g. raw scan,
                denoised, contrast-enhanced -- all crops of the same
                document region).

        Returns:
            One `ImageExtractionResult` per input variant, same order.
        """
        if not variants:
            raise ValueError("variants must contain at least one ImageVariant")

        return [self.extract_fields(v.image, variant_id=v.variant_id) for v in variants]

    def _extract_field(self, image: np.ndarray, field_def: FieldDefinition) -> FieldExtractionResult:
        """Extract a single field from an already-loaded image array."""
        try:
            crop = _crop_to_bounding_box(image, field_def.bounding_box)
        except ValueError as exc:
            return FieldExtractionResult(
                field_name=field_def.name, value="", confidence=0.0, error=str(exc)
            )

        try:
            read_result = self._ocr_engine.read_text(crop)
        except Exception as exc:
            logger.exception("OCR engine failed on field %r", field_def.name)
            return FieldExtractionResult(
                field_name=field_def.name, value="", confidence=0.0, error=str(exc)
            )

        value, confidence = self._combine_lines(read_result, field_def)
        return FieldExtractionResult(field_name=field_def.name, value=value, confidence=confidence)

    def _combine_lines(
        self, read_result: OCRReadResult, field_def: FieldDefinition
    ) -> Tuple[str, float]:
        """Combine an engine's raw line-level output into one value + confidence.

        Multi-line fields join lines with newlines; single-line fields join
        with spaces (in case one visual line was split into more than one
        detection). Confidence aggregation strategy comes from the OCR
        engine's configuration (see `PaddleOCREngine.confidence_strategy`).
        """
        non_empty = [line for line in read_result.lines if line.text.strip()]
        if not non_empty:
            return "", 0.0

        separator = "\n" if field_def.multi_line else " "
        value = separator.join(line.text.strip() for line in non_empty)

        scores = [line.confidence for line in non_empty]
        strategy = getattr(self._ocr_engine, "_confidence_strategy", "min")
        confidence = min(scores) if strategy == "min" else sum(scores) / len(scores)
        return value, float(confidence)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal runnable example. Requires `pip install paddleocr paddlepaddle`.
    # Builds two synthetic "variants" of a fake ID card in memory so this
    # can run without any real document on hand -- swap in real images to
    # try it against actual data.
    logging.basicConfig(level=logging.INFO)

    from PIL import ImageDraw

    def _make_fake_document(text_color: str) -> Image.Image:
        img = Image.new("RGB", (600, 300), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((30, 30), "Jane Doe", fill=text_color)
        draw.text((30, 130), "ID-88214-A", fill=text_color)
        draw.text((30, 230), "12 Elm Street, Springfield", fill=text_color)
        return img

    field_definitions = [
        FieldDefinition(name="full_name", bounding_box=BoundingBox(x=20, y=15, width=250, height=40)),
        FieldDefinition(name="document_id", bounding_box=BoundingBox(x=20, y=115, width=250, height=40)),
        FieldDefinition(
            name="address",
            bounding_box=BoundingBox(x=20, y=215, width=400, height=40),
            multi_line=True,
        ),
    ]

    variants = [
        ImageVariant(variant_id="original", image=_make_fake_document("black")),
        ImageVariant(variant_id="high_contrast", image=_make_fake_document("#111111")),
    ]

    try:
        engine = PaddleOCREngine(lang="en")
    except ImportError as exc:
        print(f"Skipping live demo: {exc}")
    else:
        extractor = FieldExtractor(field_definitions=field_definitions, ocr_engine=engine)
        results = extractor.extract_fields_batch(variants)
        for image_result in results:
            print(f"\nVariant: {image_result.variant_id}")
            for name in extractor.field_names:
                field_result = image_result.get(name)
                print(f"  {name!r}: {field_result.value!r} (confidence={field_result.confidence:.2f})")
