import unittest

import numpy as np

from kyc_engine.contracts import BoundingBox, FieldCrop
from kyc_engine.ocr import PaddleOCRTextRecognizer


class FakeRecognition:
    def predict(self, images):
        return tuple(
            {"rec_text": f"line-{index}", "rec_score": 0.8 + index * 0.01}
            for index, _ in enumerate(images)
        )


class FakeOCR:
    def predict(self, images):
        return tuple(
            {"rec_texts": ["first", "second"], "rec_scores": [0.9, 0.7]}
            for _ in images
        )


class PaddleStyleRecognitionResult:
    def __init__(self, text: str, score: float) -> None:
        self.rec_text = text
        self.rec_score = score


class GeneratorRecognition:
    def predict(self, images):
        return (
            PaddleStyleRecognitionResult(f"object-{index}", 0.85)
            for index, _ in enumerate(images)
        )


class IsolatedFailureRecognition:
    def predict(self, images):
        if len(images) > 1:
            raise RuntimeError("batch rejected")
        if int(images[0][0, 0, 0]) == 200:
            raise RuntimeError("crop rejected")
        return (PaddleStyleRecognitionResult("recovered", 0.91),)


class CaptureRecognition:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def predict(self, images):
        self.shapes.extend(tuple(image.shape) for image in images)
        return tuple({"rec_text": "ok", "rec_score": 0.9} for _ in images)


class OcrAdapterTests(unittest.TestCase):
    def test_routes_modes_in_batches_and_preserves_input_order(self) -> None:
        recognizer = PaddleOCRTextRecognizer.__new__(PaddleOCRTextRecognizer)
        recognizer.model_version = "test"
        recognizer._recognition = FakeRecognition()
        recognizer._ocr = FakeOCR()
        crops = (
            self._crop("single-a", "single_line"),
            self._crop("multi", "multiline"),
            self._crop("single-b", "single_line"),
        )
        originals = tuple(crop.image.copy() for crop in crops)
        results = recognizer.recognize_batch(crops)
        self.assertEqual(("line-0", "first\nsecond", "line-1"), tuple(item.raw_value for item in results))
        self.assertEqual((0.8, 0.7, 0.81), tuple(item.confidence for item in results))
        self.assertTrue(all(np.array_equal(crop.image, original) for crop, original in zip(crops, originals)))

    def test_unsupported_mode_returns_structured_failure(self) -> None:
        recognizer = PaddleOCRTextRecognizer.__new__(PaddleOCRTextRecognizer)
        recognizer.model_version = "test"
        recognizer._recognition = FakeRecognition()
        recognizer._ocr = FakeOCR()
        result = recognizer.recognize_batch((self._crop("field", "unsupported"),))[0]
        self.assertEqual("UNSUPPORTED_OCR_MODE", result.error_code)

    def test_accepts_iterable_paddle_style_results(self) -> None:
        recognizer = PaddleOCRTextRecognizer.__new__(PaddleOCRTextRecognizer)
        recognizer.model_version = "test"
        recognizer._recognition = GeneratorRecognition()
        recognizer._ocr = FakeOCR()
        results = recognizer.recognize_batch(
            (self._crop("first", "single_line"), self._crop("second", "single_line"))
        )
        self.assertEqual(("object-0", "object-1"), tuple(item.raw_value for item in results))

    def test_retries_individual_crops_after_batch_failure(self) -> None:
        recognizer = PaddleOCRTextRecognizer.__new__(PaddleOCRTextRecognizer)
        recognizer.model_version = "test"
        recognizer._recognition = IsolatedFailureRecognition()
        recognizer._ocr = FakeOCR()
        crops = (
            self._crop("recoverable", "single_line", fill=100),
            self._crop("failed", "single_line", fill=200),
        )
        results = recognizer.recognize_batch(crops)
        self.assertEqual("recovered", results[0].raw_value)
        self.assertIsNone(results[0].error_code)
        self.assertEqual("OCR_FAILED", results[1].error_code)

    def test_converts_supported_crop_channels_to_bgr_for_paddle(self) -> None:
        recognizer = PaddleOCRTextRecognizer.__new__(PaddleOCRTextRecognizer)
        recognizer.model_version = "test"
        capture = CaptureRecognition()
        recognizer._recognition = capture
        recognizer._ocr = FakeOCR()
        crops = (
            self._crop("gray", "single_line", image=np.full((10, 30), 100, dtype=np.uint8)),
            self._crop("single", "single_line", image=np.full((10, 30, 1), 100, dtype=np.uint8)),
            self._crop("bgr", "single_line", image=np.full((10, 30, 3), 100, dtype=np.uint8)),
            self._crop("bgra", "single_line", image=np.full((10, 30, 4), 100, dtype=np.uint8)),
        )

        recognizer.recognize_batch(crops)

        self.assertEqual([(10, 30, 3)] * 4, capture.shapes)

    @staticmethod
    def _crop(
        name: str,
        mode: str,
        *,
        fill: int = 100,
        image: np.ndarray | None = None,
    ) -> FieldCrop:
        return FieldCrop(
            field_name=name,
            generator="original",
            variant_name="original",
            image=image if image is not None else np.full((10, 30, 3), fill, dtype=np.uint8),
            bounding_box=BoundingBox(0, 0, 30, 10),
            ocr_mode=mode,
        )


if __name__ == "__main__":
    unittest.main()
