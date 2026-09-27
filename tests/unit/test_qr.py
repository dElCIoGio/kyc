from __future__ import annotations

import unittest

import cv2
import numpy as np

from kyc_engine.contracts import BoundingBox, QrCodeDefinition, QrCodeStatus, VariantBatch, VariantInfo
from kyc_engine.qr import QrCodeExtractor, _preferred_variants, parse_qr_payload
from kyc_engine.variants import BalancedVariantPolicy


PAYLOAD = (
    "MARIA SILVA 000000000LA000 LUANDA 04/01/2002 FEMININO SOLTEIRA "
    "05/07/2023 04/07/2033 LUANDA V01"
)
UNICODE_PAYLOAD = (
    "JOSÉ ÁGUA 000000000LA000 LUANDA 04/01/2002 FEMININO SOLTEIRA "
    "05/07/2023 04/07/2033 LUANDA V01"
)


class QrPayloadTests(unittest.TestCase):
    def test_parses_structured_payload_with_multiword_name(self) -> None:
        data = parse_qr_payload(PAYLOAD)
        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual("MARIA SILVA", data.full_name)
        self.assertEqual("000000000LA000", data.id_number)
        self.assertEqual("LUANDA", data.birth_province)
        self.assertEqual("FEMININO", data.sex)
        self.assertEqual("SOLTEIRA", data.marital_status)
        self.assertEqual("LUANDA", data.issuing_province)
        self.assertEqual("01", data.version)

    def test_rejects_malformed_payloads(self) -> None:
        self.assertIsNone(parse_qr_payload(PAYLOAD.removesuffix(" V01")))
        self.assertIsNone(parse_qr_payload(PAYLOAD.replace("FEMININO", "OUTRO")))
        self.assertIsNone(parse_qr_payload(PAYLOAD.replace("04/07/2033", "invalid")))


class QrExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = QrCodeDefinition(BoundingBox(0, 0, 128, 128))

    def test_decodes_synthetic_qr_and_preserves_source(self) -> None:
        encoded = cv2.QRCodeEncoder_create().encode(PAYLOAD)
        image = cv2.resize(encoded, (128, 128), interpolation=cv2.INTER_NEAREST)
        source = image.copy()
        result = QrCodeExtractor().extract(self._batches(image), self.definition)

        self.assertEqual(QrCodeStatus.DECODED, result.status)
        self.assertEqual(PAYLOAD, result.raw_payload)
        self.assertEqual("MARIA SILVA", result.data.full_name if result.data else None)
        self.assertEqual("original", result.variant_name)
        self.assertTrue(np.array_equal(image, source))

    def test_uses_utf8_barcode_bytes_and_preserves_payload_line_breaks(self) -> None:
        raw_payload = UNICODE_PAYLOAD.replace(" FEMININO", "\nFEMININO")
        decoder = _FakeDecoder({0: ("corrupted fallback", raw_payload.encode("utf-8"))})
        image = np.zeros((128, 128), dtype=np.uint8)
        result = QrCodeExtractor(decoder).extract(self._batches(image), self.definition)

        self.assertEqual(QrCodeStatus.DECODED, result.status)
        self.assertEqual(raw_payload, result.raw_payload)
        self.assertEqual("JOSÉ ÁGUA", result.data.full_name if result.data else None)

    def test_repairs_mojibake_only_when_raw_bytes_are_unavailable(self) -> None:
        mojibake = UNICODE_PAYLOAD.encode("utf-8").decode("latin-1")
        decoder = _FakeDecoder({0: mojibake})
        result = QrCodeExtractor(decoder).extract(
            self._batches(np.zeros((128, 128), dtype=np.uint8)), self.definition
        )
        self.assertEqual(UNICODE_PAYLOAD, result.raw_payload)
        self.assertEqual("JOSÉ ÁGUA", result.data.full_name if result.data else None)

    def test_uses_preferred_variant_order(self) -> None:
        decoder = _FakeDecoder({1: None, 2: PAYLOAD})
        original = np.full((128, 128), 1, dtype=np.uint8)
        grayscale = np.full((128, 128), 2, dtype=np.uint8)
        result = QrCodeExtractor(decoder).extract(
            (
                VariantBatch("original", (VariantInfo("original", original, {}),)),
                VariantBatch("grayscale", (VariantInfo("grayscale", grayscale, {}),)),
            ),
            self.definition,
        )
        self.assertEqual(QrCodeStatus.DECODED, result.status)
        self.assertEqual("grayscale", result.variant_name)
        self.assertEqual([1, 2], decoder.calls)

    def test_distinguishes_decode_and_parse_failures(self) -> None:
        image = np.zeros((128, 128), dtype=np.uint8)
        parse_failed = QrCodeExtractor(_FakeDecoder({0: "INVALID V01"})).extract(
            self._batches(image), self.definition
        )
        self.assertEqual(QrCodeStatus.PARSE_FAILED, parse_failed.status)
        self.assertEqual("INVALID V01", parse_failed.raw_payload)

        not_detected = QrCodeExtractor(_FakeDecoder({0: None})).extract(
            self._batches(image), self.definition
        )
        self.assertEqual(QrCodeStatus.NOT_DETECTED, not_detected.status)

    def test_preferred_variant_names_are_generated_by_balanced_policy(self) -> None:
        image = np.zeros((32, 32), dtype=np.uint8)
        batches = BalancedVariantPolicy().generate(image)
        generated_names = {variant.name for batch in batches for variant in batch.variants}
        preferred_names = tuple(variant.name for variant in _preferred_variants(batches))
        self.assertEqual(
            ("original", "grayscale", "clahe", "gamma_0_75", "gamma_1_25"),
            preferred_names,
        )
        self.assertTrue(set(preferred_names).issubset(generated_names))

    @staticmethod
    def _batches(image: np.ndarray) -> tuple[VariantBatch, ...]:
        return (VariantBatch("original", (VariantInfo("original", image, {}),)),)


class _Barcode:
    def __init__(self, text: str, raw_bytes: bytes | None = None) -> None:
        self.text = text
        if raw_bytes is not None:
            self.bytes = raw_bytes


class _FakeDecoder:
    class BarcodeFormat:
        QRCode = object()

    __version__ = "test"

    def __init__(self, values: dict[int, str | tuple[str, bytes] | None]) -> None:
        self.values = values
        self.calls: list[int] = []

    def read_barcode(self, image: np.ndarray, *, formats: object) -> _Barcode | None:
        key = int(image[0, 0])
        self.calls.append(key)
        value = self.values.get(key)
        if isinstance(value, tuple):
            return _Barcode(*value)
        return _Barcode(value) if value is not None else None


if __name__ == "__main__":
    unittest.main()
