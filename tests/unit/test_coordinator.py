import unittest

import numpy as np

from kyc_engine.coordinator import DocumentCoordinator
from kyc_engine.contracts import (
    KycExtractionResult,
    ProcessingStatus,
)


def result(side: str, status: ProcessingStatus = ProcessingStatus.SUCCESS) -> KycExtractionResult:
    return KycExtractionResult(
        schema_version="1.1",
        processing_id=f"{side}-id",
        status=status,
        document_type="ao_id_card",
        side=side,
        profile_id=f"ao_id_card/{side}/v1",
        detection=None,
        fields={},
        issues=(),
        timings_ms={"ocr": 1.0},
    )


class FakePipeline:
    def __init__(self, side: str, status: ProcessingStatus = ProcessingStatus.SUCCESS, *, fail=False, events=None):
        self.side = side
        self.value = result(side, status)
        self.fail = fail
        self.sources = []
        self.events = events

    def process(self, source):
        if self.events is not None:
            self.events.append(self.side)
        self.sources.append(source)
        if self.fail:
            raise RuntimeError("private backend detail")
        return self.value


class DocumentCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.front = FakePipeline("front")
        self.back = FakePipeline("back")
        self.coordinator = DocumentCoordinator(self.front, self.back)

    def test_runs_both_sides_in_stable_order_and_preserves_results(self):
        front_source = np.zeros((2, 2), dtype=np.uint8)
        back_source = np.ones((2, 2), dtype=np.uint8)
        output = self.coordinator.process(front=front_source, back=back_source)
        self.assertEqual(output.status, ProcessingStatus.SUCCESS)
        self.assertIs(output.front, self.front.value)
        self.assertIs(output.back, self.back.value)
        self.assertEqual(self.front.sources, [front_source])
        self.assertEqual(self.back.sources, [back_source])

    def test_runs_front_before_back(self):
        events = []
        coordinator = DocumentCoordinator(
            FakePipeline("front", events=events),
            FakePipeline("back", events=events),
        )
        coordinator.process(front=b"front", back=b"back")
        self.assertEqual(events, ["front", "back"])

    def test_missing_front_is_partial(self):
        output = self.coordinator.process(back=b"back")
        self.assertEqual(output.status, ProcessingStatus.PARTIAL)
        self.assertIsNone(output.front)
        self.assertEqual(output.back, self.back.value)
        self.assertEqual(output.issues[0].code, "SIDE_MISSING")

    def test_missing_back_is_partial(self):
        output = self.coordinator.process(front=b"front")
        self.assertEqual(output.status, ProcessingStatus.PARTIAL)
        self.assertIsNone(output.back)
        self.assertEqual(output.issues[0].code, "SIDE_MISSING")

    def test_partial_side_remains_partial(self):
        self.back.value = result("back", ProcessingStatus.PARTIAL)
        output = self.coordinator.process(front=b"front", back=b"back")
        self.assertEqual(output.status, ProcessingStatus.PARTIAL)

    def test_failed_side_is_contained(self):
        self.back.fail = True
        output = self.coordinator.process(front=b"front", back=b"back")
        self.assertEqual(output.status, ProcessingStatus.PARTIAL)
        self.assertIs(output.front, self.front.value)
        self.assertIsNone(output.back)
        self.assertEqual(output.issues[0].code, "BACK_PROCESSING_FAILED")

    def test_all_failed_is_failed(self):
        self.front.fail = True
        self.back.fail = True
        output = self.coordinator.process(front=b"front", back=b"back")
        self.assertEqual(output.status, ProcessingStatus.FAILED)
        self.assertEqual([issue.code for issue in output.issues], [
            "FRONT_PROCESSING_FAILED", "BACK_PROCESSING_FAILED"
        ])

    def test_no_side_is_rejected(self):
        with self.assertRaises(ValueError):
            self.coordinator.process()

    def test_serializes_nested_results_without_cross_side_data(self):
        output = self.coordinator.process(front=b"front")
        serialized = output.to_dict()
        self.assertEqual(serialized["front"]["processing_id"], "front-id")
        self.assertIsNone(serialized["back"])
        self.assertEqual(serialized["status"], "partial")
        self.assertIn("timings_ms", serialized)


if __name__ == "__main__":
    unittest.main()
