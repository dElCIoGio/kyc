from __future__ import annotations

import logging
import unittest

from kyc_engine.coordinator import DocumentCoordinator
from kyc_engine.instrumentation import observe_pipeline_stage


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _ObservedPipeline:
    def process(self, _source):
        with observe_pipeline_stage("intake"):
            pass
        return None


class PipelineInstrumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("kyc_engine.pipeline")
        self.handler = _CaptureHandler()
        self.original_handlers = self.logger.handlers[:]
        self.original_level = self.logger.level
        self.original_propagate = self.logger.propagate
        self.logger.handlers = [self.handler]
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def tearDown(self) -> None:
        self.logger.handlers = self.original_handlers
        self.logger.setLevel(self.original_level)
        self.logger.propagate = self.original_propagate

    def test_completed_stage_emits_started_and_completed_with_duration(self) -> None:
        with observe_pipeline_stage("detection"):
            pass

        self.assertEqual(
            ["pipeline_stage_started", "pipeline_stage_completed"],
            [record.event for record in self.handler.records],
        )
        self.assertEqual("detection", self.handler.records[0].stage)
        self.assertEqual("detection", self.handler.records[1].stage)
        self.assertEqual("success", self.handler.records[1].status)
        self.assertGreaterEqual(self.handler.records[1].duration_ms, 0)

    def test_failed_stage_logs_failure_and_reraises_without_completion(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "synthetic-private-id-123456789"):
            with observe_pipeline_stage("ocr", error_code="OCR_ENGINE_FAILED"):
                raise RuntimeError("synthetic-private-id-123456789")

        self.assertEqual(
            ["pipeline_stage_started", "pipeline_stage_failed"],
            [record.event for record in self.handler.records],
        )
        failure = self.handler.records[-1]
        self.assertEqual("ocr", failure.stage)
        self.assertEqual("failed", failure.status)
        self.assertEqual("OCR_ENGINE_FAILED", failure.error_code)
        self.assertEqual("RuntimeError", failure.exception_type)

    def test_coordinator_binds_side_only_for_that_pipeline_invocation(self) -> None:
        coordinator = DocumentCoordinator(_ObservedPipeline(), _ObservedPipeline())
        coordinator.process(front=b"front", back=b"back")
        with observe_pipeline_stage("standalone"):
            pass

        completed = [
            record for record in self.handler.records
            if record.event == "pipeline_stage_completed"
        ]
        self.assertEqual(["front", "back", None], [getattr(record, "side", None) for record in completed])


if __name__ == "__main__":
    unittest.main()
