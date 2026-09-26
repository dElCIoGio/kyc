from __future__ import annotations

import unittest

import numpy as np

from kyc_engine import (
    AntiSpoofResult,
    LivenessEvaluationConfig,
    LivenessEvaluator,
)


class FakeAntiSpoofDetector:
    def __init__(self, results: list[AntiSpoofResult]) -> None:
        self._results = iter(results)
        self.frames: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> AntiSpoofResult:
        self.frames.append(frame)
        return next(self._results)


class LivenessEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = [np.full((2, 2), value, dtype=np.uint8) for value in (1, 2, 3, 4)]

    def test_all_real_frames_pass_and_preserve_input_order(self) -> None:
        detector = FakeAntiSpoofDetector(
            [
                AntiSpoofResult(True, 0.8),
                AntiSpoofResult(True, 0.9),
                AntiSpoofResult(True, 1.0),
            ]
        )
        result = LivenessEvaluator(detector).evaluate(self.frames)

        self.assertTrue(result.passed)
        self.assertEqual(3, result.frames_evaluated)
        self.assertEqual(3, result.real_frames)
        self.assertAlmostEqual(0.9, result.passive_score)
        self.assertEqual([1, 2, 3], [int(frame[0, 0]) for frame in detector.frames])

    def test_all_spoof_frames_fail(self) -> None:
        result = LivenessEvaluator(
            FakeAntiSpoofDetector(
                [
                    AntiSpoofResult(False, 0.1),
                    AntiSpoofResult(False, 0.2),
                    AntiSpoofResult(False, 0.3),
                ]
            )
        ).evaluate(self.frames)

        self.assertFalse(result.passed)
        self.assertEqual(0, result.real_frames)
        self.assertAlmostEqual(0.2, result.passive_score)

    def test_mixed_frames_apply_the_configured_real_frame_threshold(self) -> None:
        two_real = [
            AntiSpoofResult(True, 0.9),
            AntiSpoofResult(False, 0.1),
            AntiSpoofResult(True, 0.8),
        ]
        self.assertTrue(LivenessEvaluator(FakeAntiSpoofDetector(two_real)).evaluate(self.frames).passed)

        stricter = LivenessEvaluationConfig(frame_count=3, minimum_real_ratio=1.0)
        self.assertFalse(
            LivenessEvaluator(FakeAntiSpoofDetector(two_real), stricter).evaluate(self.frames).passed
        )

    def test_zero_and_insufficient_frames_fail_deterministically(self) -> None:
        empty = LivenessEvaluator(FakeAntiSpoofDetector([])).evaluate(())
        self.assertEqual((False, 0.0, 0, 0), (empty.passed, empty.passive_score, empty.frames_evaluated, empty.real_frames))

        insufficient = LivenessEvaluator(
            FakeAntiSpoofDetector([AntiSpoofResult(True, 0.8), AntiSpoofResult(True, 0.9)])
        ).evaluate(self.frames[:2])
        self.assertEqual((False, 2, 2), (insufficient.passed, insufficient.frames_evaluated, insufficient.real_frames))

    def test_configuration_and_contract_values_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            LivenessEvaluationConfig(frame_count=0)
        with self.assertRaises(ValueError):
            LivenessEvaluationConfig(minimum_real_ratio=0.0)
        with self.assertRaises(ValueError):
            AntiSpoofResult(True, 1.1)
        with self.assertRaises(ValueError):
            AntiSpoofResult(True, float("nan"))

    def test_detector_failure_is_not_converted_to_a_liveness_result(self) -> None:
        class FailingDetector:
            def detect(self, frame: np.ndarray) -> AntiSpoofResult:
                raise RuntimeError("detector unavailable")

        with self.assertRaisesRegex(RuntimeError, "detector unavailable"):
            LivenessEvaluator(FailingDetector()).evaluate(self.frames)


if __name__ == "__main__":
    unittest.main()
