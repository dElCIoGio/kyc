from __future__ import annotations

import unittest

import cv2
import numpy as np

from kyc_engine import (
    AntiSpoofResult,
    BoundingBox,
    FaceCandidate,
    FaceDetectionResult,
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


class FakeFaceDetector:
    def __init__(self, candidates: list[tuple[FaceCandidate, ...]]) -> None:
        self._candidates = iter(candidates)
        self.frames: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> FaceDetectionResult:
        self.frames.append(frame)
        return FaceDetectionResult(next(self._candidates))


def _frame(*, blurred: bool = False) -> np.ndarray:
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    checker = ((np.indices((80, 80)).sum(axis=0) // 8) % 2 * 180 + 35)
    checker = checker.astype(np.uint8)
    if blurred:
        checker = cv2.GaussianBlur(checker, (9, 9), 2)
    frame[60:140, 60:140] = np.repeat(checker[:, :, None], 3, axis=2)
    return frame


def _face(x: int = 60, y: int = 60, width: int = 80, height: int = 80) -> FaceCandidate:
    return FaceCandidate(BoundingBox(x, y, width, height))


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

    def test_no_eligible_live_face_keeps_the_passive_result_unchanged(self) -> None:
        evaluator = LivenessEvaluator(
            FakeAntiSpoofDetector([AntiSpoofResult(True, 0.9)] * 3),
            face_detector=FakeFaceDetector([(), (), ()]),
        )
        evaluation = evaluator.evaluate_with_live_face([_frame()] * 3)

        self.assertTrue(evaluation.result.passed)
        self.assertEqual(3, evaluation.live_face.candidate_frame_count)
        self.assertEqual(0, evaluation.live_face.eligible_frame_count)
        self.assertIsNone(evaluation.live_face.selected_frame_index)
        self.assertEqual("no_eligible_frame", evaluation.live_face.selection_outcome)

    def test_one_eligible_live_face_returns_a_readonly_crop(self) -> None:
        evaluator = LivenessEvaluator(
            FakeAntiSpoofDetector([AntiSpoofResult(True, 0.9)] * 3),
            face_detector=FakeFaceDetector([(_face(),), (), ()]),
        )
        evaluation = evaluator.evaluate_with_live_face([_frame()] * 3)

        self.assertEqual(0, evaluation.live_face.selected_frame_index)
        self.assertEqual(1, evaluation.live_face.eligible_frame_count)
        assert evaluation.live_face.face_crop is not None
        self.assertEqual((80, 80, 3), evaluation.live_face.face_crop.shape)
        self.assertFalse(evaluation.live_face.face_crop.flags.writeable)

    def test_several_eligible_frames_choose_the_best_deterministically(self) -> None:
        config = LivenessEvaluationConfig(min_sharpness=0.0)
        evaluator = LivenessEvaluator(
            FakeAntiSpoofDetector([AntiSpoofResult(True, 0.9)] * 3),
            config,
            FakeFaceDetector([(_face(),), (_face(),), (_face(),)]),
        )
        evaluation = evaluator.evaluate_with_live_face([_frame(), _frame(), _frame()])

        self.assertEqual(0, evaluation.live_face.selected_frame_index)
        self.assertEqual(3, evaluation.live_face.eligible_frame_count)

    def test_blurry_high_liveness_frame_loses_to_a_sharper_usable_frame(self) -> None:
        config = LivenessEvaluationConfig(min_sharpness=0.0)
        evaluator = LivenessEvaluator(
            FakeAntiSpoofDetector(
                [AntiSpoofResult(True, 1.0), AntiSpoofResult(True, 0.8), AntiSpoofResult(False, 0.1)]
            ),
            config,
            FakeFaceDetector([(_face(),), (_face(),)]),
        )
        evaluation = evaluator.evaluate_with_live_face([_frame(blurred=True), _frame(), _frame()])

        self.assertTrue(evaluation.result.passed)
        self.assertEqual(1, evaluation.live_face.selected_frame_index)

    def test_multiple_and_too_small_faces_are_rejected(self) -> None:
        multi = LivenessEvaluator(
            FakeAntiSpoofDetector([AntiSpoofResult(True, 0.9)] * 3),
            face_detector=FakeFaceDetector([(_face(), _face(10, 10)), (), ()]),
        ).evaluate_with_live_face([_frame()] * 3)
        self.assertEqual(0, multi.live_face.eligible_frame_count)
        self.assertTrue(multi.live_face.face_detected)
        self.assertEqual(2, multi.live_face.face_count)

        small = LivenessEvaluator(
            FakeAntiSpoofDetector([AntiSpoofResult(True, 0.9)] * 3),
            face_detector=FakeFaceDetector([(_face(85, 85, 30, 30),), (), ()]),
        ).evaluate_with_live_face([_frame()] * 3)
        self.assertEqual(0, small.live_face.eligible_frame_count)
        self.assertEqual("no_eligible_frame", small.live_face.selection_outcome)


if __name__ == "__main__":
    unittest.main()
