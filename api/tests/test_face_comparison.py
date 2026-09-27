from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from kyc_engine import (
    FaceEmbedding,
    FaceRecognitionResult,
    FaceRecognitionNoFaceError,
)
from kyc_api.face_comparison import (
    FaceComparisonNonFiniteSimilarityError,
    FaceComparisonProbeUnavailable,
    FaceComparisonRecognitionError,
    FaceComparisonReferenceUnavailable,
    FaceComparisonService,
    FaceEmbeddingDimensionMismatchError,
    FaceEmbeddingEmptyError,
    FaceEmbeddingNonFiniteError,
    FaceEmbeddingZeroNormError,
)
from kyc_api.logging import JsonFormatter
from kyc_api.main import create_app

from helpers import FakeCoordinator, settings


class _Store:
    def __init__(self, reference=None, probe=None) -> None:
        self.reference = reference
        self.probe = probe
        self.calls: list[str] = []

    def resolve_face_match_portrait(self, session_id: str):
        self.calls.append(f"reference:{session_id}")
        return self.reference

    def resolve_live_face_artifact(self, session_id: str):
        self.calls.append(f"probe:{session_id}")
        return self.probe


class _Recognizer:
    provider_name = "synthetic"
    model_id = "synthetic-development"

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def encode(self, _image):
        outcome = self.outcomes[self.calls % len(self.outcomes)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return FaceRecognitionResult(FaceEmbedding(np.asarray(outcome)), 1, 0.9)


def _image() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


class FaceComparisonServiceTests(unittest.TestCase):
    def _service(self, outcomes: list[object], *, reference=True, probe=True):
        store = _Store(_image() if reference else None, _image() if probe else None)
        recognizer = _Recognizer(outcomes)
        return FaceComparisonService(session_store=store, recognizer=recognizer), store, recognizer

    def test_valid_embeddings_use_both_eligibility_aware_resolvers(self) -> None:
        service, store, _recognizer = self._service([[1.0, 0.0], [1.0, 0.0]])
        result = service.compare("session")
        self.assertAlmostEqual(1.0, result.similarity)
        self.assertEqual(["reference:session", "probe:session"], store.calls)

    def test_different_embeddings_have_lower_raw_similarity(self) -> None:
        service, _store, _recognizer = self._service([[1.0, 0.0], [0.0, 1.0]])
        self.assertLess(service.compare("session").similarity, 1.0)

    def test_unavailable_reference_or_probe_is_explicit(self) -> None:
        with self.assertRaises(FaceComparisonReferenceUnavailable):
            self._service([[1.0], [1.0]], reference=False)[0].compare("session")
        with self.assertRaises(FaceComparisonProbeUnavailable):
            self._service([[1.0], [1.0]], probe=False)[0].compare("session")

    def test_zero_and_multiple_faces_are_role_aware(self) -> None:
        service, _store, _recognizer = self._service([FaceRecognitionNoFaceError("none"), [1.0]])
        with self.assertRaises(FaceComparisonRecognitionError) as zero:
            service.compare("session")
        self.assertEqual("FACE_COMPARISON_REFERENCE_ZERO_FACES", zero.exception.code)

        from kyc_engine import FaceRecognitionMultipleFacesError

        service, _store, _recognizer = self._service([[1.0], FaceRecognitionMultipleFacesError(2)])
        with self.assertRaises(FaceComparisonRecognitionError) as multiple:
            service.compare("session")
        self.assertEqual("FACE_COMPARISON_PROBE_MULTIPLE_FACES", multiple.exception.code)

    def test_embedding_validation_errors_are_explicit(self) -> None:
        cases = [
            ([[], [1.0]], FaceEmbeddingEmptyError),
            ([[0.0, 0.0], [1.0, 0.0]], FaceEmbeddingZeroNormError),
            ([[np.nan], [1.0]], FaceEmbeddingNonFiniteError),
            ([[1.0], [1.0, 0.0]], FaceEmbeddingDimensionMismatchError),
            ([[1e308, 0.0], [1e308, 0.0]], FaceComparisonNonFiniteSimilarityError),
        ]
        for outcomes, error in cases:
            with self.subTest(error=error.__name__):
                with self.assertRaises(error):
                    self._service(outcomes)[0].compare("session")

    def test_recognizer_failure_and_instance_reuse(self) -> None:
        service, _store, recognizer = self._service([RuntimeError("backend"), [1.0]])
        with self.assertRaises(FaceComparisonRecognitionError):
            service.compare("session")

        service, _store, recognizer = self._service([[1.0], [1.0]])
        service.compare("one")
        service.compare("two")
        self.assertEqual(4, recognizer.calls)

    def test_logs_and_result_exclude_vectors_and_similarity(self) -> None:
        service, _store, _recognizer = self._service([[4.25, 9.75], [4.25, 9.75]])
        records: list[logging.LogRecord] = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("kyc_api.face_comparison")
        handler = _Handler()
        logger.addHandler(handler)
        try:
            result = service.compare("session")
        finally:
            logger.removeHandler(handler)
        rendered = "\n".join(JsonFormatter(environment="test").format(record) for record in records)
        self.assertNotIn("4.25", rendered)
        self.assertNotIn("9.75", rendered)
        self.assertNotIn(str(result.similarity), rendered)
        self.assertEqual({"similarity"}, set(result.__dict__))

    def test_startup_constructs_one_recognizer_and_adds_no_face_route(self) -> None:
        recognizer = _Recognizer([[1.0], [1.0]])
        configured = settings(
            face_recognition_enabled=True,
            face_recognition_model_root="private-models/recognition",
            face_recognition_model_id="development-pack",
        )
        from fastapi.testclient import TestClient

        with patch("kyc_api.main.create_face_recognizer", return_value=recognizer) as factory:
            app = create_app(settings=configured, coordinator=FakeCoordinator())
            with TestClient(app) as client:
                self.assertIs(recognizer, app.state.face_comparison._recognizer)
                self.assertEqual(200, client.get("/openapi.json").status_code)
                self.assertNotIn("/v1/sessions/{session_id}/face-comparison", client.get("/openapi.json").json()["paths"])
        factory.assert_called_once_with(configured)

    def test_lifespan_shuts_down_application_owned_face_dispatcher(self) -> None:
        recognizer = _Recognizer([[1.0], [1.0]])
        app = create_app(
            settings=settings(), coordinator=FakeCoordinator(), face_recognizer=recognizer
        )

        with TestClient(app):
            dispatcher = app.state.face_match_dispatcher

        with self.assertRaises(RuntimeError):
            dispatcher._executor.submit(lambda: None)


if __name__ == "__main__":
    unittest.main()
