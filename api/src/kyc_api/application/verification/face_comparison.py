"""Internal, ephemeral face-embedding comparison service."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Literal

from kyc_engine import (
    FaceComparisonResult,
    FaceEmbedding,
    FaceRecognitionError,
    FaceRecognitionMultipleFacesError,
    FaceRecognitionNoFaceError,
    FaceRecognizer,
)
from kyc_engine.face_comparison import (
    FaceComparisonError,
    FaceComparisonNonFiniteSimilarityError,
    FaceEmbeddingDimensionMismatchError,
    FaceEmbeddingEmptyError,
    FaceEmbeddingNonFiniteError,
    FaceEmbeddingValidationError,
    FaceEmbeddingZeroNormError,
    compare_face_embeddings,
    validate_face_embedding,
)

from ...infrastructure.observability.metrics import MetricsRegistry
from ..sessions.store import SessionStore


logger = logging.getLogger("kyc_api.face_comparison")
_Role = Literal["reference", "probe"]


class FaceComparisonReferenceUnavailable(FaceComparisonError):
    code = "FACE_COMPARISON_REFERENCE_UNAVAILABLE"


class FaceComparisonProbeUnavailable(FaceComparisonError):
    code = "FACE_COMPARISON_PROBE_UNAVAILABLE"


class FaceComparisonRecognitionError(FaceComparisonError):
    def __init__(self, role: _Role, code: str, face_count: int | None = None) -> None:
        super().__init__(f"{role.capitalize()} face recognition failed")
        self.code = code
        self.role = role
        self.face_count = face_count


class FaceComparisonService:
    """Resolve eligible session artifacts and compare ephemeral embeddings only."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        recognizer: FaceRecognizer,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._session_store = session_store
        self._recognizer = recognizer
        self._metrics = metrics

    def compare(self, session_id: str) -> FaceComparisonResult:
        reference = self._session_store.resolve_face_match_portrait(session_id)
        if reference is None:
            raise FaceComparisonReferenceUnavailable("Eligible document portrait is unavailable")
        probe = self._session_store.resolve_live_face_artifact(session_id)
        if probe is None:
            raise FaceComparisonProbeUnavailable("Eligible liveness frame is unavailable")

        reference_embedding = self._encode("reference", reference)
        probe_embedding = self._encode("probe", probe)
        started = perf_counter()
        try:
            result = compare_face_embeddings(reference_embedding, probe_embedding)
        except FaceComparisonError:
            self._record("comparison", "failed", perf_counter() - started)
            raise
        self._record("comparison", "success", perf_counter() - started)
        return result

    def _encode(self, role: _Role, image) -> FaceEmbedding:
        started = perf_counter()
        try:
            result = self._recognizer.encode(image)
            embedding = validate_face_embedding(result.embedding)
        except FaceRecognitionNoFaceError as exc:
            self._record(role, "failed", perf_counter() - started, face_count=0, error_code="ZERO_FACES")
            raise FaceComparisonRecognitionError(role, f"FACE_COMPARISON_{role.upper()}_ZERO_FACES", 0) from exc
        except FaceRecognitionMultipleFacesError as exc:
            face_count = getattr(exc, "face_count", None)
            self._record(role, "failed", perf_counter() - started, face_count=face_count, error_code="MULTIPLE_FACES")
            raise FaceComparisonRecognitionError(
                role, f"FACE_COMPARISON_{role.upper()}_MULTIPLE_FACES", face_count
            ) from exc
        except FaceRecognitionError as exc:
            self._record(role, "failed", perf_counter() - started, error_code=exc.code)
            raise FaceComparisonRecognitionError(
                role, f"FACE_COMPARISON_{role.upper()}_RECOGNITION_FAILED"
            ) from exc
        except FaceComparisonError as exc:
            self._record(role, "failed", perf_counter() - started, error_code=exc.code)
            raise
        except Exception as exc:
            self._record(role, "failed", perf_counter() - started, error_code="INVALID_RESULT")
            raise FaceComparisonRecognitionError(
                role, f"FACE_COMPARISON_{role.upper()}_RECOGNITION_FAILED"
            ) from exc
        self._record(role, "success", perf_counter() - started, face_count=result.face_count)
        return embedding

    def _record(
        self,
        stage: str,
        status: str,
        duration_seconds: float,
        *,
        face_count: int | None = None,
        error_code: str | None = None,
    ) -> None:
        duration_ms = round(max(0.0, duration_seconds) * 1000.0, 3)
        extra = {
            "event": "face_comparison.stage_completed",
            "stage": stage,
            "status": status,
            "duration_ms": duration_ms,
            "provider": getattr(self._recognizer, "provider_name", "unknown"),
            "model_id": getattr(self._recognizer, "model_id", "unknown"),
        }
        if face_count is not None:
            extra["face_count"] = face_count
        if error_code is not None:
            extra["error_code"] = error_code
        logger.info("face comparison stage completed", extra=extra)
        if self._metrics is not None:
            self._metrics.record_face_comparison(
                stage=stage, status=status, duration_seconds=duration_seconds
            )
