"""Shared internal primitives for raw face-embedding comparison.

These functions deliberately have no session, persistence, logging, or decision
semantics.  They are shared by the runtime session service and offline
evaluation so both use the exact same embedding validation and cosine math.
"""

from __future__ import annotations

import numpy as np

from .contracts import FaceComparisonResult, FaceEmbedding


class FaceComparisonError(RuntimeError):
    """Base error for internal raw face comparison."""

    code = "FACE_COMPARISON_FAILED"


class FaceEmbeddingValidationError(FaceComparisonError):
    code = "FACE_COMPARISON_INVALID_EMBEDDING"


class FaceEmbeddingEmptyError(FaceEmbeddingValidationError):
    code = "FACE_COMPARISON_EMPTY_EMBEDDING"


class FaceEmbeddingZeroNormError(FaceEmbeddingValidationError):
    code = "FACE_COMPARISON_ZERO_NORM_EMBEDDING"


class FaceEmbeddingNonFiniteError(FaceEmbeddingValidationError):
    code = "FACE_COMPARISON_NONFINITE_EMBEDDING"


class FaceEmbeddingDimensionMismatchError(FaceEmbeddingValidationError):
    code = "FACE_COMPARISON_EMBEDDING_DIMENSION_MISMATCH"


class FaceComparisonNonFiniteSimilarityError(FaceComparisonError):
    code = "FACE_COMPARISON_NONFINITE_SIMILARITY"


def validate_face_embedding(embedding: FaceEmbedding) -> FaceEmbedding:
    """Reject vectors that cannot safely participate in cosine comparison."""

    vector = np.asarray(embedding.vector, dtype=np.float64)
    if vector.ndim != 1:
        raise FaceEmbeddingValidationError("Face embedding must be one-dimensional")
    if vector.size == 0:
        raise FaceEmbeddingEmptyError("Face embedding is empty")
    if not np.all(np.isfinite(vector)):
        raise FaceEmbeddingNonFiniteError("Face embedding is non-finite")
    return embedding


def cosine_similarity(reference: FaceEmbedding, probe: FaceEmbedding) -> float:
    """Return finite raw cosine similarity without applying a decision threshold."""

    reference_vector = np.asarray(reference.vector, dtype=np.float64)
    probe_vector = np.asarray(probe.vector, dtype=np.float64)
    if reference_vector.shape != probe_vector.shape:
        raise FaceEmbeddingDimensionMismatchError("Face embedding dimensions differ")
    reference_norm = float(np.linalg.norm(reference_vector))
    probe_norm = float(np.linalg.norm(probe_vector))
    if reference_norm == 0.0 or probe_norm == 0.0:
        raise FaceEmbeddingZeroNormError("Face embedding has zero norm")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        similarity = float(np.dot(reference_vector, probe_vector) / (reference_norm * probe_norm))
    if not np.isfinite(similarity):
        raise FaceComparisonNonFiniteSimilarityError("Face comparison similarity is non-finite")
    return similarity


def compare_face_embeddings(reference: FaceEmbedding, probe: FaceEmbedding) -> FaceComparisonResult:
    """Validate two transient embeddings and return their raw cosine similarity."""

    return FaceComparisonResult(
        similarity=cosine_similarity(validate_face_embedding(reference), validate_face_embedding(probe))
    )
