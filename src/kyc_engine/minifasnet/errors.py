from __future__ import annotations

from kyc_engine.liveness import LivenessFrameError, LivenessNoFaceError


class MiniFASNetError(RuntimeError):
    """Base error for the optional MiniFASNet adapter."""

    code = "MINIFASNET_ERROR"


class MiniFASNetInitializationError(MiniFASNetError):
    """Raised when trusted local model resources cannot be initialized."""

    code = "MINIFASNET_INITIALIZATION_FAILED"


class MiniFASNetInputError(MiniFASNetError, LivenessFrameError):
    """Raised when one decoded frame cannot be evaluated."""

    code = "MINIFASNET_INVALID_FRAME"


class MiniFASNetNoFaceError(MiniFASNetInputError, LivenessNoFaceError):
    """Raised when the required face-region detector has no usable region."""

    code = "MINIFASNET_NO_FACE"
