from __future__ import annotations


class MiniFASNetError(RuntimeError):
    """Base error for the optional MiniFASNet adapter."""

    code = "MINIFASNET_ERROR"


class MiniFASNetInitializationError(MiniFASNetError):
    """Raised when trusted local model resources cannot be initialized."""

    code = "MINIFASNET_INITIALIZATION_FAILED"


class MiniFASNetInputError(MiniFASNetError):
    """Raised when one decoded frame cannot be evaluated."""

    code = "MINIFASNET_INVALID_FRAME"


class MiniFASNetNoFaceError(MiniFASNetInputError):
    """Raised when the required face-region detector has no usable region."""

    code = "MINIFASNET_NO_FACE"
