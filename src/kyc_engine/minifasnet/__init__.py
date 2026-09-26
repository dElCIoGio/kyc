"""Optional PyTorch MiniFASNet passive anti-spoof adapter."""

from .adapter import MiniFASNetAntiSpoofDetector
from .checkpoint import MiniFASNetModelSpec, discover_models, parse_model_filename
from .errors import (
    MiniFASNetError,
    MiniFASNetInitializationError,
    MiniFASNetInputError,
    MiniFASNetNoFaceError,
)

__all__ = [
    "MiniFASNetAntiSpoofDetector",
    "MiniFASNetError",
    "MiniFASNetInitializationError",
    "MiniFASNetInputError",
    "MiniFASNetModelSpec",
    "MiniFASNetNoFaceError",
    "discover_models",
    "parse_model_filename",
]
