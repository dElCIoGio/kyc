"""Model-agnostic internal face-recognition error contracts."""

from __future__ import annotations


class FaceRecognitionError(RuntimeError):
    """Base error for local face-recognition adapters."""

    code = "FACE_RECOGNITION_FAILED"


class FaceRecognitionInitializationError(FaceRecognitionError):
    """Configured local recognition resources could not be initialized."""

    code = "FACE_RECOGNITION_INITIALIZATION_FAILED"


class FaceRecognitionInferenceError(FaceRecognitionError):
    """One image could not complete model-managed recognition inference."""

    code = "FACE_RECOGNITION_INFERENCE_FAILED"


class FaceRecognitionNoFaceError(FaceRecognitionInferenceError):
    """Recognition detection found no faces."""

    code = "FACE_RECOGNITION_ZERO_FACES"
    face_count = 0


class FaceRecognitionMultipleFacesError(FaceRecognitionInferenceError):
    """Recognition detection found more than one face and was rejected."""

    code = "FACE_RECOGNITION_MULTIPLE_FACES"

    def __init__(self, face_count: int) -> None:
        super().__init__("Recognition requires exactly one detected face")
        self.face_count = face_count
