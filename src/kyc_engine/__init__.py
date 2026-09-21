"""Offline Angolan ID extraction engine."""

from .contracts import (
    BoundingBox,
    DetectionResult,
    DocumentProfile,
    ExtractedField,
    FieldDefinition,
    InputImage,
    KycExtractionResult,
    OCRCandidate,
    PipelineIssue,
    Point,
    ProcessingStatus,
    Quadrilateral,
)
from .pipeline import KycPipeline
from .onnx_detection import OnnxDetectorManifest, OnnxDocumentDetector
from .ocr import PaddleOcrModelManifest, PaddleOCRTextRecognizer, TextRecognizer

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "DocumentProfile",
    "ExtractedField",
    "FieldDefinition",
    "InputImage",
    "KycExtractionResult",
    "KycPipeline",
    "OCRCandidate",
    "OnnxDetectorManifest",
    "OnnxDocumentDetector",
    "PaddleOcrModelManifest",
    "PaddleOCRTextRecognizer",
    "PipelineIssue",
    "Point",
    "ProcessingStatus",
    "Quadrilateral",
    "TextRecognizer",
]
