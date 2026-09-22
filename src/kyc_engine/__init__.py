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
    QrCodeResult,
    QrCodeStatus,
    QrIdentityData,
    Quadrilateral,
)
from .pipeline import KycPipeline
from .coordinator import DocumentCoordinator, DocumentExtractionResult
from .onnx_detection import OnnxDetectorManifest, OnnxDocumentDetector
from .ocr import PaddleOcrModelManifest, PaddleOCRTextRecognizer, TextRecognizer
from .qr import QrCodeExtractor

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "DocumentProfile",
    "ExtractedField",
    "FieldDefinition",
    "InputImage",
    "KycExtractionResult",
    "KycPipeline",
    "DocumentCoordinator",
    "DocumentExtractionResult",
    "OCRCandidate",
    "OnnxDetectorManifest",
    "OnnxDocumentDetector",
    "PaddleOcrModelManifest",
    "PaddleOCRTextRecognizer",
    "PipelineIssue",
    "Point",
    "ProcessingStatus",
    "QrCodeExtractor",
    "QrCodeResult",
    "QrCodeStatus",
    "QrIdentityData",
    "Quadrilateral",
    "TextRecognizer",
]
