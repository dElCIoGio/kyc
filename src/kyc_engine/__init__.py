"""Offline Angolan ID extraction engine."""

from .contracts import (
    BoundingBox,
    DetectionResult,
    DocumentProfile,
    ExtractedField,
    FieldDefinition,
    FieldStatus,
    ImageSource,
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
from .intake import IntakeLimits
from .pipeline import KycPipeline
from .coordinator import DocumentCoordinator, DocumentExtractionResult
from .defaults import build_paddle_document_coordinator
from .onnx_detection import OnnxDetectorManifest, OnnxDocumentDetector
from .ocr import PaddleOcrModelManifest, PaddleOCRTextRecognizer, TextRecognizer
from .qr import QrCodeExtractor

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "DocumentProfile",
    "ExtractedField",
    "FieldDefinition",
    "FieldStatus",
    "ImageSource",
    "InputImage",
    "IntakeLimits",
    "KycExtractionResult",
    "KycPipeline",
    "build_paddle_document_coordinator",
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
