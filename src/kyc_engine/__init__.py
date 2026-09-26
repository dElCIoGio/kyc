"""Offline Angolan ID extraction engine."""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .contracts import (
    BoundingBox,
    CaptureAssessment,
    CaptureIssue,
    CaptureIssueCode,
    CaptureMetrics,
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
from .capture import CaptureAssessmentConfig, CaptureAssessmentInputError, DocumentCaptureAssessor
from .intake import IntakeLimits
from .pipeline import KycPipeline
from .coordinator import DocumentCoordinator, DocumentExtractionResult
from .defaults import build_paddle_document_coordinator
from .onnx_detection import OnnxDetectorManifest, OnnxDocumentDetector
from .ocr import PaddleOcrModelManifest, PaddleOCRTextRecognizer, TextRecognizer
from .qr import QrCodeExtractor

__all__ = [
    "BoundingBox",
    "CaptureAssessment",
    "CaptureAssessmentConfig",
    "CaptureAssessmentInputError",
    "CaptureIssue",
    "CaptureIssueCode",
    "CaptureMetrics",
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
    "DocumentCaptureAssessor",
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
