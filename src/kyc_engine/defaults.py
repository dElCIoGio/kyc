from __future__ import annotations

from pathlib import Path

from .coordinator import DocumentCoordinator
from .detection import DocumentDetector, OpenCVDocumentDetector
from .fields import FieldLocalizer
from .intake import ImageIntake, IntakeLimits
from .normalization import DocumentNormalizer
from .ocr import PaddleOCRTextRecognizer, PaddleOcrModelManifest, TextRecognizer
from .pipeline import KycPipeline
from .profiles import ProfileRegistry, load_default_profiles
from .quality import QualityAssessmentPipeline, StructuralQualityGate
from .qr import QrCodeExtractor
from .reconciliation import CandidateReconciler
from .variants import BalancedVariantPolicy


def build_balanced_pipeline(
    recognizer: TextRecognizer,
    *,
    detector: DocumentDetector | None = None,
    intake_limits: IntakeLimits | None = None,
    side: str = "front",
) -> KycPipeline:
    """Build one side pipeline for advanced composition and tests."""
    return KycPipeline(
        intake=ImageIntake(intake_limits),
        detector=detector or OpenCVDocumentDetector(side=side),
        normalizer=DocumentNormalizer(),
        profiles=ProfileRegistry(load_default_profiles()),
        variants=BalancedVariantPolicy(),
        quality=QualityAssessmentPipeline(),
        quality_gate=StructuralQualityGate(),
        localizer=FieldLocalizer(),
        recognizer=recognizer,
        reconciler=CandidateReconciler(),
        qr_extractor=QrCodeExtractor(),
    )


def build_paddle_pipeline(
    *,
    model_manifest: str | Path,
    device: str | None = None,
    detector: DocumentDetector | None = None,
    intake_limits: IntakeLimits | None = None,
    side: str = "front",
) -> KycPipeline:
    """Build one PaddleOCR side pipeline for advanced composition."""
    manifest = PaddleOcrModelManifest.load(model_manifest)
    recognizer = PaddleOCRTextRecognizer.from_manifest(
        manifest,
        device=device,
    )
    return build_balanced_pipeline(
        recognizer,
        detector=detector,
        intake_limits=intake_limits,
        side=side,
    )


def build_paddle_document_coordinator(
    *,
    model_manifest: str | Path,
    device: str | None = None,
    intake_limits: IntakeLimits | None = None,
    front_detector: DocumentDetector | None = None,
    back_detector: DocumentDetector | None = None,
) -> DocumentCoordinator:
    """Build the supported two-sided local extraction entry point.

    Each side receives its own PaddleOCR pipeline and may use a separately
    configured detector. When no detector is supplied, the development-grade
    OpenCV detector remains the local default.
    """
    front_pipeline = build_paddle_pipeline(
        model_manifest=model_manifest,
        device=device,
        detector=front_detector,
        intake_limits=intake_limits,
        side="front",
    )
    back_pipeline = build_paddle_pipeline(
        model_manifest=model_manifest,
        device=device,
        detector=back_detector,
        intake_limits=intake_limits,
        side="back",
    )
    return DocumentCoordinator(front_pipeline, back_pipeline)
