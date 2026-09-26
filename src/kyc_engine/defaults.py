from __future__ import annotations

from pathlib import Path

from .coordinator import DocumentCoordinator
from .detection import DocumentDetector, OpenCVDocumentDetector
from .fields import FieldLocalizer
from .intake import ImageIntake, IntakeLimits
from .normalization import DocumentNormalizer
from .ocr import PaddleOCRTextRecognizer, PaddleOcrModelManifest, TextRecognizer
from .pipeline import KycPipeline
from .portrait import OpenCVHaarFaceDetector, PortraitExtractor
from .portrait_artifacts import InMemoryPortraitArtifactStore
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
    portrait_artifacts: InMemoryPortraitArtifactStore | None = None,
    portrait_extractor: PortraitExtractor | None = None,
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
        portrait_extractor=portrait_extractor
        or PortraitExtractor(
            OpenCVHaarFaceDetector(),
            portrait_artifacts or InMemoryPortraitArtifactStore(),
        ),
    )


def build_paddle_pipeline(
    *,
    model_manifest: str | Path,
    device: str | None = None,
    detector: DocumentDetector | None = None,
    intake_limits: IntakeLimits | None = None,
    side: str = "front",
    portrait_artifacts: InMemoryPortraitArtifactStore | None = None,
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
        portrait_artifacts=portrait_artifacts,
    )


def build_paddle_document_coordinator(
    *,
    model_manifest: str | Path,
    device: str | None = None,
    intake_limits: IntakeLimits | None = None,
    front_detector: DocumentDetector | None = None,
    back_detector: DocumentDetector | None = None,
    portrait_artifacts: InMemoryPortraitArtifactStore | None = None,
) -> DocumentCoordinator:
    """Build the supported two-sided local extraction entry point.

    Each side receives its own PaddleOCR pipeline and may use a separately
    configured detector. When no detector is supplied, the development-grade
    OpenCV detector remains the local default.
    """
    artifacts = portrait_artifacts or InMemoryPortraitArtifactStore()
    front_pipeline = build_paddle_pipeline(
        model_manifest=model_manifest,
        device=device,
        detector=front_detector,
        intake_limits=intake_limits,
        side="front",
        portrait_artifacts=artifacts,
    )
    back_pipeline = build_paddle_pipeline(
        model_manifest=model_manifest,
        device=device,
        detector=back_detector,
        intake_limits=intake_limits,
        side="back",
        portrait_artifacts=artifacts,
    )
    return DocumentCoordinator(front_pipeline, back_pipeline, portrait_artifacts=artifacts)
