from __future__ import annotations

from pathlib import Path

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
    side: str = "front",
) -> KycPipeline:
    manifest = PaddleOcrModelManifest.load(model_manifest)
    recognizer = PaddleOCRTextRecognizer.from_manifest(
        manifest,
        device=device,
    )
    return build_balanced_pipeline(recognizer, detector=detector, side=side)
