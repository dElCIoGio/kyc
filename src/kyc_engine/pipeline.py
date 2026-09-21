from __future__ import annotations

from time import perf_counter
from typing import Callable
from uuid import uuid4

from .contracts import (
    ImageSource,
    IssueSeverity,
    KycExtractionResult,
    PipelineIssue,
    ProcessingStatus,
)
from .detection import DetectionError, DocumentDetector
from .fields import FieldLocalizationError, FieldLocalizer
from .intake import ImageIntake, IntakeError
from .normalization import DocumentNormalizer, NormalizationError
from .ocr import TextRecognizer
from .profiles import ProfileRegistry
from .quality import QualityAssessmentPipeline, StructuralQualityGate
from .reconciliation import CandidateReconciler
from .variants import BalancedVariantPolicy


class KycPipeline:
    def __init__(
        self,
        *,
        intake: ImageIntake,
        detector: DocumentDetector,
        normalizer: DocumentNormalizer,
        profiles: ProfileRegistry,
        variants: BalancedVariantPolicy,
        quality: QualityAssessmentPipeline,
        quality_gate: StructuralQualityGate,
        localizer: FieldLocalizer,
        recognizer: TextRecognizer,
        reconciler: CandidateReconciler,
    ) -> None:
        self.intake = intake
        self.detector = detector
        self.normalizer = normalizer
        self.profiles = profiles
        self.variants = variants
        self.quality = quality
        self.quality_gate = quality_gate
        self.localizer = localizer
        self.recognizer = recognizer
        self.reconciler = reconciler

    def process(self, source: ImageSource) -> KycExtractionResult:
        processing_id = uuid4().hex
        timings: dict[str, float] = {}
        issues: list[PipelineIssue] = []

        try:
            input_image = _timed(
                "intake",
                timings,
                lambda: self.intake.load(source, processing_id=processing_id),
            )
        except IntakeError as exc:
            return _failed(processing_id, "intake", exc.code, str(exc), timings)

        try:
            detection = _timed(
                "detection",
                timings,
                lambda: self.detector.detect(input_image.image),
            )
        except DetectionError as exc:
            return _failed(processing_id, "detection", exc.code, str(exc), timings)

        try:
            profile = self.profiles.resolve(detection.document_type, detection.side)
        except KeyError:
            return _failed(
                processing_id,
                "profile",
                "UNSUPPORTED_DOCUMENT_PROFILE",
                "No profile is configured for the detected document",
                timings,
                detection=detection,
            )

        if profile.review_status != "reviewed":
            issues.append(
                PipelineIssue(
                    stage="profile",
                    code="PROFILE_PROVISIONAL",
                    severity=IssueSeverity.WARNING,
                    message="The selected document profile has not completed layout review",
                )
            )

        try:
            normalized = _timed(
                "normalization",
                timings,
                lambda: self.normalizer.normalize(input_image.image, detection, profile),
            )
        except NormalizationError as exc:
            return _failed(
                processing_id,
                "normalization",
                exc.code,
                str(exc),
                timings,
                detection=detection,
                profile_id=profile.profile_id,
            )

        try:
            batches = _timed(
                "variants",
                timings,
                lambda: self.variants.generate(normalized.image),
            )
            assessments = _timed(
                "quality",
                timings,
                lambda: self.quality.assess(batches),
            )
            usable_batches = self.quality_gate.select(batches, assessments)
        except (ValueError, RuntimeError) as exc:
            return _failed(
                processing_id,
                "variants",
                "VARIANT_PROCESSING_FAILED",
                "Image variants could not be generated safely",
                timings,
                detection=detection,
                profile_id=profile.profile_id,
            )
        if not usable_batches:
            return _failed(
                processing_id,
                "quality",
                "NO_USABLE_VARIANTS",
                "No generated image variant contained usable intensity information",
                timings,
                detection=detection,
                profile_id=profile.profile_id,
            )

        try:
            crops = _timed(
                "field_localization",
                timings,
                lambda: self.localizer.localize(usable_batches, profile),
            )
        except FieldLocalizationError:
            return _failed(
                processing_id,
                "field_localization",
                "FIELD_LOCALIZATION_FAILED",
                "Configured fields could not be localized safely",
                timings,
                detection=detection,
                profile_id=profile.profile_id,
            )

        try:
            candidates = _timed(
                "ocr",
                timings,
                lambda: self.recognizer.recognize_batch(crops),
            )
        except Exception:
            return _failed(
                processing_id,
                "ocr",
                "OCR_ENGINE_FAILED",
                "The local OCR engine failed",
                timings,
                detection=detection,
                profile_id=profile.profile_id,
            )
        if len(candidates) != len(crops):
            raise RuntimeError("TextRecognizer must return one candidate per field crop")

        reconciled = _timed(
            "reconciliation",
            timings,
            lambda: self.reconciler.reconcile(profile, candidates),
        )
        issues.extend(reconciled.issues)
        status = (
            ProcessingStatus.PARTIAL
            if any(issue.severity == IssueSeverity.ERROR for issue in issues)
            else ProcessingStatus.SUCCESS
        )
        return KycExtractionResult(
            schema_version="1.0",
            processing_id=processing_id,
            status=status,
            document_type=detection.document_type,
            side=detection.side,
            profile_id=profile.profile_id,
            detection=detection,
            fields=reconciled.fields,
            issues=tuple(issues),
            timings_ms=timings,
        )


def _timed(name: str, timings: dict[str, float], operation: Callable[[], object]):
    started = perf_counter()
    try:
        return operation()
    finally:
        timings[name] = round((perf_counter() - started) * 1000.0, 3)


def _failed(
    processing_id: str,
    stage: str,
    code: str,
    message: str,
    timings: dict[str, float],
    *,
    detection=None,
    profile_id: str | None = None,
) -> KycExtractionResult:
    return KycExtractionResult(
        schema_version="1.0",
        processing_id=processing_id,
        status=ProcessingStatus.FAILED,
        document_type=getattr(detection, "document_type", None),
        side=getattr(detection, "side", None),
        profile_id=profile_id,
        detection=detection,
        fields={},
        issues=(
            PipelineIssue(
                stage=stage,
                code=code,
                severity=IssueSeverity.ERROR,
                message=message,
            ),
        ),
        timings_ms=timings,
    )

