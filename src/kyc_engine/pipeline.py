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
    QrCodeResult,
    QrCodeStatus,
)
from .detection import DetectionError, DocumentDetector
from .fields import FieldLocalizationError, FieldLocalizer
from .intake import ImageIntake, IntakeError
from .instrumentation import observe_pipeline_stage
from .normalization import DocumentNormalizer, NormalizationError
from .ocr import TextRecognizer
from .profiles import ProfileRegistry
from .quality import QualityAssessmentPipeline, StructuralQualityGate
from .qr import QrCodeExtractor
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
        qr_extractor: QrCodeExtractor | None = None,
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
        self.qr_extractor = qr_extractor or QrCodeExtractor()

    def process(self, source: ImageSource) -> KycExtractionResult:
        processing_id = uuid4().hex
        timings: dict[str, float] = {}
        issues: list[PipelineIssue] = []

        try:
            with observe_pipeline_stage("intake"):
                input_image = _timed(
                    "intake",
                    timings,
                    lambda: self.intake.load(source, processing_id=processing_id),
                )
        except IntakeError as exc:
            return _failed(processing_id, "intake", exc.code, str(exc), timings)

        try:
            with observe_pipeline_stage("detection"):
                detection = _timed(
                    "detection",
                    timings,
                    lambda: self.detector.detect(input_image.image),
                )
        except DetectionError as exc:
            return _failed(processing_id, "detection", exc.code, str(exc), timings)

        try:
            with observe_pipeline_stage(
                "profile", error_code="UNSUPPORTED_DOCUMENT_PROFILE"
            ):
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
            with observe_pipeline_stage("normalization"):
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
            with observe_pipeline_stage(
                "variants", error_code="VARIANT_PROCESSING_FAILED"
            ):
                batches = _timed(
                    "variants",
                    timings,
                    lambda: self.variants.generate(normalized.image),
                )
        except (ValueError, RuntimeError):
            return _failed(
                processing_id,
                "variants",
                "VARIANT_PROCESSING_FAILED",
                "Image variants could not be generated safely",
                timings,
                detection=detection,
                profile_id=profile.profile_id,
            )
        try:
            with observe_pipeline_stage(
                "quality", error_code="VARIANT_PROCESSING_FAILED"
            ):
                assessments = _timed(
                    "quality",
                    timings,
                    lambda: self.quality.assess(batches),
                )
                usable_batches = self.quality_gate.select(batches, assessments)
        except (ValueError, RuntimeError):
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

        qr_code: QrCodeResult | None = None
        if profile.qr_code is not None:
            try:
                with observe_pipeline_stage("qr", error_code="QR_PROCESSING_FAILED"):
                    qr_code = _timed(
                        "qr",
                        timings,
                        lambda: self.qr_extractor.extract(usable_batches, profile.qr_code),
                    )
            except Exception:
                issues.append(
                    PipelineIssue(
                        stage="qr",
                        code="QR_PROCESSING_FAILED",
                        severity=IssueSeverity.ERROR,
                        message="The local QR decoder could not process the configured QR region",
                    )
                )
            else:
                if qr_code.status != QrCodeStatus.DECODED:
                    issues.append(
                        PipelineIssue(
                            stage="qr",
                            code=_qr_issue_code(qr_code.status),
                            severity=IssueSeverity.ERROR,
                            message="The configured QR code could not be decoded into a structured record",
                        )
                    )

        try:
            with observe_pipeline_stage(
                "field_localization", error_code="FIELD_LOCALIZATION_FAILED"
            ):
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
            with observe_pipeline_stage("ocr", error_code="OCR_ENGINE_FAILED"):
                candidates = _timed(
                    "ocr",
                    timings,
                    lambda: self.recognizer.recognize_batch(crops),
                )
                if len(candidates) != len(crops):
                    raise RuntimeError("TextRecognizer must return one candidate per field crop")
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
        with observe_pipeline_stage("reconciliation"):
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
            schema_version="1.1",
            processing_id=processing_id,
            status=status,
            document_type=detection.document_type,
            side=detection.side,
            profile_id=profile.profile_id,
            detection=detection,
            fields=reconciled.fields,
            issues=tuple(issues),
            timings_ms=timings,
            qr_code=qr_code,
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
        schema_version="1.1",
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


def _qr_issue_code(status: QrCodeStatus) -> str:
    return {
        QrCodeStatus.NOT_DETECTED: "QR_NOT_DETECTED",
        QrCodeStatus.DECODE_FAILED: "QR_DECODE_FAILED",
        QrCodeStatus.PARSE_FAILED: "QR_PARSE_FAILED",
    }[status]
