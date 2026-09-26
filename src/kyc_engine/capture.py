from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    CaptureAssessment,
    CaptureIssue,
    CaptureIssueCode,
    CaptureMetrics,
)
from .intake import ImageIntake, IntakeError
from .quality import BrightnessAssessor, ContrastAssessor, SharpnessAssessor


class CaptureAssessmentInputError(ValueError):
    """A safe, machine-readable failure while validating a capture input."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class CaptureAssessmentConfig:
    min_sharpness: float = 25.0
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    min_contrast: float = 12.0

    def __post_init__(self) -> None:
        values = (
            self.min_sharpness,
            self.min_brightness,
            self.max_brightness,
            self.min_contrast,
        )
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            raise TypeError("Capture assessment thresholds must be numeric")
        if self.min_sharpness < 0 or self.min_contrast < 0:
            raise ValueError("Capture sharpness and contrast thresholds must be non-negative")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ValueError("Capture brightness thresholds must satisfy 0 <= min < max <= 255")


class DocumentCaptureAssessor:
    """Fast deterministic gate deciding whether an image merits OCR processing."""

    def __init__(
        self,
        intake: ImageIntake,
        config: CaptureAssessmentConfig | None = None,
    ) -> None:
        self._intake = intake
        self._config = config or CaptureAssessmentConfig()
        self._sharpness = SharpnessAssessor()
        self._brightness = BrightnessAssessor()
        self._contrast = ContrastAssessor()

    def assess(self, source: bytes) -> CaptureAssessment:
        try:
            input_image = self._intake.load(source)
        except IntakeError as exc:
            raise CaptureAssessmentInputError(exc.code) from exc

        image = input_image.image
        sharpness = float(self._sharpness.assess(image).metrics["laplacian_variance"])
        brightness = float(self._brightness.assess(image).metrics["mean_intensity"])
        contrast = float(self._contrast.assess(image).metrics["intensity_stddev"])
        issues: list[CaptureIssue] = []
        if sharpness < self._config.min_sharpness:
            issues.append(CaptureIssue(CaptureIssueCode.TOO_BLURRY, "The document image is too blurry."))
        if brightness < self._config.min_brightness:
            issues.append(CaptureIssue(CaptureIssueCode.TOO_DARK, "The document image is too dark."))
        if brightness > self._config.max_brightness:
            issues.append(CaptureIssue(CaptureIssueCode.OVEREXPOSED, "The document image is overexposed."))
        if contrast < self._config.min_contrast:
            issues.append(CaptureIssue(CaptureIssueCode.LOW_CONTRAST, "The document image has too little contrast."))
        return CaptureAssessment(
            accepted=not issues,
            issues=tuple(issues),
            metrics=CaptureMetrics(
                width=input_image.width,
                height=input_image.height,
                sharpness=sharpness,
                brightness=brightness,
                contrast=contrast,
            ),
        )
