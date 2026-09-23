from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

from .contracts import (
    ImageSource,
    IssueSeverity,
    KycExtractionResult,
    PipelineIssue,
    ProcessingStatus,
    immutable_mapping,
)
from .pipeline import KycPipeline
from .instrumentation import pipeline_side_context


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentExtractionResult:
    """The ordered, side-preserving result for one physical document."""

    schema_version: str
    status: ProcessingStatus
    front: KycExtractionResult | None
    back: KycExtractionResult | None
    issues: tuple[PipelineIssue, ...]
    timings_ms: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "timings_ms", immutable_mapping(self.timings_ms))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "front": self.front.to_dict() if self.front is not None else None,
            "back": self.back.to_dict() if self.back is not None else None,
            "issues": [
                {
                    "stage": issue.stage,
                    "code": issue.code,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "field_name": issue.field_name,
                }
                for issue in self.issues
            ],
            "timings_ms": dict(self.timings_ms),
        }


class DocumentCoordinator:
    """Run the already-configured front and back pipelines independently."""

    def __init__(self, front_pipeline: KycPipeline, back_pipeline: KycPipeline) -> None:
        if not callable(getattr(front_pipeline, "process", None)):
            raise TypeError("front_pipeline must provide a callable process method")
        if not callable(getattr(back_pipeline, "process", None)):
            raise TypeError("back_pipeline must provide a callable process method")
        self.front_pipeline = front_pipeline
        self.back_pipeline = back_pipeline

    def process(
        self,
        *,
        front: ImageSource | None = None,
        back: ImageSource | None = None,
    ) -> DocumentExtractionResult:
        if front is None and back is None:
            raise ValueError("At least one of front or back must be supplied")

        issues: list[PipelineIssue] = []
        timings: dict[str, float] = {}
        front_result = None
        back_result = None

        if front is None:
            issues.append(_missing_issue("front"))
        else:
            front_result = self._run_side("front", front, timings, issues)

        if back is None:
            issues.append(_missing_issue("back"))
        else:
            back_result = self._run_side("back", back, timings, issues)

        supplied_results = tuple(
            result for result in (front_result, back_result) if result is not None
        )
        status = _overall_status(
            supplied_results,
            missing_side=front is None or back is None,
            side_exception=any(
                issue.code in {"FRONT_PROCESSING_FAILED", "BACK_PROCESSING_FAILED"}
                for issue in issues
            ),
        )
        return DocumentExtractionResult(
            schema_version="1.0",
            status=status,
            front=front_result,
            back=back_result,
            issues=tuple(issues),
            timings_ms=timings,
        )

    def _run_side(
        self,
        side: str,
        source: ImageSource,
        timings: dict[str, float],
        issues: list[PipelineIssue],
    ) -> KycExtractionResult | None:
        started = perf_counter()
        try:
            with pipeline_side_context(side):
                return (
                    self.front_pipeline if side == "front" else self.back_pipeline
                ).process(source)
        except Exception as exc:
            logger.exception(
                "document side processing failed",
                extra={
                    "event": "document_side_processing_failed",
                    "side": side,
                    "stage": "coordinator",
                    "error_code": f"{side.upper()}_PROCESSING_FAILED",
                    "exception_type": type(exc).__name__,
                },
            )
            issues.append(
                PipelineIssue(
                    stage="coordinator",
                    code=f"{side.upper()}_PROCESSING_FAILED",
                    severity=IssueSeverity.ERROR,
                    message=f"The {side} side could not be processed",
                )
            )
            return None
        finally:
            timings[f"{side}_total"] = round((perf_counter() - started) * 1000.0, 3)


def _missing_issue(side: str) -> PipelineIssue:
    return PipelineIssue(
        stage="coordinator",
        code="SIDE_MISSING",
        severity=IssueSeverity.WARNING,
        message=f"The {side} side was not supplied",
    )


def _overall_status(
    results: tuple[KycExtractionResult, ...],
    *,
    missing_side: bool,
    side_exception: bool,
) -> ProcessingStatus:
    if not results or all(result.status == ProcessingStatus.FAILED for result in results):
        return ProcessingStatus.FAILED
    if (
        missing_side
        or side_exception
        or any(result.status != ProcessingStatus.SUCCESS for result in results)
    ):
        return ProcessingStatus.PARTIAL
    return ProcessingStatus.SUCCESS
