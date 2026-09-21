from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import (
    DocumentProfile,
    ExtractedField,
    FieldDefinition,
    FieldStatus,
    IssueSeverity,
    OCRCandidate,
    PipelineIssue,
)
from .validation import FieldValueProcessor, ValueAssessment


@dataclass(frozen=True)
class ReconciliationResult:
    fields: Mapping[str, ExtractedField]
    issues: tuple[PipelineIssue, ...]


class CandidateReconciler:
    def __init__(
        self,
        processor: FieldValueProcessor | None = None,
        *,
        conflict_confidence_margin: float = 0.05,
    ) -> None:
        self.processor = processor or FieldValueProcessor()
        self.conflict_confidence_margin = conflict_confidence_margin

    def reconcile(
        self,
        profile: DocumentProfile,
        candidates: Sequence[OCRCandidate],
    ) -> ReconciliationResult:
        by_field: dict[str, list[OCRCandidate]] = defaultdict(list)
        for candidate in candidates:
            by_field[candidate.field_name].append(candidate)

        fields: dict[str, ExtractedField] = {}
        issues: list[PipelineIssue] = []
        for definition in profile.fields:
            field_result = self._reconcile_field(definition, by_field.get(definition.name, []))
            fields[definition.name] = field_result
            if field_result.status != FieldStatus.VALID:
                severity = IssueSeverity.ERROR if definition.required else IssueSeverity.WARNING
                issues.append(
                    PipelineIssue(
                        stage="reconciliation",
                        code=f"FIELD_{field_result.status.value.upper()}",
                        severity=severity,
                        message="A configured field could not be resolved reliably",
                        field_name=definition.name,
                    )
                )
        return ReconciliationResult(fields, tuple(issues))

    def _reconcile_field(
        self,
        definition: FieldDefinition,
        candidates: Sequence[OCRCandidate],
    ) -> ExtractedField:
        successful = [item for item in candidates if not item.error_code and item.raw_value.strip()]
        if not successful:
            status = FieldStatus.ERROR if candidates and all(item.error_code for item in candidates) else FieldStatus.MISSING
            return ExtractedField(
                name=definition.name,
                status=status,
                raw_value=None,
                normalized_value=None,
                confidence=0.0,
                selected_candidate=None,
                alternatives=tuple(candidates),
            )

        groups: dict[str, list[tuple[int, OCRCandidate, ValueAssessment]]] = defaultdict(list)
        for index, candidate in enumerate(successful):
            key = self.processor.comparison_key(definition, candidate.raw_value)
            assessment = self.processor.assess(definition, candidate.raw_value)
            groups[key].append((index, candidate, assessment))

        ranked = sorted(
            groups.values(),
            key=_group_score,
            reverse=True,
        )
        selected_group = ranked[0]
        selected_index, selected, assessment = max(
            selected_group,
            key=lambda item: (item[1].confidence, -item[0]),
        )
        conflict = False
        if len(ranked) > 1:
            first_score = _group_score(selected_group)
            second_score = _group_score(ranked[1])
            conflict = (
                first_score[:2] == second_score[:2]
                and abs(first_score[2] - second_score[2]) <= self.conflict_confidence_margin
            )

        status = FieldStatus.UNCERTAIN if conflict else assessment.status
        warnings = list(assessment.warnings)
        if conflict:
            warnings.append("Competing OCR candidate groups have similar evidence")
        alternatives = tuple(item for item in candidates if item is not selected)
        return ExtractedField(
            name=definition.name,
            status=status,
            raw_value=selected.raw_value,
            normalized_value=assessment.normalized_value,
            confidence=selected.confidence,
            selected_candidate=selected,
            alternatives=alternatives,
            warnings=tuple(warnings),
        )


def _group_score(
    group: Sequence[tuple[int, OCRCandidate, ValueAssessment]],
) -> tuple[int, int, float, int]:
    rank = {
        FieldStatus.VALID: 2,
        FieldStatus.UNCERTAIN: 1,
        FieldStatus.INVALID: 0,
    }
    validation_rank = max(rank.get(item[2].status, 0) for item in group)
    confidence = max(item[1].confidence for item in group)
    first_index = min(item[0] for item in group)
    return validation_rank, len(group), confidence, -first_index
