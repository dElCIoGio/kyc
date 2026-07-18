from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import Mapping, Sequence

from ..models import VariantBatch
from .assessors.base import ImageQualityAssessor
from .models import VariantAssessment, VariantBatchAssessment
from .validator import validate_variant_batch


class VariantQualityAssessmentPipeline:
    def __init__(
        self,
        assessors: Sequence[ImageQualityAssessor],
    ) -> None:
        self._assessors: dict[str, ImageQualityAssessor] = {}

        for assessor in assessors:
            name = assessor.name
            if not isinstance(name, str):
                raise TypeError("Assessor name must be a string")

            if not name.strip():
                raise ValueError("Assessor name cannot be empty")

            if name in self._assessors:
                raise ValueError(f"Duplicate assessor name: {name}")

            self._assessors[name] = assessor

    @property
    def assessor_names(self) -> tuple[str, ...]:
        return tuple(self._assessors)

    def assess(
        self,
        batch: VariantBatch,
        *,
        selected_assessors: Sequence[str] | None = None,
        parameters: Mapping[str, Mapping[str, object]] | None = None,
    ) -> VariantBatchAssessment:
        validate_variant_batch(batch)
        names = self._resolve_assessor_names(selected_assessors)
        parameter_map = self._resolve_parameter_map(parameters)

        assessments = tuple(
            VariantAssessment(
                variant_name=variant.name,
                generator=batch.generator,
                variant_parameters=variant.parameters,
                results=tuple(
                    self._assessors[name].assess(
                        variant.image.copy(),
                        parameters=parameter_map.get(name),
                    )
                    for name in names
                ),
            )
            for variant in batch.variants
        )

        return VariantBatchAssessment(
            generator=batch.generator,
            assessments=assessments,
        )

    def _resolve_assessor_names(
        self,
        selected_assessors: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if selected_assessors is None:
            return self.assessor_names

        if isinstance(selected_assessors, (str, bytes)) or not isinstance(
            selected_assessors, SequenceABC
        ):
            raise TypeError("selected_assessors must be a sequence of names")

        names = tuple(selected_assessors)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("selected_assessors must contain non-empty names")

        if len(set(names)) != len(names):
            raise ValueError("selected_assessors cannot contain duplicate names")

        unknown = tuple(name for name in names if name not in self._assessors)
        if unknown:
            raise KeyError(f"Unknown assessor names: {', '.join(unknown)}")

        return names

    def _resolve_parameter_map(
        self,
        parameters: Mapping[str, Mapping[str, object]] | None,
    ) -> Mapping[str, Mapping[str, object]]:
        if parameters is None:
            return {}

        if not isinstance(parameters, MappingABC):
            raise TypeError(
                "Expected parameters to be a mapping, received "
                f"{type(parameters).__name__}"
            )

        unknown = tuple(name for name in parameters if name not in self._assessors)
        if unknown:
            formatted = ", ".join(repr(name) for name in unknown)
            raise KeyError(
                f"Parameters provided for unknown assessor names: {formatted}"
            )

        for name, assessor_parameters in parameters.items():
            if not isinstance(assessor_parameters, MappingABC):
                raise TypeError(f"Parameters for assessor '{name}' must be a mapping")

        return parameters
