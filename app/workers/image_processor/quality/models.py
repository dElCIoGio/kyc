from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


def _immutable_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class AssessmentResult:
    assessor: str
    metrics: Mapping[str, float]
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "parameters", _immutable_mapping(self.parameters))


@dataclass(frozen=True)
class VariantAssessment:
    variant_name: str
    generator: str
    variant_parameters: Mapping[str, object]
    results: tuple[AssessmentResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variant_parameters",
            _immutable_mapping(self.variant_parameters),
        )
        object.__setattr__(self, "results", tuple(self.results))


@dataclass(frozen=True)
class VariantBatchAssessment:
    generator: str
    assessments: tuple[VariantAssessment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assessments", tuple(self.assessments))
