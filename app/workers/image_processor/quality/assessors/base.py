from typing import Mapping, Protocol

from ...models import Image
from ..models import AssessmentResult


class ImageQualityAssessor(Protocol):
    name: str

    def assess(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> AssessmentResult:
        ...
