from collections.abc import Mapping as MappingABC
from typing import Mapping, Sequence

from .generators.base import VariantGenerator
from .models import Image, VariantBatch
from .validator import validate_image


class VariantGenerationContext:
    def __init__(self, generators: Sequence[VariantGenerator]) -> None:
        self._generators: dict[str, VariantGenerator] = {}

        for generator in generators:
            name = generator.name
            if not name:
                raise ValueError("Generator name cannot be empty")

            if name in self._generators:
                raise ValueError(f"Duplicate generator name: {name}")

            self._generators[name] = generator

    @property
    def generator_names(self) -> tuple[str, ...]:
        return tuple(self._generators)

    def generate(
        self,
        image: Image,
        *,
        selected_generators: Sequence[str] | None = None,
        parameters: Mapping[str, Mapping[str, object]] | None = None,
    ) -> tuple[VariantBatch, ...]:
        validate_image(image)

        names = self._resolve_generator_names(selected_generators)
        parameter_map = self._resolve_parameter_map(parameters)

        return tuple(
            self._generators[name].generate(
                image,
                parameters=parameter_map.get(name),
            )
            for name in names
        )

    def _resolve_generator_names(
        self,
        selected_generators: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if selected_generators is None:
            return self.generator_names

        names = tuple(selected_generators)
        unknown = tuple(name for name in names if name not in self._generators)
        if unknown:
            raise KeyError(f"Unknown generator names: {', '.join(unknown)}")

        return names

    def _resolve_parameter_map(
        self,
        parameters: Mapping[str, Mapping[str, object]] | None,
    ) -> Mapping[str, Mapping[str, object]]:
        if parameters is None:
            return {}

        if not isinstance(parameters, MappingABC):
            raise TypeError(
                f"Expected parameters to be a mapping, received {type(parameters).__name__}"
            )

        unknown = tuple(name for name in parameters if name not in self._generators)
        if unknown:
            raise KeyError(
                f"Parameters provided for unknown generator names: {', '.join(unknown)}"
            )

        for name, generator_parameters in parameters.items():
            if not isinstance(generator_parameters, MappingABC):
                raise TypeError(
                    f"Parameters for generator '{name}' must be a mapping"
                )

        return parameters
