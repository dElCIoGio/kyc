import sys
import unittest
from pathlib import Path
from typing import Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from image_processor.context import VariantGenerationContext
from image_processor.models import Image, VariantBatch, VariantInfo


class RecordingGenerator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.received_parameters: Mapping[str, object] | None = None

    def generate(
        self,
        image: Image,
        parameters: Mapping[str, object] | None = None,
    ) -> VariantBatch:
        self.received_parameters = parameters
        return VariantBatch(
            generator=self.name,
            variants=(
                VariantInfo(
                    name=f"{self.name}_variant",
                    image=image.copy(),
                    parameters=dict(parameters or {}),
                ),
            ),
        )


class VariantGenerationContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)

    def test_runs_registered_generators(self) -> None:
        context = VariantGenerationContext(
            generators=(RecordingGenerator("first"), RecordingGenerator("second"))
        )

        batches = context.generate(self.image)

        self.assertEqual(("first", "second"), tuple(batch.generator for batch in batches))

    def test_runs_selected_generators(self) -> None:
        context = VariantGenerationContext(
            generators=(RecordingGenerator("first"), RecordingGenerator("second"))
        )

        batches = context.generate(self.image, selected_generators=("second",))

        self.assertEqual(("second",), tuple(batch.generator for batch in batches))

    def test_passes_parameters_to_correct_generator(self) -> None:
        first = RecordingGenerator("first")
        second = RecordingGenerator("second")
        context = VariantGenerationContext(generators=(first, second))

        context.generate(
            self.image,
            parameters={"second": {"value": 3}},
        )

        self.assertIsNone(first.received_parameters)
        self.assertEqual({"value": 3}, second.received_parameters)

    def test_rejects_duplicate_generator_names(self) -> None:
        with self.assertRaises(ValueError):
            VariantGenerationContext(
                generators=(RecordingGenerator("same"), RecordingGenerator("same"))
            )

    def test_rejects_unknown_selected_generator(self) -> None:
        context = VariantGenerationContext(generators=(RecordingGenerator("known"),))

        with self.assertRaises(KeyError):
            context.generate(self.image, selected_generators=("missing",))

    def test_rejects_unknown_parameter_generator(self) -> None:
        context = VariantGenerationContext(generators=(RecordingGenerator("known"),))

        with self.assertRaises(KeyError):
            context.generate(self.image, parameters={"missing": {}})

    def test_rejects_non_mapping_generator_parameters(self) -> None:
        context = VariantGenerationContext(generators=(RecordingGenerator("known"),))

        with self.assertRaises(TypeError):
            context.generate(
                self.image,
                parameters={"known": ["not", "a", "mapping"]},  # type: ignore[dict-item]
            )


if __name__ == "__main__":
    unittest.main()
