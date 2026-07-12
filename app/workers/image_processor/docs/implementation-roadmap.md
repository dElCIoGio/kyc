# Implementation Roadmap

## Ordered Steps

1. Add dependencies and package structure.
   - Add `numpy` and `opencv-python` to `pyproject.toml`.
   - Create `generators/` as a package.

2. Add strongly typed models.
   - Create `models.py`.
   - Define `Image`, `VariantInfo`, and `VariantBatch`.
   - Use frozen dataclasses for result objects.

3. Align validation with models.
   - Update `validator.py` to import the new `Image` alias successfully.
   - Keep existing checks for `None`, array type, dtype, dimensions, and channel counts.

4. Add the base generator contract.
   - Create `generators/base.py`.
   - Define the `VariantGenerator` protocol with `name` and `generate(...)`.

5. Implement generators one by one.
   - `OriginalVariantGenerator`
   - `GrayscaleVariantGenerator`
   - `ClaheVariantGenerator`
   - `GammaVariantGenerator`
   - `ContrastBrightnessVariantGenerator`
   - `SharpeningVariantGenerator`
   - `DenoisingVariantGenerator`
   - `ThresholdVariantGenerator`

6. Add `VariantGenerationContext`.
   - Accept generator instances.
   - Reject duplicate generator names.
   - Validate the input image once.
   - Route per-generator runtime parameters.
   - Return one `VariantBatch` per executed generator.

7. Add tests.
   - Add unit tests for models, validation, generators, and context behavior.
   - Use small synthetic `np.uint8` arrays so tests do not depend on external image files.

8. Wire the worker entrypoint.
   - Replace placeholder `main.py` behavior with a thin orchestration path.
   - Keep queue or persistence integration outside the first implementation unless separate requirements are added.

## Acceptance Criteria

- One image read happens before generation with `cv2.imread(path)`.
- The loaded image is validated before generators run.
- Each configured generator returns exactly one `VariantBatch`.
- Multi-scale generators return one `VariantInfo` per scale/config value.
- Runtime parameters can override generator constructor defaults.
- Invalid images and invalid parameters fail with clear exceptions.
- Generators do not mutate the original input image.
- Generated variants remain in memory as `np.ndarray` values.

## Test Plan

Unit tests for each generator:

- Produces the expected number of variants.
- Preserves valid `uint8` output.
- Does not mutate the input image.
- Rejects invalid parameter values.

Context tests:

- Runs multiple generators and returns multiple `VariantBatch` objects.
- Passes parameters to the correct generator.
- Raises a clear error for unknown generator names.
- Raises a clear error for duplicate generator names.

Validation tests:

- Rejects `None`.
- Rejects empty arrays.
- Rejects non-`np.ndarray` values.
- Rejects non-`uint8` arrays.
- Rejects unsupported dimensions.
- Rejects unsupported channel counts.

## Assumptions And Defaults

- Use `dataclasses` and `typing.Protocol`, not Pydantic, unless runtime model validation becomes a requirement later.
- Use OpenCV BGR arrays internally because `cv2.imread(path)` returns BGR.
- Keep saving generated variants to disk out of scope for the first implementation.
- Prioritize clean architecture and testability before worker queue integration.
- Validate once in `VariantGenerationContext` before dispatching to generators.
