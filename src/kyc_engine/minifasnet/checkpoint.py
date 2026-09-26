"""MiniFASNet filename metadata adapted from Silent-Face-Anti-Spoofing.

Source: Minivision Silent-Face-Anti-Spoofing ``src/utility.py`` (Apache-2.0).
The parser is narrowed to the two supported inference architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable

from .errors import MiniFASNetInitializationError


_SUPPORTED_ARCHITECTURES = frozenset({"MiniFASNetV2", "MiniFASNetV1SE"})


@dataclass(frozen=True)
class MiniFASNetModelSpec:
    path: Path
    input_height: int
    input_width: int
    architecture: str
    scale: float | None


def parse_model_filename(filename: str) -> tuple[int, int, str, float | None]:
    """Parse the upstream `<scale>_<height>x<width>_<architecture>.pth` format."""
    name = Path(filename).name
    if name != filename or not name.endswith(".pth"):
        raise MiniFASNetInitializationError("Unsupported MiniFASNet model filename")
    parts = name[:-4].split("_")
    if len(parts) < 3 or parts[-1] not in _SUPPORTED_ARCHITECTURES:
        raise MiniFASNetInitializationError("Unsupported MiniFASNet model filename")
    try:
        height_text, width_text = parts[-2].split("x", maxsplit=1)
        height, width = int(height_text), int(width_text)
    except (TypeError, ValueError) as exc:
        raise MiniFASNetInitializationError("Unsupported MiniFASNet model filename") from exc
    if height <= 0 or width <= 0:
        raise MiniFASNetInitializationError("Unsupported MiniFASNet model filename")
    if parts[0] == "org":
        scale: float | None = None
    else:
        try:
            scale = float(parts[0])
        except ValueError as exc:
            raise MiniFASNetInitializationError("Unsupported MiniFASNet model filename") from exc
        if not isfinite(scale) or scale <= 0:
            raise MiniFASNetInitializationError("Unsupported MiniFASNet model filename")
    return height, width, parts[-1], scale


def discover_models(model_dir: Path) -> tuple[MiniFASNetModelSpec, ...]:
    if not model_dir.is_dir():
        raise MiniFASNetInitializationError("MiniFASNet model directory is unavailable")
    try:
        candidates = sorted(
            (path for path in model_dir.iterdir() if path.is_file() and path.suffix == ".pth"),
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise MiniFASNetInitializationError("MiniFASNet model directory is unreadable") from exc
    if not candidates:
        raise MiniFASNetInitializationError("MiniFASNet model files are missing")
    specs: list[MiniFASNetModelSpec] = []
    for path in candidates:
        if not _is_readable(path):
            raise MiniFASNetInitializationError("MiniFASNet model file is unreadable")
        height, width, architecture, scale = parse_model_filename(path.name)
        specs.append(
            MiniFASNetModelSpec(
                path=path,
                input_height=height,
                input_width=width,
                architecture=architecture,
                scale=scale,
            )
        )
    return tuple(specs)


def validate_resources(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file() or not _is_readable(path):
            raise MiniFASNetInitializationError("Required MiniFASNet resource is unavailable")


def _is_readable(path: Path) -> bool:
    try:
        with path.open("rb"):
            return True
    except OSError:
        return False
