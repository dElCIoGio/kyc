from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from kyc_engine.contracts import KycExtractionResult
from kyc_engine.defaults import build_paddle_pipeline
from kyc_engine.pipeline import KycPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local PaddleOCR calibration without exposing recognized values on stdout."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    pipeline_factory: Callable[..., KycPipeline] = build_paddle_pipeline,
    private_root: Path | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path = _private_output_path(args.output, private_root=private_root)
    except ValueError:
        print("calibration_failed=INVALID_PRIVATE_OUTPUT")
        return 2

    try:
        with _suppress_backend_output():
            pipeline = pipeline_factory(
                model_manifest=args.model_manifest,
                device=args.device,
            )
    except (ImportError, OSError, TypeError, ValueError):
        print("calibration_failed=MODEL_CONFIGURATION_FAILED")
        return 2

    try:
        with _suppress_backend_output():
            result = pipeline.process(args.image)
    except Exception:
        print("calibration_failed=PROCESSING_FAILED")
        return 1

    try:
        _write_result(output_path, result)
    except OSError:
        print("calibration_failed=OUTPUT_WRITE_FAILED")
        return 1

    _print_safe_summary(result)
    return 0


def _private_output_path(output: Path, *, private_root: Path | None = None) -> Path:
    root = (private_root or Path.cwd() / "private-data").resolve()
    resolved = output.resolve()
    if resolved.suffix.lower() != ".json" or not resolved.is_relative_to(root):
        raise ValueError("Calibration output must be a JSON file within private-data")
    return resolved


def _write_result(output: Path, result: KycExtractionResult) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _print_safe_summary(result: KycExtractionResult) -> None:
    statuses = Counter(field.status.value for field in result.fields.values())
    issue_codes = sorted({issue.code for issue in result.issues})
    timings: dict[str, Any] = dict(result.timings_ms)
    print(f"status={result.status.value}")
    print(f"field_status_counts={dict(sorted(statuses.items()))}")
    print(f"issue_codes={issue_codes}")
    print(f"timings_ms={dict(sorted(timings.items()))}")


@contextmanager
def _suppress_backend_output():
    """Keep dependency warnings and OCR text out of a calibration terminal."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


if __name__ == "__main__":
    raise SystemExit(main())
