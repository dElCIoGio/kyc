from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from kyc_engine import DocumentExtractionResult, build_paddle_document_coordinator


# Put the two private card images in these locations, or override them on the
# command line with --front and --back.
DEFAULT_FRONT = Path("private-data/ao-id-front/front.jpeg")
DEFAULT_BACK = Path("private-data/ao-id-back/back.jpeg")
DEFAULT_MODEL_MANIFEST = Path(
    "private-models/paddleocr/ao-id-front-v5-mobile/manifest.json"
)
DEFAULT_OUTPUT = Path("private-data/document-coordinator-result.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process the front and back of an Angolan ID with DocumentCoordinator."
    )
    parser.add_argument("--front", type=Path, default=DEFAULT_FRONT)
    parser.add_argument("--back", type=Path, default=DEFAULT_BACK)
    parser.add_argument(
        "--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        output_path = _private_output_path(args.output)
        _require_input(args.front)
        _require_input(args.back)
    except ValueError as exc:
        print(f"coordinator_failed={exc}")
        return 2

    try:
        with _suppress_backend_output():
            coordinator = build_paddle_document_coordinator(
                model_manifest=args.model_manifest,
                device=args.device,
            )
    except (ImportError, OSError, TypeError, ValueError):
        print("coordinator_failed=MODEL_CONFIGURATION_FAILED")
        return 2

    try:
        with _suppress_backend_output():
            result = coordinator.process(
                front=args.front,
                back=args.back,
            )
    except Exception:
        print("coordinator_failed=PROCESSING_FAILED")
        return 1

    try:
        _write_result(output_path, result)
    except OSError:
        print("coordinator_failed=OUTPUT_WRITE_FAILED")
        return 1

    _print_safe_summary(result)
    return 0


def _private_output_path(output: Path) -> Path:
    root = (Path.cwd() / "private-data").resolve()
    resolved = output.resolve()
    if resolved.suffix.lower() != ".json" or not resolved.is_relative_to(root):
        raise ValueError("INVALID_PRIVATE_OUTPUT")
    return resolved


def _require_input(path: Path) -> None:
    if not path.is_file():
        raise ValueError("INPUT_NOT_FOUND")


def _write_result(output: Path, result: DocumentExtractionResult) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _print_safe_summary(result: DocumentExtractionResult) -> None:
    print(f"status={result.status.value}")
    for side, side_result in (("front", result.front), ("back", result.back)):
        if side_result is None:
            print(f"{side}_status=not_processed")
            continue
        statuses = Counter(field.status.value for field in side_result.fields.values())
        print(f"{side}_status={side_result.status.value}")
        print(f"{side}_field_status_counts={dict(sorted(statuses.items()))}")
        qr_status = (
            side_result.qr_code.status.value
            if side_result.qr_code is not None
            else "not_configured"
        )
        print(f"{side}_qr_status={qr_status}")
    print(f"issue_codes={sorted({issue.code for issue in result.issues})}")
    print(f"timings_ms={dict(sorted(result.timings_ms.items()))}")


@contextmanager
def _suppress_backend_output():
    """Keep Paddle warnings and recognized text out of the terminal."""
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
