"""Run aggregate-only offline face-similarity evaluation on local consented data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kyc_engine import InsightFaceRecognizer
from kyc_engine.face_evaluation import CsvFaceEvaluationDatasetLoader, FaceEvaluationService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an aggregate genuine/impostor similarity report from local data."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="Safe configured model label, not a path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    recognizer = InsightFaceRecognizer(args.model_root, args.model_id)
    report = FaceEvaluationService(recognizer).evaluate(
        CsvFaceEvaluationDatasetLoader(args.dataset_root)
    )
    output = args.output.resolve()
    private_root = Path("private-data").resolve()
    if not output.is_relative_to(private_root):
        parser.error("--output must be located beneath the ignored private-data directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
