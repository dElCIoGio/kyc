# Offline Face-Similarity Evaluation

This development tool evaluates raw similarity distributions only. It is not a
KYC workflow: it does not create sessions, use the artifact store, make a match
decision, or implement a threshold.

Provision a separately approved local development model under the configured
`<model-root>/models/<model-id>/` directory. The directory may contain unrelated
ONNX files, but InsightFace must identify exactly one usable `detection` model
and exactly one usable `recognition` model. Model files must remain outside Git.
Current InsightFace pretrained packs are development/test-only until the exact
weights' commercial rights are confirmed separately.

Place consented evaluation inputs under an ignored local directory, for example
`private-data/face-evaluation/`, with a `manifest.csv`:

```csv
subject_id,reference_path,probe_path
subject-a,reference/a-1.jpg,probe/a-1.jpg
subject-b,reference/b-1.jpg,probe/b-1.jpg
```

`subject_id` is an opaque local label used only to separate genuine from
cross-subject impostor pairs. Image paths are relative to the dataset root and
must stay beneath it. The dataset needs at least two subjects.

Run the evaluator after installing the recognition extra:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_face_evaluation.py `
  --dataset-root private-data/face-evaluation `
  --model-root private-models/recognition `
  --model-id development-pack `
  --output private-data/face-evaluation/report.json
```

The JSON report contains only aggregate genuine/impostor counts, quantiles,
histograms, and exploratory false-match/false-non-match rate tables across a
fixed raw-similarity grid. It contains no images, labels, embeddings, pair-level
scores, or thresholds. Do not treat it as a recommendation for operational
MATCH, REVIEW, or NO_MATCH boundaries.
