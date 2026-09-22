# Testing Strategy

## Routine Tests

Unit and integration tests use deterministic arrays, generated card scenes, and fake detector/OCR implementations. Only synthetic or irreversibly redacted fixtures may enter Git.

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The root suite covers bounded intake, profile validation, zero/one/multiple-card behavior, perspective transforms, the ten-variant policy, all five quality metrics, field crops, OCR mode batching, model manifests, reconciliation, coordinator behavior, failure states, result serialization, and input immutability.

## Invariants

- Inputs and intermediate source arrays are not mutated.
- Sequence order is deterministic across generators, variants, assessors, fields, candidates, and issues.
- Confidence values and geometry are finite and bounded.
- Every selected field points to a real candidate and retains generator, variant, OCR model, and box provenance.
- Unknown strategies, parameters, profiles, and malformed model outputs fail clearly.
- Serialized results contain no pixels, paths, EXIF, stack traces, or implicit debug artifacts.

## Synthetic Coverage to Expand

Generate cards with controlled rotation, perspective, scaling, blur, noise, glare, shadows, clipping, obstruction, and difficult backgrounds. Include zero-card and spatially separated multi-card scenes. Synthetic text must not copy a real person's identity.

## Optional Model Tests

Live PaddleOCR and ONNX tests are opt-in because model weights remain outside Git. Tests must use explicit local model paths and expected checksums. Missing required production weights is a configuration failure, not permission to use a classical fallback or download a model.

## Private Evaluation

Private data must be split by physical card or person to prevent identity leakage. Track:

- detector precision, recall, ambiguity rate, and semantic corner error;
- valid canonical-crop rate;
- per-field raw and normalized exact match plus character error rate;
- success, partial, and failed result rates;
- p50/p95 total and per-stage latency plus peak memory;
- metrics by lighting, blur, perspective, obstruction, device, and card revision.

Unit-test success is not evidence of production detection or OCR accuracy. Aggregated evaluation reports must suppress document images and field values.
