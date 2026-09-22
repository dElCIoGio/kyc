# Development Roadmap

`Implemented` means the core behavior and synthetic contract tests exist. It does not mean production accuracy has been established.

| Milestone | Status | Remaining release gate |
| --- | --- | --- |
| 1. Repository hygiene and documentation | Complete | Continue enforcing the private-data policy. |
| 2. Root `kyc_engine` package | Implemented | Sole supported library package under `src/kyc_engine`. |
| 3. Secure image intake | Implemented | Add a deployment-level hard wall-clock timeout and benchmark adversarial decodes. |
| 4. Card detection | Implemented for development | Train/provision private RTMDet + corner models and qualify ONNX accuracy. |
| 5. Perspective normalization | Implemented | Calibrate orientation anchors or use qualified semantic corner output. |
| 6. Variants and quality | Implemented | Calibrate any non-structural gates on private data. |
| 7. Versioned field profile | Provisional | Confirm complete field inventory and geometry from authoritative examples. |
| 8. Local OCR adapter | Implemented | Benchmark Portuguese model choices, pin selected artifacts, and test live inference. |
| 9. Candidate reconciliation | Implemented | Calibrate conflict margins against private validation results. |
| 10. Field normalization/validation | Implemented conservatively | Add Angolan document-number rules only from an authoritative specification. |
| 11. Core orchestration | Implemented | Side-specific pipelines and a two-sided document coordinator are available; production CLI, APIs, queues, and persistence remain intentionally deferred. |
| 12. Private accuracy/performance evaluation | Pending external data | Define release thresholds and demonstrate p95 CPU latency under ten seconds. |
| 13. Local HTTP orchestration service | Implemented | Single-process, memory-only sessions; shared queues, durable stores, and user identity remain deferred. |

## Next Work

1. Review and sign off `ao_id_card/front/v1`, including all front-side textual fields and canonical coordinates.
2. Build a private, identity-separated train/validation/test dataset with one box and semantic corner annotations.
3. Train and export detector models to the documented ONNX contract; provision only checksum-pinned artifacts.
4. Benchmark Portuguese PaddleOCR mobile/small recognition models and select by macro normalized exact match, then p95 CPU latency and memory when accuracy is within one percentage point.
5. Run end-to-end private evaluation and calibrate detector thresholds, reconciliation conflict margins, and optional quality gates.
6. Record reference CPU hardware and require p95 processing below ten seconds before release.

## Definition of Production-Ready v1

- The profile is reviewed and versioned from authoritative layout evidence.
- Required detector and OCR artifacts are available locally and pass checksum validation.
- No component downloads a model during processing or silently falls back to another detector.
- Held-out private metrics meet written detector, crop, field-accuracy, failure-rate, latency, and memory thresholds.
- Real identity data remains outside Git and routine CI artifacts.
- Security tests demonstrate bounded inputs and PII-safe diagnostics.
