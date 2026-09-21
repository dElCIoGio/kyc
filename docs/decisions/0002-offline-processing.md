# ADR 0002: Process Documents Offline

- Status: Accepted
- Date: 2026-09-21

## Context

Source images and OCR results contain identity data. External OCR and inference services introduce additional transport, authorization, residency, retention, and vendor-risk requirements.

## Decision

Version one performs decoding, detection, preprocessing, OCR, reconciliation, and validation on controlled local infrastructure. Runtime processing does not call cloud OCR or remote inference APIs.

## Consequences

- Images and field values do not leave the processing boundary by design.
- Models and native dependencies must be installed, versioned, and monitored locally.
- Model acquisition may occur during an explicit installation process, but production inference must not silently download artifacts.
- A future external-service adapter requires a new decision record and threat model.

