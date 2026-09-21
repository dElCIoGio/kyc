# ADR 0001: Use a Modular Monolith

- Status: Accepted
- Date: 2026-09-21

## Context

Variant generation and field extraction currently live in separate Python projects, but the intended workflow is one ordered in-memory pipeline. Introducing serialized worker boundaries now would add deployment and data-protection complexity before stage contracts are stable.

## Decision

Build one installable `kyc_engine` package. Organize processing stages as separate modules with typed contracts and pluggable protocols. Keep orchestration in one process for the first complete engine.

## Consequences

- Local development, testing, dependency management, and PII containment are simpler.
- Stage ownership remains explicit and stages can later become workers behind the same contracts.
- Heavy detector and OCR dependencies should remain behind adapters and optional dependency groups.
- Existing modules require a controlled migration with compatibility tests.

