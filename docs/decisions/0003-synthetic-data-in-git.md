# ADR 0003: Keep Only Synthetic Identity Data in Git

- Status: Accepted
- Date: 2026-09-21

## Context

Real identity-card images and OCR results create long-lived exposure when committed because Git history, forks, caches, and CI artifacts are difficult to erase reliably.

## Decision

Only synthetic or irreversibly redacted identity-document fixtures may be committed. Real samples and private evaluation datasets stay in access-controlled storage outside the repository. Generated variants and OCR result files are ignored.

## Consequences

- Routine tests must rely on generated scenes, fake identities, and fake model adapters.
- Production accuracy evaluation requires a separate controlled process.
- Unknown existing images are removed from the current tree until their origin is verified.
- Confirmed exposure in Git history triggers the incident process rather than an uncoordinated history rewrite.

