# Security and Data Handling

Identity-card photographs and extracted fields are sensitive personal data. The default design minimizes collection, processing time, logging, and persistence. Version one runs entirely on controlled local infrastructure.

## Repository Policy

- Only synthetic or irreversibly redacted document fixtures may be committed.
- Real documents, private evaluation datasets, generated variants, OCR output, logs containing field values, and model caches remain outside Git.
- Unknown images are treated as sensitive until reviewed.
- Test fixtures must not use real names, document numbers, addresses, dates, portraits, signatures, or machine-readable codes copied from a person.
- Secrets belong in ignored local environment files or the deployment secret store, never in configuration committed to Git.

## Controls by Stage

### Intake

- Restrict supported formats and verify decoded content rather than trusting an extension or MIME claim.
- Enforce configurable request-body, file-size, width, height, total-pixel, and decode-time limits before expensive work.
- Resolve caller paths safely and reject unintended directory traversal at service boundaries.
- Reject empty, truncated, malformed, unsupported, and decompression-bomb inputs with safe errors.
- Discard EXIF and unrelated metadata from internal and returned models.

### Detection and Normalization

- Validate confidence, corner count, point ordering, finite coordinates, bounds, and minimum card area.
- Bound perspective-transform dimensions and memory allocation.
- Keep detection separate from authenticity and communicate that distinction in results.
- Reject impossible geometry instead of producing an unsafe crop.

### Variants and Quality

- Keep source arrays immutable and bound generator counts, image dimensions, and concurrent work.
- Avoid writing intermediate images unless an explicit debug mode is enabled in a safe local directory.
- Debug output must have a retention policy and must never be enabled by default in production.

### OCR

- Use local models in version one; no source images or field crops leave controlled infrastructure.
- Pin model and library versions and record model provenance. Production model files should support integrity checks.
- Do not log recognized text, field crops, document identifiers, or full OCR response objects.
- Convert backend exceptions into safe stage errors while retaining diagnostic detail only in protected local tooling.

### Reconciliation, Validation, and Output

- Preserve candidate provenance and never fabricate missing values.
- Avoid embedding field values in warning or exception messages.
- Return only fields required by the selected profile and caller contract.
- Do not persist results implicitly. The CLI writes structured values only when explicitly requested.
- Apply authorization, encryption, retention, and deletion controls in any future persistence or service layer.

## Logging and Diagnostics

Allowed logs include processing identifiers, stage names, durations, image dimensions after validation, model/profile versions, safe error codes, aggregate counts, and exception types. Logs must exclude pixels, field crops, OCR text, normalized field values, original filenames, user-provided paths, exception-message text, and arbitrary serialized objects. Application JSON logs may include stack-frame locations for unexpected exceptions, but never exception representations or locals that could contain sensitive payloads.

Diagnostic bundles containing images or OCR values are sensitive artifacts. They require an explicit opt-in destination, restrictive access, and a documented expiry process.

## HTTP API

- Require a constant-time compared environment API key for every `/v1/*` route.
- Keep health responses public only when they contain no configuration, paths, session identifiers, or document data.
- Bound request and image sizes and keep accepted multipart uploads below the in-memory spool threshold.
- Store uploads and results only in the process-local session store; clear image bytes immediately after extraction.
- Expire and delete complete session state, including results, after the configured TTL.
- Keep status responses separate from the result endpoint so polling does not repeatedly disclose extracted identity values.
- Apply a valid-key fixed-window request limit and return a safe `429` response with `Retry-After` when exhausted.
- Apply a soft job deadline that invalidates the session result and clears retained state; do not release its worker capacity until native OCR exits.
- Publish only authenticated aggregate metrics: response classes, safe error codes, job outcomes, and latency summaries.
- Run a single server process until sessions and jobs use a shared external backend.
- Put TLS, network access controls, and deployment-level request-size/time limits in front of any non-local deployment.
- Run Uvicorn with `--no-access-log`; application logs must not include request bodies, filenames, paths, OCR output, or extraction results.

## Dependency and Model Safety

- Lock production dependencies and review updates to OpenCV, NumPy, Pillow, PaddleOCR, PaddlePaddle, and detector runtimes.
- Obtain models from documented sources, record checksums, and do not execute untrusted serialized model formats.
- Keep model download behavior explicit; production startup must not silently fetch code or models from the network.

## Incident: Sensitive Data in Git

1. Stop sharing or mirroring the repository and restrict access.
2. Identify affected paths, commits, remotes, caches, CI artifacts, and forks without copying the sensitive content into tickets or chat.
3. Remove the files from the current tree immediately.
4. Decide with repository owners whether coordinated history rewriting is required.
5. If approved, purge with a history-rewrite tool, force-update every remote, invalidate old clones and artifacts, and document the response.
6. Rotate credentials if any were exposed and review how the file bypassed repository controls.

History rewriting is intentionally not part of ordinary cleanup because it disrupts every clone and requires explicit coordination.
