# KYC API beta contract

`/v1` is the public HTTP compatibility boundary. It is a technical-verification
workflow API, not an identity approval API: it never returns `verified`,
`rejected`, a match/no-match decision, or a face-similarity score.

## Session state

Every session response has `status`, `created_at`, `expires_at`, a derived
`next_action`, and summaries for `document`, `liveness`, and
`face_comparison`. `next_action` is never stored independently and is one of
`submit_document_front`, `submit_document_back`, `submit_liveness`, `wait`, or
`null`.

The server calculates `status` from the checks required for that session:

1. Any required terminal failure makes the session `failed`.
2. Every required successful terminal state makes it `completed`.
3. Every other combination is `in_progress`.

Face comparison is required only when it is configured for that verification.
When it is required, only `face_comparison.status: "completed"` succeeds and
`"failed"` terminates the session. `"not_available"` means comparison was not
configured for this verification and is excluded from the aggregation; it is
never treated as successful when comparison is required.

The extraction engine reports `partial` when required document evidence is
missing, invalid, conflicting, or errored. The detailed result retains
`document.status: "partial"` for diagnostics, but the public session treats
that outcome as a terminal document failure: `document.status: "failed"` and
session `status: "failed"`. It cannot complete a verification.

| Authoritative state | Public status | `next_action` |
| --- | --- | --- |
| Front absent | `in_progress` | `submit_document_front` |
| Front present, back absent | `in_progress` | `submit_document_back` |
| Both captures accepted; document work pending or running | `in_progress` | `wait` |
| Document complete; required liveness not submitted | `in_progress` | `submit_liveness` |
| Required liveness or comparison running | `in_progress` | `wait` |
| All required checks technically complete | `completed` | `null` |
| Any required check terminally failed | `failed` | `null` |

## Session examples

New session:

```json
{"session_id":"vs_123","status":"in_progress","next_action":"submit_document_front","document":{"status":"awaiting_capture","front_capture":"missing","back_capture":"missing","result_available":false},"liveness":{"status":"not_started"},"face_comparison":{"status":"not_available"}}
```

Front captured:

```json
{"session_id":"vs_123","status":"in_progress","next_action":"submit_document_back","document":{"status":"awaiting_capture","front_capture":"accepted","back_capture":"missing","result_available":false},"liveness":{"status":"not_started"},"face_comparison":{"status":"not_available"}}
```

Document processing:

```json
{"session_id":"vs_123","status":"in_progress","next_action":"wait","document":{"status":"processing","front_capture":"accepted","back_capture":"accepted","result_available":false},"liveness":{"status":"not_started"},"face_comparison":{"status":"not_available"}}
```

Document complete and liveness pending:

```json
{"session_id":"vs_123","status":"in_progress","next_action":"submit_liveness","document":{"status":"completed","front_capture":"accepted","back_capture":"accepted","result_available":true},"liveness":{"status":"not_started"},"face_comparison":{"status":"not_available"}}
```

Liveness passed while required comparison runs:

```json
{"session_id":"vs_123","status":"in_progress","next_action":"wait","document":{"status":"completed","front_capture":"accepted","back_capture":"accepted","result_available":true},"liveness":{"status":"passed"},"face_comparison":{"status":"processing"}}
```

All required technical checks complete:

```json
{"session_id":"vs_123","status":"completed","next_action":null,"document":{"status":"completed","front_capture":"accepted","back_capture":"accepted","result_available":true},"liveness":{"status":"passed"},"face_comparison":{"status":"completed"}}
```

Required subsystem failure:

```json
{"session_id":"vs_123","status":"failed","next_action":null,"document":{"status":"failed","front_capture":"accepted","back_capture":"accepted","result_available":false},"liveness":{"status":"not_started"},"face_comparison":{"status":"not_available"}}
```

An expired session has no session representation:

```json
{"error":{"code":"SESSION_EXPIRED","message":"Session has expired"}}
```

This response is HTTP `410`. Integrators create a new session after a terminal
failure or expiry; that instruction is not a `next_action` value.

## Endpoints and data

- `POST /v1/sessions` creates a session.
- `POST /v1/sessions/{id}/images/front|back` submits a document side.
- `POST /v1/sessions/{id}/liveness` submits the configured liveness frames.
- `POST /v1/sessions/{id}/process` is only an idempotent recovery operation for
  deferred document processing; browser integrations do not call it normally.
- `GET /v1/sessions/{id}` returns status without document PII.
- `GET /v1/sessions/{id}/result` returns canonical normalized document values
  once document processing has a terminal result.
- `DELETE /v1/sessions/{id}` is an idempotent no-op when the session no longer
  exists and returns `204`.

Result fields contain canonical normalized values and field status. The API
does not return raw OCR candidates, confidence/provenance internals, boxes,
detector/model metadata, processing IDs, timings, raw QR payloads, images,
portraits, artifacts, embeddings, or raw face similarity.

`Idempotency-Key` support is deferred as beta hardening; capture and liveness
replay semantics are intentionally not part of this contract.

## Webhooks and errors

Webhooks are PII-minimal and delivered at least once. Types are
`verification.session.created`, `verification.document.completed`,
`verification.document.failed`, `verification.liveness.passed`,
`verification.liveness.failed`, `verification.processing.completed`, and
`verification.processing.failed`.

```json
{"schema_version":"1","id":"evt_123","type":"verification.processing.completed","created_at":"2026-09-27T21:00:00Z","data":{"session_id":"vs_123","sequence":8,"status":"completed","next_action":null,"result_available":true}}
```

Consumers verify the existing HMAC headers, deduplicate by `id`, and retrieve
the session or result instead of expecting PII in a webhook. When a required
subsystem fails, its subsystem event is delivered before the paired
`verification.processing.failed` event for the same transition. The outbox
deduplicates by session, sequence, and event type, so both events coexist.
Errors always use `{"error":{"code":"...","message":"..."}}`; messages
are safe and do not contain internal exception detail.

## Browser credential boundary

The customer backend authenticates with `X-API-Key`, creates the session, and
later obtains a short-lived opaque credential scoped to that one session. A
browser credential may submit document/liveness material and read safe session
status, but may not retrieve results, delete sessions, or access another
session. Token format and issuance are intentionally not implemented here.
