# KYC Operator Console

The `web` module is the browser-facing Go service for the session-based KYC
API. It keeps `KYC_API_KEY` on the server, stores only opaque API-session
identifiers in memory, and never persists images or results.

For local development, compile the committed frontend assets after changing a
template or Tailwind source, then point the console to a running API:

```powershell
Set-Location web
npm install --no-package-lock --ignore-scripts
npm run build:assets
$env:KYC_API_KEY = "replace-with-a-long-random-secret"
$env:KYC_WEB_API_URL = "http://127.0.0.1:8000"
$env:KYC_WEBHOOK_SECRET = "replace-with-a-long-random-secret-of-at-least-32-characters"
go run .
```

The console listens on `:8080` by default. `KYC_WEB_MAX_UPLOAD_BYTES` defaults
to 15 MiB and should match the API's `KYC_MAX_UPLOAD_BYTES`. Set
`KYC_WEB_COOKIE_SECURE=true` when serving the console behind HTTPS.
`KYC_WEBHOOK_SECRET` is required and must match the API's
`KYC_WEBHOOK_SECRET`; signed session-lifecycle webhooks arrive at
`POST /webhooks` and are forwarded to the browser over `GET /events/stream`.
