# KYC Integration Sandbox

The `web` module is a small Go application that behaves like a customer KYC backend for end-to-end hosted-verifier testing. It keeps `KYC_API_KEY` on the server and uses it only for session creation, browser-credential issuance, safe-session reads, normalized-result retrieval, and deletion.

The browser never receives the API key. It receives an opaque browser credential only in the intended hosted URL fragment, for example `/verify/{session_id}#bt_...`. The hosted verifier remains the end-user UI; the sandbox only opens or copies that link.

The sandbox keeps a cookie-scoped list of recent test sessions in memory until they are deleted, expire, or the Go service restarts. It never stores images or processing artifacts. Its in-memory records contain safe session state, webhook history, integration timestamps, the active hosted URL, and fetched normalized results.

For local development:

```powershell
Set-Location web
npm install --no-package-lock --ignore-scripts
npm run build:assets
$env:KYC_API_KEY = "replace-with-a-long-random-secret"
$env:KYC_WEB_API_URL = "http://127.0.0.1:8000"
$env:KYC_WEBHOOK_SECRET = "replace-with-a-long-random-secret-of-at-least-32-characters"
go run .
```

The sandbox listens on `:8080` by default. Set `KYC_WEB_COOKIE_SECURE=true` when it is served behind HTTPS. Signed lifecycle webhooks arrive at `POST /webhooks`; the sandbox verifies their HMAC before recording or showing anything.

For expiry scenarios, override the existing API settings only in a local development environment, for example `KYC_SESSION_TTL_SECONDS=120` and `KYC_BROWSER_TOKEN_TTL_SECONDS=180`. Keep the browser credential TTL slightly longer than the session TTL when testing the API's expiry observation path; do not change production defaults for sandbox testing.
