from __future__ import annotations

from hmac import compare_digest

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from ..application.security.browser_credentials import BrowserCredentialError

from ..infrastructure.rate_limit.api import ApiKeyRateLimiter


_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    request: Request,
    supplied_key: str | None = Security(_API_KEY_HEADER),
) -> None:
    _verify_api_key(request, supplied_key)


def require_session_access(request: Request, session_id: str) -> None:
    """Allow API-key callers or a browser credential bound to this session."""
    supplied_key = request.headers.get("X-API-Key")
    if supplied_key is not None:
        _verify_api_key(request, supplied_key)
        return
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token or " " in token:
        _browser_unauthorized()
    credentials = request.app.state.browser_credentials
    try:
        credential_id = credentials.authorize(token, session_id, safe_read=request.method == "GET")
    except BrowserCredentialError:
        _browser_unauthorized()
    limiter = request.app.state.browser_rate_limiter
    retry_after = limiter.allow(credential_id)
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "RATE_LIMITED", "message": "Request rate limit exceeded"},
            headers={"Retry-After": str(retry_after)},
        )


def _verify_api_key(request: Request, supplied_key: str | None) -> None:
    expected = request.app.state.settings.api_key.get_secret_value()
    if supplied_key is None or not compare_digest(supplied_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Authentication failed"},
            headers={"WWW-Authenticate": "ApiKey"},
        )
    limiter: ApiKeyRateLimiter = request.app.state.rate_limiter
    retry_after = limiter.allow()
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "RATE_LIMITED", "message": "Request rate limit exceeded"},
            headers={"Retry-After": str(retry_after)},
        )


def _browser_unauthorized() -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "UNAUTHORIZED", "message": "Authentication failed"},
        headers={"WWW-Authenticate": "Bearer"},
    )
