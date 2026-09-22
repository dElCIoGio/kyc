from __future__ import annotations

from hmac import compare_digest

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from .rate_limit import ApiKeyRateLimiter


_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    request: Request,
    supplied_key: str | None = Security(_API_KEY_HEADER),
) -> None:
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
