from __future__ import annotations

import json
from uuid import uuid4
from time import perf_counter
from typing import Awaitable, Callable

from .metrics import MetricsRegistry
from .logging import logging_context


Receive = Callable[[], Awaitable[dict[str, object]]]
Send = Callable[[dict[str, object]], Awaitable[None]]
AsgiApp = Callable[[dict[str, object], Receive, Send], Awaitable[None]]


class RequestContextMiddleware:
    """Bind a server-generated request identifier for the complete HTTP request."""

    def __init__(self, app: AsgiApp) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, object], receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        async def add_request_id(message: dict[str, object]) -> None:
            if message.get("type") == "http.response.start":
                headers = [
                    (bytes(key), bytes(value))
                    for key, value in message.get("headers", [])  # type: ignore[arg-type]
                    if bytes(key).lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        with logging_context(request_id=request_id):
            await self.app(scope, receive, add_request_id)


class RequestBodyLimitMiddleware:
    """Bound complete API request bodies before multipart parsing begins."""

    def __init__(
        self,
        app: AsgiApp,
        *,
        limit_provider: Callable[[], int],
        metrics_provider: Callable[[], MetricsRegistry],
    ) -> None:
        self.app = app
        self._limit_provider = limit_provider
        self._metrics_provider = metrics_provider

    async def __call__(self, scope: dict[str, object], receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or not str(scope.get("path", "")).startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        started = perf_counter()
        limit = self._limit_provider()
        headers = {bytes(key).lower(): bytes(value) for key, value in scope.get("headers", [])}  # type: ignore[arg-type]
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self._reject(send, 400, "INVALID_REQUEST", "The request is invalid", started)
                return
            if declared_length > limit:
                await self._reject(send, 413, "REQUEST_TOO_LARGE", "Request body exceeds the size limit", started)
                return

        body = bytearray()
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                return
            if message_type != "http.request":
                continue
            body.extend(bytes(message.get("body", b"")))
            if len(body) > limit:
                await self._reject(send, 413, "REQUEST_TOO_LARGE", "Request body exceeds the size limit", started)
                return
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay_receive() -> dict[str, object]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_receive, send)

    async def _reject(
        self,
        send: Send,
        status_code: int,
        code: str,
        message: str,
        started: float,
    ) -> None:
        metrics = self._metrics_provider()
        metrics.record_error(code)
        metrics.record_response(status_code, (perf_counter() - started) * 1000.0)
        await _send_problem(send, status_code, code, message)


class MetricsMiddleware:
    def __init__(self, app: AsgiApp, *, metrics_provider: Callable[[], MetricsRegistry]) -> None:
        self.app = app
        self._metrics_provider = metrics_provider

    async def __call__(self, scope: dict[str, object], receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        status_code = 500

        async def observing_send(message: dict[str, object]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, observing_send)
        finally:
            self._metrics_provider().record_response(
                status_code,
                (perf_counter() - started) * 1000.0,
            )


async def _send_problem(send: Send, status_code: int, code: str, message: str) -> None:
    payload = json.dumps({"error": {"code": code, "message": message}}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})
