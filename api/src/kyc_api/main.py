from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import Executor
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.formparsers import MultiPartParser

from kyc_engine import DocumentCoordinator

from . import __version__
from .auth import require_api_key
from .composition import create_coordinator
from .jobs import JobCapacityExceeded, JobManager
from .metrics import MetricsRegistry
from .middleware import MetricsMiddleware, RequestBodyLimitMiddleware
from .models import (
    DeleteResponse,
    DocumentSide,
    ErrorResponse,
    HealthResponse,
    JobStatusResponse,
    MetricsResponse,
    SessionResponse,
    UploadResponse,
)
from .rate_limit import ApiKeyRateLimiter
from .sessions import (
    SessionCapacityExceeded,
    SessionConflict,
    SessionNotFound,
    SessionSnapshot,
    SessionStore,
    SessionStoreError,
)
from .settings import ApiSettings


_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def create_app(
    *,
    settings: ApiSettings | None = None,
    coordinator: DocumentCoordinator | None = None,
    session_store: SessionStore | None = None,
    executor: Executor | None = None,
    metrics: MetricsRegistry | None = None,
    rate_limiter: ApiKeyRateLimiter | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings or ApiSettings()  # type: ignore[call-arg]
        resolved_store = session_store or SessionStore(
            ttl_seconds=resolved_settings.session_ttl_seconds,
            max_sessions=resolved_settings.max_sessions,
        )
        resolved_coordinator = coordinator or create_coordinator(resolved_settings)
        resolved_metrics = metrics or MetricsRegistry()
        resolved_rate_limiter = rate_limiter or ApiKeyRateLimiter(
            max_requests=resolved_settings.rate_limit_requests,
            window_seconds=resolved_settings.rate_limit_window_seconds,
        )
        manager = JobManager(
            resolved_coordinator,
            resolved_store,
            workers=resolved_settings.job_workers,
            capacity=resolved_settings.max_sessions,
            timeout_seconds=resolved_settings.job_timeout_seconds,
            metrics=resolved_metrics,
            executor=executor,
        )

        old_spool_size = MultiPartParser.spool_max_size
        old_part_size = MultiPartParser.max_part_size
        MultiPartParser.spool_max_size = resolved_settings.max_upload_bytes + 1
        MultiPartParser.max_part_size = resolved_settings.max_upload_bytes + 1
        application.state.settings = resolved_settings
        application.state.sessions = resolved_store
        application.state.jobs = manager
        application.state.metrics = resolved_metrics
        application.state.rate_limiter = resolved_rate_limiter
        cleanup_stop = asyncio.Event()
        cleanup_task = asyncio.create_task(
            _cleanup_sessions(
                resolved_store,
                cleanup_stop,
                resolved_settings.session_cleanup_interval_seconds,
            )
        )
        try:
            yield
        finally:
            cleanup_stop.set()
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
            manager.shutdown()
            MultiPartParser.spool_max_size = old_spool_size
            MultiPartParser.max_part_size = old_part_size

    application = FastAPI(
        title="Angolan KYC API",
        version=__version__,
        lifespan=lifespan,
    )
    application.add_middleware(
        MetricsMiddleware,
        metrics_provider=lambda: application.state.metrics,
    )
    application.add_middleware(
        RequestBodyLimitMiddleware,
        limit_provider=lambda: application.state.settings.max_request_bytes,
        metrics_provider=lambda: application.state.metrics,
    )
    _install_error_handlers(application)

    @application.get("/healthz", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @application.get(
        "/v1/metrics",
        response_model=MetricsResponse,
        dependencies=[Depends(require_api_key)],
        responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    def get_metrics(request: Request) -> MetricsResponse:
        return MetricsResponse(version=__version__, **request.app.state.metrics.snapshot())

    @application.post(
        "/v1/sessions",
        response_model=SessionResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_api_key)],
        responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    def create_session(request: Request) -> SessionResponse:
        return _session_response(request.app.state.sessions.create())

    @application.post(
        "/v1/sessions/{session_id}/images/{side}",
        response_model=UploadResponse,
        dependencies=[Depends(require_api_key)],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
        },
    )
    async def upload_image(
        session_id: str,
        side: DocumentSide,
        request: Request,
        image: UploadFile = File(...),
    ) -> UploadResponse:
        _validate_upload_metadata(image)
        limit = request.app.state.settings.max_upload_bytes
        try:
            content = await image.read(limit + 1)
        finally:
            await image.close()
        if len(content) > limit:
            raise _problem(413, "UPLOAD_TOO_LARGE", "Uploaded image exceeds the size limit")
        if not content or not _has_supported_signature(content, image.content_type):
            raise _problem(415, "UNSUPPORTED_IMAGE", "Uploaded content is not a supported image")

        snapshot = request.app.state.sessions.upload(session_id, side, content)
        return UploadResponse(
            session_id=snapshot.session_id,
            status=snapshot.status,
            side=side,
            size_bytes=len(content),
            uploaded_sides=snapshot.uploaded_sides,
            expires_at=snapshot.expires_at,
        )

    @application.post(
        "/v1/sessions/{session_id}/process",
        response_model=JobStatusResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    def process_session(session_id: str, request: Request) -> JobStatusResponse:
        snapshot = request.app.state.jobs.submit(session_id)
        assert snapshot.job_id is not None
        return JobStatusResponse(
            session_id=snapshot.session_id,
            job_id=snapshot.job_id,
            status=snapshot.status,
        )

    @application.get(
        "/v1/sessions/{session_id}",
        response_model=SessionResponse,
        dependencies=[Depends(require_api_key)],
        responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )
    def get_session(session_id: str, request: Request) -> SessionResponse:
        return _session_response(request.app.state.sessions.get(session_id))

    @application.get(
        "/v1/sessions/{session_id}/result",
        dependencies=[Depends(require_api_key)],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def get_result(session_id: str, request: Request) -> JSONResponse:
        result = request.app.state.sessions.result(session_id)
        return JSONResponse(content=result.to_dict())

    @application.delete(
        "/v1/sessions/{session_id}",
        response_model=DeleteResponse,
        dependencies=[Depends(require_api_key)],
        responses={401: {"model": ErrorResponse}},
    )
    def delete_session(session_id: str, request: Request) -> DeleteResponse:
        request.app.state.sessions.delete(session_id)
        return DeleteResponse()

    return application


def _session_response(snapshot: SessionSnapshot) -> SessionResponse:
    return SessionResponse(
        session_id=snapshot.session_id,
        status=snapshot.status,
        created_at=snapshot.created_at,
        expires_at=snapshot.expires_at,
        job_id=snapshot.job_id,
        uploaded_sides=snapshot.uploaded_sides,
        result_available=snapshot.result_available,
    )


def _validate_upload_metadata(image: UploadFile) -> None:
    suffix = Path(image.filename or "").suffix.lower()
    if image.content_type not in _ALLOWED_CONTENT_TYPES or suffix not in _ALLOWED_SUFFIXES:
        raise _problem(415, "UNSUPPORTED_IMAGE", "Only JPEG and PNG uploads are supported")
    if image.content_type == "image/png" and suffix != ".png":
        raise _problem(415, "UNSUPPORTED_IMAGE", "Image type and extension do not match")
    if image.content_type == "image/jpeg" and suffix not in {".jpg", ".jpeg"}:
        raise _problem(415, "UNSUPPORTED_IMAGE", "Image type and extension do not match")


def _has_supported_signature(content: bytes, content_type: str | None) -> bool:
    if content_type == "image/png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if content_type == "image/jpeg":
        return content.startswith(b"\xff\xd8\xff")
    return False


def _problem(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _install_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        code = str(detail.get("code", "HTTP_ERROR"))
        message = str(detail.get("message", "The request could not be completed"))
        _record_error(request, code)
        return _error_response(exc.status_code, code, message, headers=exc.headers)

    @application.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        _record_error(request, "INVALID_REQUEST")
        return _error_response(422, "INVALID_REQUEST", "The request is invalid")

    @application.exception_handler(SessionNotFound)
    async def session_not_found(request: Request, _exc: SessionNotFound) -> JSONResponse:
        _record_error(request, "SESSION_NOT_FOUND")
        return _error_response(404, "SESSION_NOT_FOUND", "Session was not found")

    @application.exception_handler(SessionConflict)
    async def session_conflict(request: Request, exc: SessionConflict) -> JSONResponse:
        _record_error(request, exc.code)
        return _error_response(409, exc.code, str(exc))

    @application.exception_handler(SessionCapacityExceeded)
    async def session_capacity(
        request: Request, _exc: SessionCapacityExceeded
    ) -> JSONResponse:
        _record_error(request, "SESSION_CAPACITY_EXCEEDED")
        return _error_response(429, "SESSION_CAPACITY_EXCEEDED", "Session capacity has been reached")

    @application.exception_handler(JobCapacityExceeded)
    async def job_capacity(request: Request, _exc: JobCapacityExceeded) -> JSONResponse:
        _record_error(request, "JOB_CAPACITY_EXCEEDED")
        return _error_response(429, "JOB_CAPACITY_EXCEEDED", "Job capacity has been reached")

    @application.exception_handler(SessionStoreError)
    async def session_error(request: Request, _exc: SessionStoreError) -> JSONResponse:
        _record_error(request, "JOB_FAILED")
        return _error_response(500, "JOB_FAILED", "Document extraction failed")

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, _exc: Exception) -> JSONResponse:
        _record_error(request, "INTERNAL_ERROR")
        return _error_response(500, "INTERNAL_ERROR", "The request could not be completed")


def _record_error(request: Request, code: str) -> None:
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.record_error(code)


async def _cleanup_sessions(
    store: SessionStore,
    stop: asyncio.Event,
    interval_seconds: int,
) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            store.cleanup()


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


app = create_app()
