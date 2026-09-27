from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from concurrent.futures import Executor
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from urllib.parse import quote
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.formparsers import MultiPartParser

from kyc_engine import (
    CaptureAssessmentInputError,
    DocumentCaptureAssessor,
    DocumentCoordinator,
    FaceRecognizer,
    LivenessEvaluator,
)
from kyc_engine.intake import ImageIntake
from kyc_engine.portrait_artifacts import InMemoryPortraitArtifactStore

from . import __version__
from .auth import require_api_key, require_session_access
from .browser_credentials import BrowserCredentialStore
from .browser_rate_limit import BrowserCredentialRateLimiter
from .composition import (
    create_capture_assessor,
    create_coordinator,
    create_face_recognizer,
    create_liveness_evaluator,
    create_liveness_intake,
)
from .jobs import JobCapacityExceeded, JobManager
from .face_comparison import FaceComparisonService
from .logging import configure_logging, logging_context
from .metrics import MetricsRegistry
from .orchestration import FaceMatchDispatcher, VerificationOrchestrator
from .middleware import (
    MetricsMiddleware,
    RequestBodyLimitMiddleware,
    RequestContextMiddleware,
)
from .telemetry import (
    build_otlp_export_configuration,
    configure_metrics,
    configure_tracing,
    flush_metrics,
    flush_tracing,
)
from .models import (
    CaptureIssueResponse,
    CaptureResponse,
    BrowserTokenResponse,
    DocumentSide,
    ErrorResponse,
    HealthResponse,
    JobStatusResponse,
    LivenessSubmissionResponse,
    MetricsResponse,
    SessionResponse,
    VerificationResultResponse,
)
from .projection import result_response, session_response
from .rate_limit import ApiKeyRateLimiter
from .sessions import (
    SessionCapacityExceeded,
    SessionConflict,
    SessionExpired,
    SessionNotFound,
    SessionStore,
    SessionStoreError,
)
from .settings import ApiSettings
from .verification import LivenessSubmissionError, VerificationManager
from .webhooks import WebhookDispatcher, WebhookOutbox


_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: ApiSettings | None = None,
    coordinator: DocumentCoordinator | None = None,
    capture_assessor: DocumentCaptureAssessor | None = None,
    liveness_evaluator: LivenessEvaluator | None = None,
    liveness_intake: ImageIntake | None = None,
    face_recognizer: FaceRecognizer | None = None,
    session_store: SessionStore | None = None,
    executor: Executor | None = None,
    face_match_executor: Executor | None = None,
    metrics: MetricsRegistry | None = None,
    rate_limiter: ApiKeyRateLimiter | None = None,
    browser_rate_limiter: BrowserCredentialRateLimiter | None = None,
    webhook_dispatcher: WebhookDispatcher | None = None,
    portrait_artifacts: InMemoryPortraitArtifactStore | None = None,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings or ApiSettings()  # type: ignore[call-arg]
        face_match_dispatcher: FaceMatchDispatcher | None = None
        configure_logging(
            level=resolved_settings.log_level,
            environment=resolved_settings.environment,
        )
        try:
            telemetry_export = build_otlp_export_configuration(resolved_settings)
            configure_tracing(
                environment=resolved_settings.environment,
                service_version=__version__,
                export=telemetry_export,
            )
            configure_metrics(
                environment=resolved_settings.environment,
                service_version=__version__,
                export=telemetry_export,
            )
            configured_artifact_stores = [
                store
                for store in (
                    portrait_artifacts,
                    getattr(coordinator, "portrait_artifacts", None),
                    getattr(session_store, "portrait_artifacts", None),
                )
                if store is not None
            ]
            if len({id(store) for store in configured_artifact_stores}) > 1:
                raise RuntimeError(
                    "Coordinator, SessionStore, and explicit portrait artifact store "
                    "must share the same instance"
                )
            resolved_artifacts = (
                portrait_artifacts
                or getattr(coordinator, "portrait_artifacts", None)
                or getattr(session_store, "portrait_artifacts", None)
                or InMemoryPortraitArtifactStore(
                    pending_ttl_seconds=max(
                        60.0, resolved_settings.job_timeout_seconds + 5.0
                    )
                )
            )
            resolved_face_recognizer = face_recognizer or create_face_recognizer(
                resolved_settings
            )
            resolved_liveness_evaluator = (
                liveness_evaluator
                if liveness_evaluator is not None
                else create_liveness_evaluator(resolved_settings)
            )
            face_comparison_required = (
                resolved_face_recognizer is not None
                and resolved_liveness_evaluator is not None
            )
            resolved_store = session_store or SessionStore(
                ttl_seconds=resolved_settings.session_ttl_seconds,
                max_sessions=resolved_settings.max_sessions,
                portrait_artifacts=resolved_artifacts,
                liveness_required=resolved_liveness_evaluator is not None,
                face_match_enabled=face_comparison_required,
            )
            if session_store is not None:
                resolved_store.configure_requirements(
                    liveness_required=resolved_liveness_evaluator is not None,
                    face_match_required=face_comparison_required,
                )
            resolved_coordinator = coordinator or create_coordinator(
                resolved_settings, portrait_artifacts=resolved_artifacts
            )
            resolved_capture_assessor = capture_assessor or create_capture_assessor(
                resolved_settings
            )
            resolved_liveness_intake = liveness_intake or create_liveness_intake(
                resolved_settings
            )
            resolved_metrics = metrics or MetricsRegistry()
            resolved_face_comparison = (
                FaceComparisonService(
                    session_store=resolved_store,
                    recognizer=resolved_face_recognizer,
                    metrics=resolved_metrics,
                )
                if resolved_face_recognizer is not None
                else None
            )
            face_match_dispatcher = (
                FaceMatchDispatcher(
                    workers=resolved_settings.face_match_workers,
                    capacity=resolved_settings.max_sessions,
                    executor=face_match_executor,
                )
                if resolved_face_comparison is not None
                else None
            )
            resolved_rate_limiter = rate_limiter or ApiKeyRateLimiter(
                max_requests=resolved_settings.rate_limit_requests,
                window_seconds=resolved_settings.rate_limit_window_seconds,
            )
            resolved_browser_rate_limiter = browser_rate_limiter or BrowserCredentialRateLimiter(
                max_requests=resolved_settings.browser_rate_limit_requests,
                window_seconds=resolved_settings.rate_limit_window_seconds,
            )
            browser_credentials = BrowserCredentialStore(
                sessions=resolved_store,
                ttl_seconds=resolved_settings.browser_token_ttl_seconds,
            )
            resolved_dispatcher = webhook_dispatcher
            if resolved_dispatcher is None and resolved_settings.webhook_enabled:
                assert resolved_settings.webhook_url is not None
                assert resolved_settings.webhook_secret is not None
                resolved_dispatcher = WebhookDispatcher(
                    outbox=WebhookOutbox(
                        resolved_settings.webhook_outbox_path,
                        retention_seconds=resolved_settings.webhook_retention_seconds,
                    ),
                    url=resolved_settings.webhook_url,
                    secret=resolved_settings.webhook_secret.get_secret_value(),
                    timeout_seconds=resolved_settings.webhook_timeout_seconds,
                )
                resolved_dispatcher.start()
            orchestrator = (
                VerificationOrchestrator(
                    store=resolved_store,
                    face_comparison_service=resolved_face_comparison,
                    dispatcher=face_match_dispatcher,
                    snapshot_publisher=(
                        resolved_dispatcher.enqueue
                        if resolved_dispatcher is not None
                        else None
                    ),
                )
                if resolved_face_comparison is not None
                else None
            )
            manager = JobManager(
                resolved_coordinator,
                resolved_store,
                workers=resolved_settings.job_workers,
                capacity=resolved_settings.max_sessions,
                timeout_seconds=resolved_settings.job_timeout_seconds,
                metrics=resolved_metrics,
                executor=executor,
                webhook_publisher=resolved_dispatcher.enqueue
                if resolved_dispatcher is not None
                else None,
                on_document_state_changed=(
                    orchestrator.on_document_state_changed
                    if orchestrator is not None
                    else None
                ),
            )
            verification_manager = VerificationManager(
                assessor=resolved_capture_assessor,
                store=resolved_store,
                jobs=manager,
                liveness_evaluator=resolved_liveness_evaluator,
                liveness_intake=resolved_liveness_intake,
                orchestrator=orchestrator,
                snapshot_publisher=(
                    resolved_dispatcher.enqueue
                    if resolved_dispatcher is not None
                    else None
                ),
            )
            if telemetry_export is not None:
                logger.info(
                    "OTLP telemetry export configured",
                    extra={
                        "event": "telemetry_export_configured",
                        "protocol": "http/protobuf",
                    },
                )
        except Exception as exc:
            if face_match_dispatcher is not None:
                face_match_dispatcher.shutdown()
            logger.exception(
                "application startup failed",
                extra={
                    "event": "application_startup_failed",
                    "exception_type": type(exc).__name__,
                },
            )
            raise

        old_spool_size = MultiPartParser.spool_max_size
        old_part_size = MultiPartParser.max_part_size
        MultiPartParser.spool_max_size = resolved_settings.max_upload_bytes + 1
        MultiPartParser.max_part_size = resolved_settings.max_upload_bytes + 1
        application.state.settings = resolved_settings
        application.state.sessions = resolved_store
        application.state.jobs = manager
        application.state.verifications = verification_manager
        application.state.metrics = resolved_metrics
        application.state.face_comparison = resolved_face_comparison
        application.state.orchestrator = orchestrator
        application.state.face_match_dispatcher = face_match_dispatcher
        application.state.rate_limiter = resolved_rate_limiter
        application.state.browser_rate_limiter = resolved_browser_rate_limiter
        application.state.browser_credentials = browser_credentials
        application.state.webhook_dispatcher = resolved_dispatcher
        cleanup_stop = asyncio.Event()
        cleanup_task = asyncio.create_task(
            _cleanup_sessions(
                resolved_store,
                browser_credentials,
                cleanup_stop,
                resolved_settings.session_cleanup_interval_seconds,
            )
        )
        logger.info("application started", extra={"event": "application_started"})
        try:
            yield
        finally:
            cleanup_stop.set()
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
            manager.shutdown()
            if face_match_dispatcher is not None:
                face_match_dispatcher.shutdown()
            if resolved_dispatcher is not None:
                resolved_dispatcher.shutdown()
            telemetry_timeout_millis = round(
                resolved_settings.otel_export_timeout_seconds * 1000
            )
            flush_tracing(timeout_millis=telemetry_timeout_millis)
            flush_metrics(timeout_millis=telemetry_timeout_millis)
            MultiPartParser.spool_max_size = old_spool_size
            MultiPartParser.max_part_size = old_part_size

    application = FastAPI(
        title="Angolan KYC API",
        version=__version__,
        description=(
            "External beta API for technical document, liveness, and optional "
            "face-comparison workflows. Completion is not an identity approval."
        ),
        openapi_tags=[
            {
                "name": "Verification sessions",
                "description": "Create, advance, inspect, retrieve, and delete KYC sessions.",
            },
            {
                "name": "Operations",
                "description": "Authenticated service diagnostics.",
            },
        ],
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
    application.add_middleware(RequestContextMiddleware)
    _install_error_handlers(application)

    @application.get("/verify/{session_id}", include_in_schema=False)
    def hosted_verifier(session_id: str) -> FileResponse:
        # The session id is routing-only. The fragment token is never received here.
        return FileResponse(_verifier_asset("index.html"), headers=_verifier_headers())

    @application.get("/verify/assets/{asset_path:path}", include_in_schema=False)
    def hosted_verifier_asset(asset_path: str) -> FileResponse:
        try:
            asset = _verifier_asset(asset_path)
        except ValueError:
            raise _problem(404, "NOT_FOUND", "Resource was not found") from None
        if not asset.is_file():
            raise _problem(404, "NOT_FOUND", "Resource was not found")
        media_type = "application/javascript" if asset.suffix == ".ts" else None
        return FileResponse(asset, headers=_verifier_headers(), media_type=media_type)
    @application.get("/healthz", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @application.get(
        "/v1/metrics",
        response_model=MetricsResponse,
        tags=["Operations"],
        summary="Retrieve aggregate service metrics",
        dependencies=[Depends(require_api_key)],
        responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    def get_metrics(request: Request) -> MetricsResponse:
        return MetricsResponse(
            version=__version__, **request.app.state.metrics.snapshot()
        )

    @application.post(
        "/v1/sessions",
        response_model=SessionResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Verification sessions"],
        summary="Create a verification session",
        description="Creates one technical verification workflow. The returned `next_action` guides the next current-session operation.",
        dependencies=[Depends(require_api_key)],
        responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    def create_session(request: Request) -> SessionResponse:
        created = request.app.state.sessions.create()
        dispatcher = request.app.state.webhook_dispatcher
        if dispatcher is not None:
            dispatcher.enqueue(created, "verification.session.created")
        return session_response(created)

    @application.post(
        "/v1/sessions/{session_id}/browser-token",
        response_model=BrowserTokenResponse,
        tags=["Verification sessions"],
        summary="Create a hosted browser verification credential",
        description="Creates one short-lived opaque credential scoped to this session. The credential appears only in the hosted URL fragment.",
        dependencies=[Depends(require_api_key)],
        responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 410: {"model": ErrorResponse}},
    )
    def create_browser_token(session_id: str, request: Request) -> BrowserTokenResponse:
        with _session_logging_context(request, session_id):
            token, expires_at = request.app.state.browser_credentials.issue(session_id)
            base = request.app.state.settings.public_base_url.rstrip("/")
            return BrowserTokenResponse(
                verification_url=f"{base}/verify/{quote(session_id, safe='')}#{token}",
                expires_at=expires_at,
            )

    @application.post(
        "/v1/sessions/{session_id}/images/{side}",
        response_model=CaptureResponse,
        tags=["Verification sessions"],
        summary="Submit a document side",
        description="Submits a labelled front or back image. Quality rejection is a successful response with `accepted: false` and retry-safe issues.",
        dependencies=[Depends(require_session_access)],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            410: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
        },
    )
    async def upload_image(
        session_id: str,
        side: DocumentSide,
        request: Request,
        image: UploadFile = File(...),
    ) -> CaptureResponse:
        with _session_logging_context(request, session_id):
            _validate_upload_metadata(image)
            limit = request.app.state.settings.max_upload_bytes
            try:
                content = await image.read(limit + 1)
            finally:
                await image.close()
            if len(content) > limit:
                raise _problem(
                    413, "UPLOAD_TOO_LARGE", "Uploaded image exceeds the size limit"
                )
            if not content or not _has_supported_signature(content, image.content_type):
                raise _problem(
                    415,
                    "UNSUPPORTED_IMAGE",
                    "Uploaded content is not a supported image",
                )

            try:
                submission = request.app.state.verifications.submit_document_capture(
                    session_id, side, content
                )
            except CaptureAssessmentInputError as exc:
                raise _capture_input_problem(exc.code) from exc
            snapshot = submission.snapshot
            return CaptureResponse(
                session_id=session_id,
                side=side,
                accepted=submission.assessment.accepted,
                issues=tuple(
                    CaptureIssueResponse(code=issue.code.value, message=issue.message)
                    for issue in submission.assessment.issues
                ),
                session=session_response(snapshot),
            )

    @application.post(
        "/v1/sessions/{session_id}/liveness",
        response_model=LivenessSubmissionResponse,
        tags=["Verification sessions"],
        summary="Submit liveness frames",
        description="Submits the configured liveness frame set and returns the updated public session without biometric scores or artifacts.",
        dependencies=[Depends(require_session_access)],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            410: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def submit_liveness(
        session_id: str,
        request: Request,
        frames: list[UploadFile] = File(...),
    ) -> LivenessSubmissionResponse:
        with _session_logging_context(request, session_id):
            try:
                expected_frames = request.app.state.verifications.liveness_frame_count
                if len(frames) != expected_frames:
                    raise _problem(
                        422,
                        "LIVENESS_FRAME_COUNT",
                        "Incorrect number of liveness frames",
                    )
                content = []
                limit = request.app.state.settings.max_liveness_frame_bytes
                for frame in frames:
                    _validate_upload_metadata(frame)
                    data = await frame.read(limit + 1)
                    if len(data) > limit:
                        raise _problem(
                            413,
                            "UPLOAD_TOO_LARGE",
                            "Uploaded image exceeds the size limit",
                        )
                    if not data or not _has_supported_signature(
                        data, frame.content_type
                    ):
                        raise _problem(
                            415,
                            "UNSUPPORTED_IMAGE",
                            "Uploaded content is not a supported image",
                        )
                    content.append(data)
            finally:
                for frame in frames:
                    await frame.close()

            submission = request.app.state.verifications.submit_liveness(session_id, content)
            return LivenessSubmissionResponse(session=session_response(submission.snapshot))

    @application.post(
        "/v1/sessions/{session_id}/process",
        response_model=JobStatusResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Verification sessions"],
        summary="Recover deferred document processing",
        description="Recovery-only operation for an accepted capture left pending by capacity. Normal browser flows rely on automatic processing.",
        dependencies=[Depends(require_api_key)],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            410: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    def process_session(session_id: str, request: Request) -> JobStatusResponse:
        with _session_logging_context(request, session_id):
            snapshot = request.app.state.verifications.start_document_processing(
                session_id
            )
            with logging_context(job_id=snapshot.document.job_id):
                logger.info(
                    "document processing queued",
                    extra={"event": "document.processing_queued"},
                )
            return JobStatusResponse(session=session_response(snapshot))

    @application.get(
        "/v1/sessions/{session_id}",
        response_model=SessionResponse,
        tags=["Verification sessions"],
        summary="Retrieve public session state",
        description="Returns status and the derived `next_action`, without extracted document values or biometric details.",
        dependencies=[Depends(require_session_access)],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            410: {"model": ErrorResponse},
        },
    )
    def get_session(session_id: str, request: Request) -> SessionResponse:
        with _session_logging_context(request, session_id):
            return session_response(request.app.state.sessions.get(session_id))

    @application.get(
        "/v1/sessions/{session_id}/result",
        response_model=VerificationResultResponse,
        tags=["Verification sessions"],
        summary="Retrieve the normalized document result",
        description="Available after document processing reaches a terminal result. Raw OCR, QR, engine, and biometric details are excluded.",
        dependencies=[Depends(require_api_key)],
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            410: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def get_result(session_id: str, request: Request) -> VerificationResultResponse:
        with _session_logging_context(request, session_id):
            snapshot = request.app.state.sessions.get(session_id)
            result = request.app.state.sessions.result(session_id)
            return result_response(snapshot, result)

    @application.delete(
        "/v1/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["Verification sessions"],
        summary="Delete a verification session",
        description="Idempotently clears the session and its retained sensitive state.",
        dependencies=[Depends(require_api_key)],
        responses={401: {"model": ErrorResponse}},
    )
    def delete_session(session_id: str, request: Request) -> Response:
        with _session_logging_context(request, session_id):
            request.app.state.browser_credentials.revoke_session(session_id)
            request.app.state.sessions.delete(session_id)
            return Response(status_code=status.HTTP_204_NO_CONTENT)

    return application


def _validate_upload_metadata(image: UploadFile) -> None:
    suffix = Path(image.filename or "").suffix.lower()
    if (
        image.content_type not in _ALLOWED_CONTENT_TYPES
        or suffix not in _ALLOWED_SUFFIXES
    ):
        raise _problem(
            415, "UNSUPPORTED_IMAGE", "Only JPEG and PNG uploads are supported"
        )
    if image.content_type == "image/png" and suffix != ".png":
        raise _problem(
            415, "UNSUPPORTED_IMAGE", "Image type and extension do not match"
        )
    if image.content_type == "image/jpeg" and suffix not in {".jpg", ".jpeg"}:
        raise _problem(
            415, "UNSUPPORTED_IMAGE", "Image type and extension do not match"
        )


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


def _capture_input_problem(code: str) -> HTTPException:
    if code in {"INPUT_TOO_LARGE", "IMAGE_TOO_LARGE"}:
        return _problem(
            413, "IMAGE_TOO_LARGE", "Uploaded image exceeds the supported limits"
        )
    if code in {"DECODE_FAILED", "UNSUPPORTED_FORMAT"}:
        return _problem(
            415, "UNSUPPORTED_IMAGE", "Uploaded content is not a supported image"
        )
    return _problem(422, "INVALID_IMAGE", "Uploaded image is invalid")


def _install_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        code = str(detail.get("code", "HTTP_ERROR"))
        message = str(detail.get("message", "The request could not be completed"))
        _record_error(request, code)
        return _error_response(
            exc.status_code,
            code,
            message,
            headers=exc.headers,
            request_id=_request_id(request),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        _record_error(request, "INVALID_REQUEST")
        return _error_response(
            422,
            "INVALID_REQUEST",
            "The request is invalid",
            request_id=_request_id(request),
        )

    @application.exception_handler(SessionNotFound)
    async def session_not_found(
        request: Request, _exc: SessionNotFound
    ) -> JSONResponse:
        _record_error(request, "SESSION_NOT_FOUND")
        return _error_response(
            404,
            "SESSION_NOT_FOUND",
            "Session was not found",
            request_id=_request_id(request),
        )

    @application.exception_handler(SessionExpired)
    async def session_expired(
        request: Request, _exc: SessionExpired
    ) -> JSONResponse:
        _record_error(request, "SESSION_EXPIRED")
        return _error_response(
            410,
            "SESSION_EXPIRED",
            "Session has expired",
            request_id=_request_id(request),
        )

    @application.exception_handler(SessionConflict)
    async def session_conflict(request: Request, exc: SessionConflict) -> JSONResponse:
        _record_error(request, exc.code)
        return _error_response(
            409,
            exc.code,
            "The session cannot perform this operation in its current state",
            request_id=_request_id(request),
        )

    @application.exception_handler(SessionCapacityExceeded)
    async def session_capacity(
        request: Request, _exc: SessionCapacityExceeded
    ) -> JSONResponse:
        _record_error(request, "SESSION_CAPACITY_EXCEEDED")
        return _error_response(
            429,
            "SESSION_CAPACITY_EXCEEDED",
            "Session capacity has been reached",
            request_id=_request_id(request),
        )

    @application.exception_handler(JobCapacityExceeded)
    async def job_capacity(request: Request, _exc: JobCapacityExceeded) -> JSONResponse:
        _record_error(request, "JOB_CAPACITY_EXCEEDED")
        return _error_response(
            429,
            "JOB_CAPACITY_EXCEEDED",
            "Job capacity has been reached",
            request_id=_request_id(request),
        )

    @application.exception_handler(LivenessSubmissionError)
    async def liveness_submission_error(
        request: Request, exc: LivenessSubmissionError
    ) -> JSONResponse:
        _record_error(request, exc.code)
        return _error_response(
            exc.status_code,
            exc.code,
            str(exc),
            request_id=_request_id(request),
        )

    @application.exception_handler(SessionStoreError)
    async def session_error(request: Request, _exc: SessionStoreError) -> JSONResponse:
        _record_error(request, "JOB_FAILED")
        return _error_response(
            500,
            "JOB_FAILED",
            "Document extraction failed",
            request_id=_request_id(request),
        )

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        _record_error(request, "INTERNAL_ERROR")
        with logging_context(
            request_id=_request_id(request),
            session_id=_session_id(request),
        ):
            logger.exception(
                "unexpected HTTP request failure",
                extra={
                    "event": "http_request_failed",
                    "error_code": "INTERNAL_ERROR",
                    "exception_type": type(exc).__name__,
                },
            )
        return _error_response(
            500,
            "INTERNAL_ERROR",
            "The request could not be completed",
            request_id=_request_id(request),
        )


def _record_error(request: Request, code: str) -> None:
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.record_error(code)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _session_id(request: Request) -> str | None:
    return getattr(request.state, "session_id", None)


@contextmanager
def _session_logging_context(request: Request, session_id: str):
    request.state.session_id = session_id
    with logging_context(session_id=session_id):
        yield


async def _cleanup_sessions(
    store: SessionStore,
    browser_credentials: BrowserCredentialStore,
    stop: asyncio.Event,
    interval_seconds: int,
) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            store.cleanup()
            browser_credentials.cleanup()


def _verifier_asset(name: str) -> Path:
    source_root = Path(__file__).parent / "verify"
    root = (source_root / "dist" if (source_root / "dist").is_dir() else source_root).resolve()
    asset = (root / name).resolve()
    if root not in asset.parents and asset != root:
        raise ValueError("verifier asset path escapes the static root")
    return asset


def _verifier_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": (
            "default-src 'self'; base-uri 'self'; form-action 'self'; "
            "frame-ancestors 'none'; object-src 'none'; connect-src 'self'; "
            "img-src 'self' blob: data:; media-src 'self' blob:; "
            "script-src 'self'; style-src 'self'"
        ),
    }


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    if request_id is not None:
        response_headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers=response_headers or None,
    )


app = create_app()
