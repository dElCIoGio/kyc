from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KYC_",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: SecretStr = Field(min_length=16)
    ocr_model_manifest: Path
    ocr_device: Literal["cpu", "gpu"] = "cpu"
    log_level: str = "INFO"
    environment: str = "development"
    max_upload_bytes: int = Field(default=15 * 1024 * 1024, gt=0)
    capture_min_sharpness: float = Field(default=25.0, ge=0)
    capture_min_brightness: float = Field(default=35.0, ge=0, le=255)
    capture_max_brightness: float = Field(default=220.0, ge=0, le=255)
    capture_min_contrast: float = Field(default=12.0, ge=0)
    capture_min_document_area_ratio: float = Field(default=0.12, gt=0, lt=1)
    capture_min_edge_margin_ratio: float = Field(default=0.015, ge=0, lt=0.5)
    capture_max_perspective_distortion: float = Field(default=0.30, gt=0, le=1)
    capture_max_glare_ratio: float = Field(default=0.035, gt=0, lt=1)
    liveness_enabled: bool = False
    liveness_model_root: Path | None = None
    liveness_frame_count: int = Field(default=3, gt=0)
    liveness_min_real_ratio: float = Field(default=2.0 / 3.0, gt=0.0, le=1.0)
    max_liveness_frame_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    face_recognition_enabled: bool = False
    face_recognition_model_root: Path | None = None
    face_recognition_model_id: str | None = None
    face_match_workers: int = Field(default=1, gt=0)
    nif_verification_enabled: bool = False
    nif_timeout_seconds: float = Field(default=20.0, gt=0, le=60)
    nif_standalone_queue_capacity: int = Field(default=8, gt=0)
    session_ttl_seconds: int = Field(default=30 * 60, gt=0)
    browser_token_ttl_seconds: int = Field(default=31 * 60, gt=0)
    browser_rate_limit_requests: int = Field(default=60, gt=0)
    public_base_url: str = "http://127.0.0.1:8000"
    max_sessions: int = Field(default=100, gt=0)
    job_workers: int = Field(default=1, gt=0)
    rate_limit_requests: int = Field(default=120, gt=0)
    rate_limit_window_seconds: int = Field(default=60, gt=0)
    job_timeout_seconds: int = Field(default=30, gt=0)
    session_cleanup_interval_seconds: int = Field(default=60, gt=0)
    otel_enabled: bool = False
    otel_endpoint: str | None = None
    otel_headers: SecretStr | None = None
    otel_trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    otel_metric_export_interval_seconds: float = Field(default=60.0, gt=0)
    otel_export_timeout_seconds: float = Field(default=10.0, gt=0)
    webhook_url: str | None = None
    webhook_secret: SecretStr | None = None
    webhook_outbox_path: Path = Path("webhook-outbox.sqlite3")
    webhook_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    webhook_retention_seconds: int = Field(default=24 * 60 * 60, gt=0)

    @model_validator(mode="after")
    def validate_otlp_export_configuration(self) -> "ApiSettings":
        """Reject invalid enabled OTLP configuration without exposing secrets."""
        if not self.otel_enabled:
            return self
        if not self.otel_endpoint:
            raise ValueError("KYC_OTEL_ENDPOINT is required when KYC_OTEL_ENABLED is true")

        parsed = urlsplit(self.otel_endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("KYC_OTEL_ENDPOINT must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "KYC_OTEL_ENDPOINT must not contain credentials; use KYC_OTEL_HEADERS"
            )
        if parsed.query or parsed.fragment:
            raise ValueError("KYC_OTEL_ENDPOINT must not contain a query or fragment")
        path = parsed.path.rstrip("/")
        if path.endswith("/v1/traces") or path.endswith("/v1/metrics"):
            raise ValueError(
                "KYC_OTEL_ENDPOINT must be a base endpoint, not a signal-specific OTLP URL"
            )
        return self

    @model_validator(mode="after")
    def validate_capture_assessment_configuration(self) -> "ApiSettings":
        if self.capture_min_brightness >= self.capture_max_brightness:
            raise ValueError("KYC_CAPTURE_MIN_BRIGHTNESS must be lower than KYC_CAPTURE_MAX_BRIGHTNESS")
        capture_values = (
            self.capture_min_sharpness,
            self.capture_min_brightness,
            self.capture_max_brightness,
            self.capture_min_contrast,
            self.capture_min_document_area_ratio,
            self.capture_min_edge_margin_ratio,
            self.capture_max_perspective_distortion,
            self.capture_max_glare_ratio,
        )
        if any(not isfinite(value) for value in capture_values):
            raise ValueError("Capture assessment thresholds must be finite")
        return self

    @model_validator(mode="after")
    def validate_liveness_configuration(self) -> "ApiSettings":
        if self.liveness_enabled and self.liveness_model_root is None:
            raise ValueError("KYC_LIVENESS_MODEL_ROOT is required when KYC_LIVENESS_ENABLED is true")
        if self.liveness_enabled and self.liveness_frame_count != 3:
            raise ValueError(
                "KYC_LIVENESS_FRAME_COUNT must equal 3 when KYC_LIVENESS_ENABLED is true"
            )
        return self

    @model_validator(mode="after")
    def validate_face_recognition_configuration(self) -> "ApiSettings":
        if not self.face_recognition_enabled:
            return self
        if self.face_recognition_model_root is None or self.face_recognition_model_id is None:
            raise ValueError(
                "KYC_FACE_RECOGNITION_MODEL_ROOT and KYC_FACE_RECOGNITION_MODEL_ID "
                "are required when KYC_FACE_RECOGNITION_ENABLED is true"
            )
        return self

    @model_validator(mode="after")
    def validate_face_match_worker_configuration(self) -> "ApiSettings":
        if self.face_match_workers > self.max_sessions:
            raise ValueError("KYC_FACE_MATCH_WORKERS must not exceed KYC_MAX_SESSIONS")
        return self

    @model_validator(mode="after")
    def validate_webhook_configuration(self) -> "ApiSettings":
        if bool(self.webhook_url) != bool(self.webhook_secret):
            raise ValueError("KYC_WEBHOOK_URL and KYC_WEBHOOK_SECRET must be configured together")
        if not self.webhook_url:
            return self
        if len(self.webhook_secret.get_secret_value()) < 32:  # type: ignore[union-attr]
            raise ValueError("KYC_WEBHOOK_SECRET must contain at least 32 characters")
        parsed = urlsplit(self.webhook_url)
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("KYC_WEBHOOK_URL must not contain credentials, a query, or a fragment")
        if not parsed.hostname or parsed.scheme not in {"http", "https"}:
            raise ValueError("KYC_WEBHOOK_URL must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname != "web":
            raise ValueError("KYC_WEBHOOK_URL must use HTTPS outside the internal web service")
        return self

    @model_validator(mode="after")
    def validate_public_base_url(self) -> "ApiSettings":
        parsed = urlsplit(self.public_base_url)
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("KYC_PUBLIC_BASE_URL must not contain credentials, a query, or a fragment")
        if not parsed.hostname or parsed.scheme not in {"http", "https"}:
            raise ValueError("KYC_PUBLIC_BASE_URL must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("KYC_PUBLIC_BASE_URL must use HTTPS outside localhost")
        return self

    @property
    def webhook_enabled(self) -> bool:
        return self.webhook_url is not None

    @property
    def max_request_bytes(self) -> int:
        return max(
            self.max_upload_bytes,
            self.liveness_frame_count * self.max_liveness_frame_bytes,
        ) + 64 * 1024
