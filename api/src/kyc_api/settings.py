from __future__ import annotations

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
    session_ttl_seconds: int = Field(default=30 * 60, gt=0)
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

    @property
    def webhook_enabled(self) -> bool:
        return self.webhook_url is not None

    @property
    def max_request_bytes(self) -> int:
        return self.max_upload_bytes + 64 * 1024
