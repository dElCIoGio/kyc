from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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

    @property
    def max_request_bytes(self) -> int:
        return self.max_upload_bytes + 64 * 1024
