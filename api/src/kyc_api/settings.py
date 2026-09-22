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
    max_upload_bytes: int = Field(default=15 * 1024 * 1024, gt=0)
    session_ttl_seconds: int = Field(default=30 * 60, gt=0)
    max_sessions: int = Field(default=100, gt=0)
    job_workers: int = Field(default=1, gt=0)
