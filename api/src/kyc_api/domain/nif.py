from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Protocol

from .verification import NifVerificationStatus


@dataclass(frozen=True)
class NifVerificationResult:
    """Internal verifier result; duration is intentionally never serialized."""

    status: NifVerificationStatus
    source: str | None
    name_match: bool | None = None
    error_code: str | None = None
    duration_seconds: float = 0.0


class NifVerifier(Protocol):
    async def verify(
        self, nif: str, claimed_name: str | None = None
    ) -> NifVerificationResult: ...

    async def close(self) -> None: ...


def unavailable_nif_result(code: str) -> NifVerificationResult:
    return NifVerificationResult(NifVerificationStatus.UNAVAILABLE, "minfin", error_code=code)


def normalize_nif_input(value: str) -> str:
    """Validate and normalize caller input before a dispatcher claim."""
    return _normalize_nif(value)


def _normalize_nif(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("NIF must be a string")
    normalized = "".join(char for char in value.strip().upper() if char.isalnum())
    if not normalized:
        raise ValueError("NIF must not be empty")
    return normalized


def _name_match(claimed_name: str | None, registry_name: str | None) -> bool | None:
    claimed = _name_key(claimed_name)
    registry = _name_key(registry_name)
    if claimed is None or registry is None:
        return None
    return claimed == registry


def _name_key(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    compact = "".join(char for char in decomposed if char.isalnum())
    return compact or None
