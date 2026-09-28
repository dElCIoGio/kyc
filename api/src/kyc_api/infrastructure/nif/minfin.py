"""MINFIN-specific adapter; Playwright remains in the external nif-checker package."""

from __future__ import annotations

import asyncio
import logging

from ...domain.nif import (
    NifVerificationResult,
    _name_match,
    _normalize_nif,
    unavailable_nif_result,
)
from ...domain.verification import NifVerificationStatus


logger = logging.getLogger("kyc_api.nif")


class MinfinNifVerifier:
    """Adapter that keeps checker-specific records and errors private."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self._timeout_ms = round(timeout_seconds * 1000)
        self._checker = None

    async def verify(self, nif: str, claimed_name: str | None = None) -> NifVerificationResult:
        normalized = _normalize_nif(nif)
        try:
            checker = await self._checker_instance()
            raw = await checker.verify(normalized)
        except (TimeoutError, asyncio.TimeoutError):
            return unavailable_nif_result("NIF_TIMEOUT")
        except RuntimeError as exc:
            logger.error(
                "MINFIN verifier could not be initialized",
                extra={
                    "event": "nif.verification_failed",
                    "provider": "minfin",
                    "error_code": "NIF_VERIFIER_INITIALIZATION_FAILED",
                    "exception_type": type(exc).__name__,
                },
            )
            return NifVerificationResult(
                NifVerificationStatus.FAILED,
                "minfin",
                error_code="NIF_VERIFIER_INITIALIZATION_FAILED",
            )
        except Exception as exc:
            logger.warning(
                "MINFIN verifier was unavailable",
                extra={
                    "event": "nif.verification_unavailable",
                    "provider": "minfin",
                    "error_code": "NIF_PROVIDER_UNAVAILABLE",
                    "exception_type": type(exc).__name__,
                },
            )
            return unavailable_nif_result("NIF_PROVIDER_UNAVAILABLE")

        status = getattr(raw.status, "value", str(raw.status))
        if status == "not_found":
            return NifVerificationResult(NifVerificationStatus.NOT_FOUND, "minfin")
        if status != "verified":
            return unavailable_nif_result("NIF_PROVIDER_UNAVAILABLE")

        returned = getattr(raw, "nif", None)
        try:
            if not returned or _normalize_nif(returned) != normalized:
                return NifVerificationResult(
                    NifVerificationStatus.FAILED,
                    "minfin",
                    error_code="NIF_PROVIDER_IDENTIFIER_MISMATCH",
                )
        except (TypeError, ValueError):
            return NifVerificationResult(
                NifVerificationStatus.FAILED,
                "minfin",
                error_code="NIF_PROVIDER_MALFORMED_RESPONSE",
            )
        return NifVerificationResult(
            NifVerificationStatus.VERIFIED,
            "minfin",
            name_match=_name_match(claimed_name, getattr(raw, "name", None)),
        )

    async def close(self) -> None:
        if self._checker is not None:
            await self._checker.close()
            self._checker = None

    async def reset_after_failure(self) -> None:
        """Discard browser-affine state only after an operational failure."""
        await self.close()

    async def _checker_instance(self):
        if self._checker is None:
            try:
                from nif_checker import AGTNIFVerifier
            except ImportError as exc:
                raise RuntimeError("MINFIN verifier dependency is unavailable") from exc
            self._checker = AGTNIFVerifier(headless=True, timeout_ms=self._timeout_ms)
            await self._checker.start()
        return self._checker
