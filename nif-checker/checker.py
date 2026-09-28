import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)


class NIFVerificationStatus(str, Enum):
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    PORTAL_ERROR = "portal_error"


@dataclass(frozen=True)
class NIFVerificationResult:
    status: NIFVerificationStatus
    queried_nif: str

    nif: Optional[str] = None
    name: Optional[str] = None
    taxpayer_type: Optional[str] = None
    taxpayer_status: Optional[str] = None
    defaulting: Optional[str] = None
    vat_regime: Optional[str] = None
    fiscal_residence: Optional[str] = None

    error: Optional[str] = None

    @property
    def verified(self) -> bool:
        return self.status == NIFVerificationStatus.VERIFIED


class AGTNIFVerifier:
    """
    Verifies an Angolan NIF through the AGT taxpayer portal.

    Usage:

        verifier = AGTNIFVerifier(headless=False)

        await verifier.start()

        result = await verifier.verify("007096754LA043")

        print(result)

        await verifier.close()

    Or:

        async with AGTNIFVerifier() as verifier:
            result = await verifier.verify("007096754LA043")
    """

    URL = (
        "https://portaldocontribuinte.minfin.gov.ao/"
        "consultar-nif-do-contribuinte"
    )

    # IDs contain ":" because this is a JSF / PrimeFaces page.
    NIF_INPUT_SELECTOR = '#j_id_2x\\:txtNIFNumber'
    SEARCH_BUTTON_SELECTOR = '#j_id_2x\\:j_id_34'

    RESULT_NIF_SELECTOR = "#taxPayerNidId"

    NOT_FOUND_SELECTOR = ".ui-growl-item .ui-growl-message p"

    def __init__(
        self,
        *,
        headless: bool = True,
        timeout_ms: int = 20_000,
    ):
        self.headless = headless
        self.timeout_ms = timeout_ms

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self) -> None:
        """
        Start Chromium and open the AGT portal.
        """

        if self._browser is not None:
            return

        self._playwright = await async_playwright().start()

        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
        )

        self._context = await self._browser.new_context()

        self._page = await self._context.new_page()

        self._page.set_default_timeout(self.timeout_ms)

        await self._load_portal()

    async def close(self) -> None:
        """
        Cleanly close Playwright resources.
        """

        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self._page = None

    async def verify(self, nif: str) -> NIFVerificationResult:
        """
        Search the AGT portal for a NIF.

        Returns:
            VERIFIED:
                AGT returned taxpayer information.

            NOT_FOUND:
                AGT displayed "NIF não encontrado".

            PORTAL_ERROR:
                Network, browser, timeout, DOM or unexpected portal error.
        """

        nif = self._normalise_nif(nif)

        if not self._page:
            raise RuntimeError(
                "AGTNIFVerifier has not been started. "
                "Call start() or use 'async with'."
            )

        page = self._page

        try:
            # Make sure we're still on a usable portal page.
            await self._ensure_portal_ready()

            nif_input = page.locator(self.NIF_INPUT_SELECTOR)

            await nif_input.wait_for(state="visible")

            # Clear whatever was entered previously.
            await nif_input.fill("")
            await nif_input.fill(nif)

            # Click PrimeFaces search button.
            await page.locator(
                self.SEARCH_BUTTON_SELECTOR
            ).click()

            outcome = await self._wait_for_result()

            if outcome == "not_found":
                return NIFVerificationResult(
                    status=NIFVerificationStatus.NOT_FOUND,
                    queried_nif=nif,
                )

            if outcome == "success":
                return await self._extract_result(nif)

            return NIFVerificationResult(
                status=NIFVerificationStatus.PORTAL_ERROR,
                queried_nif=nif,
                error="Unexpected AGT response.",
            )

        except PlaywrightTimeoutError:
            return NIFVerificationResult(
                status=NIFVerificationStatus.PORTAL_ERROR,
                queried_nif=nif,
                error=(
                    "Timed out waiting for a response from "
                    "the AGT portal."
                ),
            )

        except Exception as exc:
            return NIFVerificationResult(
                status=NIFVerificationStatus.PORTAL_ERROR,
                queried_nif=nif,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _load_portal(self) -> None:
        assert self._page is not None

        await self._page.goto(
            self.URL,
            wait_until="domcontentloaded",
            timeout=self.timeout_ms,
        )

        await self._page.locator(
            self.NIF_INPUT_SELECTOR
        ).wait_for(
            state="visible",
            timeout=self.timeout_ms,
        )

    async def _ensure_portal_ready(self) -> None:
        """
        Reload the page if the input is missing, for example if the
        session expired or the site navigated somewhere unexpected.
        """

        assert self._page is not None

        input_locator = self._page.locator(
            self.NIF_INPUT_SELECTOR
        )

        try:
            if await input_locator.is_visible():
                return
        except Exception:
            pass

        await self._load_portal()

    async def _wait_for_result(self) -> str:
        """
        Wait for whichever happens first:

        1. Successful taxpayer data appears.
        2. Temporary "NIF não encontrado" growl appears.

        The growl only exists briefly, so we start waiting for it
        immediately after clicking Search.
        """

        assert self._page is not None

        page = self._page

        success_task = asyncio.create_task(
            self._wait_for_success()
        )

        not_found_task = asyncio.create_task(
            self._wait_for_not_found()
        )

        done, pending = await asyncio.wait(
            {success_task, not_found_task},
            timeout=self.timeout_ms / 1000,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if not done:
            raise PlaywrightTimeoutError(
                "AGT portal did not return a recognised response."
            )

        completed = done.pop()

        try:
            return completed.result()
        finally:
            # Consume/cancel remaining task cleanly.
            for task in pending:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

    async def _wait_for_success(self) -> str:
        assert self._page is not None

        locator = self._page.locator(
            self.RESULT_NIF_SELECTOR
        )

        await locator.wait_for(
            state="visible",
            timeout=self.timeout_ms,
        )

        # Don't accept an empty result left over while AJAX is updating.
        await self._page.wait_for_function(
            """
            () => {
                const element =
                    document.querySelector('#taxPayerNidId');

                return element &&
                    element.textContent &&
                    element.textContent.trim().length > 0;
            }
            """,
            timeout=self.timeout_ms,
        )

        return "success"

    async def _wait_for_not_found(self) -> str:
        assert self._page is not None

        locator = self._page.locator(
            self.NOT_FOUND_SELECTOR
        ).filter(
            has_text="NIF não encontrado"
        )

        await locator.wait_for(
            state="visible",
            timeout=self.timeout_ms,
        )

        return "not_found"

    async def _extract_result(
        self,
        queried_nif: str,
    ) -> NIFVerificationResult:
        """
        Parse the taxpayer information returned inside panelNIF_content.
        """

        assert self._page is not None

        returned_nif = await self._text(
            self.RESULT_NIF_SELECTOR
        )

        name = await self._field_value("Nome")
        taxpayer_type = await self._field_value("Tipo")
        taxpayer_status = await self._field_value("Estado")
        defaulting = await self._field_value("Inadimplente")
        vat_regime = await self._field_value("Regime de IVA")

        fiscal_residence = (
            await self._extract_fiscal_residence()
        )

        return NIFVerificationResult(
            status=NIFVerificationStatus.VERIFIED,
            queried_nif=queried_nif,
            nif=returned_nif,
            name=name,
            taxpayer_type=taxpayer_type,
            taxpayer_status=taxpayer_status,
            defaulting=defaulting,
            vat_regime=vat_regime,
            fiscal_residence=fiscal_residence,
        )

    async def _field_value(
        self,
        field_name: str,
    ) -> Optional[str]:
        """
        Given HTML such as:

            <div class="form-group">
                <label>Nome:</label>
                <div>
                    <label>KYESI...</label>
                </div>
            </div>

        returns:

            KYESI...
        """

        assert self._page is not None

        field = self._page.locator(
            "#panelNIF_content .form-group"
        ).filter(
            has=self._page.locator(
                "label",
                has_text=field_name,
            )
        )

        if await field.count() == 0:
            return None

        value = field.first.locator(
            "div.col-sm-6 label"
        )

        if await value.count() == 0:
            return None

        text = await value.first.inner_text()

        text = text.strip()

        return text or None

    async def _extract_fiscal_residence(
        self,
    ) -> Optional[str]:
        """
        The supplied AGT HTML does not give the fiscal residence
        field a useful label. It appears as the final form-group:

            <label><!-- Indicador de Não Residente: --></label>
            ...
            <label>Residente Fiscal</label>

        So we detect the known possible returned text instead.
        """

        assert self._page is not None

        values = self._page.locator(
            "#panelNIF_content "
            ".form-group div.col-sm-6 label.control-label"
        )

        count = await values.count()

        for index in range(count):
            text = (
                await values.nth(index).inner_text()
            ).strip()

            normalised = text.casefold()

            if "residente fiscal" in normalised:
                return text

            if "não residente" in normalised:
                return text

        return None

    async def _text(
        self,
        selector: str,
    ) -> Optional[str]:
        assert self._page is not None

        locator = self._page.locator(selector)

        if await locator.count() == 0:
            return None

        value = (
            await locator.first.inner_text()
        ).strip()

        return value or None

    @staticmethod
    def _normalise_nif(nif: str) -> str:
        if not isinstance(nif, str):
            raise TypeError("NIF must be a string.")

        # IDs/NIFs such as:
        # 007096754LA043
        #
        # Keep letters and numbers while removing accidental
        # spaces or separators coming from OCR.
        value = "".join(
            char
            for char in nif.strip().upper()
            if char.isalnum()
        )

        if not value:
            raise ValueError("NIF cannot be empty.")

        return value