from __future__ import annotations

from datetime import timedelta
from urllib.parse import urlsplit
import unittest

from fastapi.testclient import TestClient

from kyc_api.main import create_app
from kyc_api.sessions import _now

from helpers import AUTH_HEADERS, FakeCoordinator, PNG_BYTES, settings


class BrowserCredentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TestClient(create_app(settings=settings(), coordinator=FakeCoordinator()))
        self.client = self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)

    def _issued(self) -> tuple[str, str]:
        session_id = self.client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
        response = self.client.post(
            f"/v1/sessions/{session_id}/browser-token", headers=AUTH_HEADERS
        )
        self.assertEqual(200, response.status_code)
        parts = urlsplit(response.json()["verification_url"])
        self.assertEqual(f"/verify/{session_id}", parts.path)
        self.assertTrue(parts.fragment.startswith("bt_"))
        return session_id, parts.fragment

    @staticmethod
    def _browser_headers(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_browser_token_reads_only_its_bound_safe_session(self) -> None:
        session_id, token = self._issued()
        response = self.client.get(
            f"/v1/sessions/{session_id}", headers=self._browser_headers(token)
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(session_id, response.json()["session_id"])

        other = self.client.post("/v1/sessions", headers=AUTH_HEADERS).json()["session_id"]
        denied = self.client.get(
            f"/v1/sessions/{other}", headers=self._browser_headers(token)
        )
        self.assertEqual(401, denied.status_code)
        self.assertEqual("UNAUTHORIZED", denied.json()["error"]["code"])

    def test_browser_token_submits_a_bound_document_side(self) -> None:
        session_id, token = self._issued()
        response = self.client.post(
            f"/v1/sessions/{session_id}/images/front",
            headers=self._browser_headers(token),
            files={"image": ("front.png", PNG_BYTES, "image/png")},
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(session_id, response.json()["session_id"])
        self.assertEqual("submit_document_back", response.json()["session"]["next_action"])

    def test_browser_token_cannot_retrieve_results_or_create_sessions(self) -> None:
        session_id, token = self._issued()
        headers = self._browser_headers(token)
        self.assertEqual(401, self.client.get(f"/v1/sessions/{session_id}/result", headers=headers).status_code)
        self.assertEqual(401, self.client.post("/v1/sessions", headers=headers).status_code)
        self.assertEqual(401, self.client.delete(f"/v1/sessions/{session_id}", headers=headers).status_code)
        self.assertEqual(401, self.client.get("/v1/metrics", headers=headers).status_code)

    def test_reissue_rotates_and_delete_revokes_browser_credential(self) -> None:
        session_id, first = self._issued()
        _, second = self._issued_for(session_id)
        self.assertEqual(
            401,
            self.client.get(f"/v1/sessions/{session_id}", headers=self._browser_headers(first)).status_code,
        )
        self.assertEqual(
            200,
            self.client.get(f"/v1/sessions/{session_id}", headers=self._browser_headers(second)).status_code,
        )
        self.client.delete(f"/v1/sessions/{session_id}", headers=AUTH_HEADERS)
        self.assertEqual(
            401,
            self.client.get(f"/v1/sessions/{session_id}", headers=self._browser_headers(second)).status_code,
        )

    def _issued_for(self, session_id: str) -> tuple[str, str]:
        response = self.client.post(
            f"/v1/sessions/{session_id}/browser-token", headers=AUTH_HEADERS
        )
        return session_id, urlsplit(response.json()["verification_url"]).fragment

    def test_expired_session_keeps_safe_get_at_stable_410(self) -> None:
        session_id, token = self._issued()
        self.client.app.state.sessions._records[session_id].expires_at = _now() - timedelta(seconds=1)
        response = self.client.get(
            f"/v1/sessions/{session_id}", headers=self._browser_headers(token)
        )
        self.assertEqual(410, response.status_code)
        self.assertEqual("SESSION_EXPIRED", response.json()["error"]["code"])
        self.assertEqual(
            410,
            self.client.post(
                f"/v1/sessions/{session_id}/images/front",
                headers=self._browser_headers(token),
            ).status_code,
        )

    def test_verifier_assets_have_no_store_security_headers(self) -> None:
        response = self.client.get("/verify/0123456789abcdef")
        self.assertEqual(200, response.status_code)
        self.assertEqual("no-store", response.headers["Cache-Control"])
        self.assertEqual("no-referrer", response.headers["Referrer-Policy"])
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        self.assertIn("assets/", response.text)


if __name__ == "__main__":
    unittest.main()
