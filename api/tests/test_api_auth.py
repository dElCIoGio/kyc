import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from kyc_api.main import create_app
from kyc_api.sessions import SessionStore

from helpers import API_KEY, FakeCoordinator, settings


class ApiAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(
            create_app(settings=settings(), coordinator=FakeCoordinator())
        )
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_health_is_public_and_safe(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["status"])

    def test_missing_api_key_is_rejected(self) -> None:
        response = self.client.post("/v1/sessions")
        self.assertEqual(401, response.status_code)
        self.assertEqual("UNAUTHORIZED", response.json()["error"]["code"])

    def test_invalid_api_key_is_rejected_without_echoing_it(self) -> None:
        response = self.client.post(
            "/v1/sessions", headers={"X-API-Key": "wrong-secret-value"}
        )
        self.assertEqual(401, response.status_code)
        self.assertNotIn("wrong-secret-value", response.text)
        self.assertNotIn(API_KEY, response.text)

    def test_missing_model_manifest_fails_application_startup(self) -> None:
        configured = settings(ocr_model_manifest=Path("missing-model-manifest.json"))
        with self.assertRaises(ValueError):
            with TestClient(create_app(settings=configured)):
                pass

    def test_unexpected_errors_do_not_expose_exception_details(self) -> None:
        class ExplodingStore(SessionStore):
            def create(self):
                raise RuntimeError("private-path-and-ocr-value")

        store = ExplodingStore(ttl_seconds=60, max_sessions=2)
        with TestClient(
            create_app(
                settings=settings(),
                coordinator=FakeCoordinator(),
                session_store=store,
            ),
            raise_server_exceptions=False,
        ) as client:
            response = client.post(
                "/v1/sessions", headers={"X-API-Key": API_KEY}
            )
            self.assertEqual(500, response.status_code)
            self.assertEqual("INTERNAL_ERROR", response.json()["error"]["code"])
            self.assertNotIn("private-path-and-ocr-value", response.text)


if __name__ == "__main__":
    unittest.main()
