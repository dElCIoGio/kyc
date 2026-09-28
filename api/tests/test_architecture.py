from __future__ import annotations

import ast
from pathlib import Path
import unittest


_API_ROOT = Path(__file__).resolve().parents[1] / "src" / "kyc_api"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _imports_banned_modules(root: Path, banned: set[str]) -> set[str]:
    found: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
            else:
                continue
            for name in names:
                root_name = name.split(".", 1)[0]
                if root_name in banned:
                    found.add(root_name)
    return found


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_engine_never_imports_api(self) -> None:
        self.assertNotIn(
            "kyc_api",
            _imports_banned_modules(_REPO_ROOT / "src" / "kyc_engine", {"kyc_api"}),
        )

    def test_domain_and_application_do_not_import_http_or_provider_runtime(self) -> None:
        banned = {"fastapi", "playwright", "nif_checker"}
        self.assertEqual(set(), _imports_banned_modules(_API_ROOT / "domain", banned))
        self.assertEqual(set(), _imports_banned_modules(_API_ROOT / "application", banned))

    def test_legacy_imports_reexport_the_canonical_objects(self) -> None:
        from kyc_api.application.sessions.store import SessionStore as CanonicalSessionStore
        from kyc_api.application.verification.service import VerificationManager
        from kyc_api.domain.nif import NifVerificationResult as CanonicalNifResult
        from kyc_api.domain.verification import NifVerificationStatus as CanonicalNifStatus
        from kyc_api.infrastructure.webhooks.outbox import WebhookEvent as CanonicalWebhookEvent
        from kyc_api.main import app, create_app
        from kyc_api.models import NifVerificationStatus
        from kyc_api.nif import NifVerificationResult
        from kyc_api.sessions import SessionStore
        from kyc_api.verification import VerificationManager as LegacyVerificationManager
        from kyc_api.webhooks import WebhookEvent

        self.assertIs(CanonicalSessionStore, SessionStore)
        self.assertIs(VerificationManager, LegacyVerificationManager)
        self.assertIs(CanonicalNifResult, NifVerificationResult)
        self.assertIs(CanonicalNifStatus, NifVerificationStatus)
        self.assertIs(CanonicalWebhookEvent, WebhookEvent)
        self.assertTrue(callable(create_app))
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
