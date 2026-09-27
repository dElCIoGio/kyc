from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kyc_api.models import DocumentStatus, FaceMatchStatus, LivenessStatus
from kyc_api.sessions import SessionStore
from kyc_api.webhooks import WebhookEvent, WebhookOutbox


class TerminalWebhookContractTests(unittest.TestCase):
    def test_outbox_migrates_the_old_single_event_per_sequence_schema(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.sqlite"
            connection = sqlite3.connect(path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE webhook_events (
                        event_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        payload BLOB NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        next_attempt_at REAL NOT NULL,
                        delivered_at REAL,
                        dead_at REAL,
                        UNIQUE(session_id, sequence)
                    );
                    CREATE INDEX webhook_events_due
                        ON webhook_events(next_attempt_at, session_id, sequence);
                    """
                )
                connection.commit()
            finally:
                connection.close()

            WebhookOutbox(path, retention_seconds=60)
            connection = sqlite3.connect(path)
            try:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(webhook_events)")
                }
            finally:
                connection.close()
            self.assertIn("event_type", columns)

    def test_required_subsystem_failures_create_public_aggregate_failure_events(self) -> None:
        base = SessionStore(ttl_seconds=60, max_sessions=1).create()
        face_base = SessionStore(
            ttl_seconds=60, max_sessions=1, face_match_enabled=True
        ).create()
        failures = (
            replace(base, document=replace(base.document, status=DocumentStatus.FAILED)),
            replace(
                base,
                document=replace(base.document, status=DocumentStatus.PASSED),
                liveness_status=LivenessStatus.FAILED,
            ),
            replace(
                face_base,
                document=replace(face_base.document, status=DocumentStatus.PASSED),
                liveness_status=LivenessStatus.PASSED,
                face_match_status=FaceMatchStatus.FAILED,
            ),
        )

        for snapshot in failures:
            event = WebhookEvent.from_snapshot(
                snapshot, transition_reason="verification.failed"
            )
            assert event is not None
            payload = json.loads(event.payload())
            self.assertEqual("verification.processing.failed", payload["type"])
            self.assertEqual("failed", payload["data"]["status"])
            self.assertIsNone(payload["data"]["next_action"])

    def test_subsystem_and_aggregate_events_coexist_once_in_transition_order(self) -> None:
        snapshot = SessionStore(ttl_seconds=60, max_sessions=1).create()
        failed = replace(
            snapshot,
            document=replace(snapshot.document, status=DocumentStatus.FAILED),
            event_sequence=7,
        )
        subsystem = WebhookEvent.from_snapshot(failed, transition_reason="document.failed")
        aggregate = WebhookEvent.from_snapshot(failed, transition_reason="verification.failed")
        assert subsystem is not None and aggregate is not None

        with TemporaryDirectory() as directory:
            outbox = WebhookOutbox(Path(directory) / "outbox.sqlite", retention_seconds=60)
            outbox.enqueue(subsystem)
            outbox.enqueue(aggregate)
            outbox.enqueue(aggregate)

            first = outbox.due()
            assert first is not None
            self.assertEqual("verification.document.failed", json.loads(first.payload)["type"])
            outbox.delivered(first.event_id)

            second = outbox.due()
            assert second is not None
            self.assertEqual("verification.processing.failed", json.loads(second.payload)["type"])
            outbox.delivered(second.event_id)
            self.assertIsNone(outbox.due())


if __name__ == "__main__":
    unittest.main()
