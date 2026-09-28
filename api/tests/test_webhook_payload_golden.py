from __future__ import annotations

import json
from pathlib import Path
import unittest

from kyc_api.webhooks import WebhookEvent


_FIXTURE = Path(__file__).with_name("fixtures") / "webhook_payload_golden.json"


class WebhookPayloadGoldenTests(unittest.TestCase):
    def test_serialized_payloads_and_signature_inputs_match_pre_refactor_goldens(self) -> None:
        fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        for case in fixture["cases"]:
            expected = json.loads(case["payload_utf8"])
            data = expected["data"]
            event = WebhookEvent(
                event_id=case["event_id"],
                session_id=data["session_id"],
                sequence=case["sequence"],
                event_type=case["name"],
                session_status=data.get("status", "in_progress"),
                next_action=data.get("next_action"),
                result_available=data.get("result_available", False),
                created_at=expected["created_at"],
                nif_verification=data.get("nif_verification"),
            )
            payload = event.payload()
            self.assertEqual(case["payload_utf8"].encode(), payload)
            self.assertEqual(expected, json.loads(payload))
            self.assertEqual(case["name"], json.loads(payload)["type"])
            self.assertEqual(case["sequence"], json.loads(payload)["data"]["sequence"])
            self.assertEqual(
                case["signature_input_utf8"].encode(),
                case["signature_timestamp"].encode() + b"." + payload,
            )


if __name__ == "__main__":
    unittest.main()
